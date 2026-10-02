"""
Compare the WGSL deletion of the support update (CorrelationImages.get_signals_to_keep, SignalBuffers.index_select)
against masknmf's _flag_components_for_deletion and the index_select of a, mask_ab and c in support_update_routine,
with the real compression results U, V and the signals of sanity_check_residual.py after the WGSL merge, with a ring
term of rank 40 from the WGSL fluctuating background update. The deletion thresholds are 0.2 and the middle of the
per-signal maxima of the support values, the min brightness None and a quarter of the way up the brightnesses.

The signals to keep must be the same as masknmf's, and the brightnesses max |a| max |c| identical. The signals where
WGSL or masknmf and float64 are on different sides of a threshold are reported with their distance to it. After a
deletion, a, c and the groups of the kept signals must be identical to masknmf's, and their residual correlation images,
with W gathered from the previous ones, at least as close to masknmf's formulas in float64 as masknmf's float32 values
are, within a factor of 2. index_select is also compared on signals with a mask_ab that differs from the support of a.
"""

import contextlib
import io
import types

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.demixing_utils import brightness_order
from masknmf.demixing.signal_demixer import (
    DemixingState,
    _compute_hals_schedule,
    _compute_residual_correlation_image,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._compression import compute_cell_tiles
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages
from project._merge import SignalMerger

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"
n_signals = 400

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
n_frames, height, width = compression["shape"]
rank = v.shape[0]
n_pixels = height * width

gpu_compression = CompressionBuffers(dmr_path)
correlation = CorrelationImages(gpu_compression)
noise = correlation.set_robust_noise(np.random.default_rng(0).choice(n_frames, size=5000, replace=False))
noise_std = torch.full((n_pixels,), noise)

# the signals of sanity_check_merge.py
rng = np.random.default_rng(1)
rows, cols = np.mgrid[:height, :width]
entries = []
centers = []
for i in range(n_signals):
    if i % 4 == 1:
        r0, c0 = centers[-1][0] + rng.uniform(-1.5, 1.5), centers[-1][1] + rng.uniform(-1.5, 1.5)
    else:
        r0, c0 = rng.uniform(12, height - 12), rng.uniform(12, width - 12)
    centers.append((r0, c0))
    radius = rng.uniform(3, 7)
    d2 = (rows - r0) ** 2 + (cols - c0) ** 2
    mask = d2 <= radius**2
    pixels = (rows[mask] * width + cols[mask]).astype(np.int64)
    entries.append((pixels, np.full(pixels.size, i), np.exp(-d2[mask] / (2 * (radius / 2) ** 2))))
pixels, signal_indices, values = (np.concatenate(x) for x in zip(*entries))
a = torch.sparse_coo_tensor(
    np.stack([pixels, signal_indices]), values.astype(np.float32), (n_pixels, n_signals)
).coalesce()
u_csr = sp.csr_matrix((u.values().numpy().astype(np.float64), u.indices().numpy()), shape=(n_pixels, rank))
a_csr = sp.csr_matrix((values, (pixels, signal_indices)), shape=(n_pixels, n_signals))
weights = np.asarray((a_csr.T @ u_csr).todense()) / np.asarray(a_csr.sum(axis=0)).T
c = (torch.from_numpy(weights.astype(np.float32)) @ v).numpy().T.copy()
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32))
merged, preserved = SignalMerger(gpu_compression).merge(signals, correlation.get_merge_pairs(signals))
n = merged.n_signals
print(f"{n} signals after the merge")

# a ring term for the merged signals, the uv_norms of the pass with it
background = FluctuatingBaseline(gpu_compression, u)
background.set_signals(merged)
background.update(40)
correlation.set_uv_norms(merged)
rrp = merged.ring_rank_padded
ring_left = read_buffer(merged.buffers["ring_left"], np.float32, (rank, rrp))
ring_right = read_buffer(merged.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))
ring_term = (ring_left[:, :40].copy(), ring_right[:40, :n_frames].copy())
ring_torch = tuple(torch.from_numpy(q) for q in ring_term)

# diag(U V V^T U^T) in float64 from the Gram matrices of the cells, for uv_norms
V = v.numpy().astype(np.float64)
cell_size, cell_col_ptr, cell_cols, tiles = compute_cell_tiles(u, (height, width))
tiles = tiles.astype(np.float64)
n_cells_x = width // cell_size
uv_sq_norms = np.zeros(n_pixels)
for cell in range(cell_col_ptr.size - 1):
    columns = cell_cols[cell_col_ptr[cell] : cell_col_ptr[cell + 1]]
    uc = tiles[cell_col_ptr[cell] * cell_size**2 : cell_col_ptr[cell + 1] * cell_size**2].reshape(columns.size, -1).T
    cell_rows = (cell // n_cells_x) * cell_size + np.arange(cell_size)[:, None]
    cell_pixels = (cell_rows * width + (cell % n_cells_x) * cell_size + np.arange(cell_size)[None, :]).ravel()
    uv_sq_norms[cell_pixels] = np.einsum("pk,kl,pl->p", uc, V[columns] @ V[columns].T, uc)


def residual_float64(a_signals: torch.Tensor, c_signals: np.ndarray, blocks) -> tuple:
    """masknmf's _compute_residual_correlation_image in float64 with the ring term: mean, normalizer, support values
    [n_entries] in the order of the entries sorted by (signal, pixel)"""
    n_signals = c_signals.shape[1]
    A = sp.csr_matrix(
        (a_signals.values().numpy().astype(np.float64), a_signals.indices().numpy()), shape=(n_pixels, n_signals)
    )
    C = c_signals.astype(np.float64)
    L, R = (q.astype(np.float64) for q in ring_term)
    v_new_sum = V.sum(axis=1) - L @ R.sum(axis=1)
    c_mean = C.mean(axis=0)
    c_bar = C - c_mean
    c_norm = np.linalg.norm(c_bar, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_tilde = np.nan_to_num(c_bar / c_norm, nan=0.0, posinf=0.0, neginf=0.0)

    mean = u_csr @ (v_new_sum / n_frames) - A @ c_mean
    norms = -2 * (u_csr @ v_new_sum) * mean + 2 * (A @ C.sum(axis=0)) * mean + n_frames * mean**2
    v_new_c = V @ C - L @ (R @ C)
    norms += -2 * np.asarray(A.multiply(u_csr @ v_new_c).sum(axis=1)).ravel()
    norms += np.asarray(A.multiply(A @ (C.T @ C)).sum(axis=1)).ravel()
    left, right = u_csr @ L, u_csr @ (V @ R.T)
    norms += uv_sq_norms - 2 * np.sum(left * right, axis=1) + np.sum((left @ (R @ R.T)) * left, axis=1)
    normalizer = np.sqrt(norms + n_frames * np.float64(noise) ** 2)

    cumulator = u_csr @ (v_new_c - np.outer(v_new_sum, c_mean)) - A @ (C.T @ c_bar) - np.outer(mean, c_bar.sum(axis=0))
    A_coo = A.tocoo()
    keys = np.sort(A_coo.col.astype(np.int64) * n_pixels + A_coo.row)
    support = np.zeros(keys.size)
    for block in blocks:
        block = block.numpy()
        a_block = A[:, block].tocoo()
        p, j = a_block.row, a_block.col
        i = block[j]
        denominator = np.sqrt(
            norms[p] + a_block.data**2 * c_norm[i] ** 2 + 2 * cumulator[p, i] * a_block.data + n_frames * noise**2
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            numerator = np.nan_to_num(cumulator[p, i] / c_norm[i], nan=0.0)
        numerator += np.asarray((A[:, block] @ (c_bar[:, block].T @ c_tilde[:, block]))[p, j]).ravel()
        support[np.searchsorted(keys, i * n_pixels + p)] = numerator / denominator
    return mean, normalizer, support


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


def get_groups(signal_buffers: SignalBuffers) -> list[list[int]]:
    """the signals of each group, in the order of masknmf's blocks"""
    s = signal_buffers.structures
    ptr = s["group_ptr"].astype(np.int64)
    return [s["group_signals"][ptr[g] : ptr[g + 1]].tolist() for g in range(ptr.size - 1)]


def is_identical(x: torch.Tensor, y: torch.Tensor) -> bool:
    """same entries and values of two coalesced sparse tensors"""
    return np.array_equal(x.indices().numpy(), y.indices().numpy()) and np.array_equal(
        x.values().numpy(), y.values().numpy()
    )


# masknmf's residual correlation images of the merged signals as its support update computes them
a_merged = merged.get_a()
c_merged = merged.get_c()
blocks = _compute_hals_schedule(a_merged.bool(), "cpu", frame_batch_size=10**9)
with contextlib.redirect_stdout(io.StringIO()):
    residual, uv_norms = _compute_residual_correlation_image(
        u,
        v,
        ring_torch,
        a_merged,
        torch.from_numpy(c_merged),
        (height, width),
        uv_norms=None,
        blocks=blocks,
        noise_std=noise_std,
        batch_size=1000,
    )

# the maxima of the support values per signal, as masknmf's _mask_expansion_routine computes them, and the
# brightnesses
a_ptr = merged.structures["a_ptr"].astype(np.int64)
has_entries = np.diff(a_ptr) > 0
support_masknmf = residual.resid_corr_img_support_values.coalesce()
max_masknmf = (
    torch.zeros(n)
    .scatter_reduce_(0, support_masknmf.indices()[1], support_masknmf.values(), "amax", include_self=False)
    .numpy()
)
_, _, support64 = residual_float64(a_merged, c_merged, blocks)
max64 = np.zeros(n)
max64[has_entries] = np.maximum.reduceat(support64, a_ptr[:-1][has_entries])
_, brightness_masknmf = brightness_order(a_merged, torch.from_numpy(c_merged))
brightness_masknmf = brightness_masknmf.numpy()
a_max64 = np.zeros(n)
np.maximum.at(a_max64, a_merged.indices()[1].numpy(), np.abs(a_merged.values().numpy()).astype(np.float64))
brightness64 = a_max64 * np.abs(c_merged).max(axis=0).astype(np.float64)

# thresholds between two values, so that none is on the threshold in float64, as Python floats like masknmf's
sorted_max = np.sort(max64[has_entries])
k = sorted_max.size // 2
sorted_brightness = np.sort(brightness64)
q = sorted_brightness.size // 4
cases = pd.DataFrame(
    {
        "deletion_threshold": [0.2, float(sorted_max[k - 1] + sorted_max[k]) / 2, 0.2],
        "min_brightness": [None, None, float(sorted_brightness[q - 1] + sorted_brightness[q]) / 2],
    },
    dtype=object,
)

keep_results = []
residual_results = []
same_brightness = True
same_selection = True
for case_index, case in enumerate(cases.itertuples(index=False)):
    name = f"deletion_threshold {case.deletion_threshold:.4f}, min_brightness {case.min_brightness}"
    print(name)
    # the residual images of the merged signals: W carried over from the merge test, then recomputed after a deletion
    correlation.update_residual(merged, preserved if case_index == 0 else np.zeros(0, dtype=np.int64))
    keep_wgsl = correlation.get_signals_to_keep(case.deletion_threshold, case.min_brightness)
    maxima = read_buffer(correlation._signal_maxima, np.float32, (n, 2))
    same_brightness &= np.array_equal(maxima[:, 1], brightness_masknmf)
    state = types.SimpleNamespace(
        residual_correlation_image=residual,
        a=a_merged,
        c=torch.from_numpy(c_merged),
        detrender=None,
        device="cpu",
        background_rank=40,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        keep_masknmf = DemixingState._flag_components_for_deletion(
            state, case.deletion_threshold, case.min_brightness
        ).numpy()
    keep64 = has_entries & (max64 > case.deletion_threshold)
    if case.min_brightness is not None:
        keep64 &= brightness64 >= case.min_brightness

    # the signals on the other side of a threshold than float64
    threshold = np.float32(case.deletion_threshold)
    for label, maxima_of_values in (("wgsl", maxima[:, 0]), ("masknmf", max_masknmf)):
        differ = (has_entries & (maxima_of_values > threshold)) != (has_entries & (max64 > case.deletion_threshold))
        distance = np.abs(max64[differ] - case.deletion_threshold).max(initial=0.0)
        print(
            f"  {label}: {differ.sum()} signals on the other side of the deletion threshold than float64, within "
            f"{distance:.2e} of it"
        )
    if case.min_brightness is not None:
        differ = (maxima[:, 1] >= np.float32(case.min_brightness)) != (brightness64 >= case.min_brightness)
        distance = np.abs(brightness64[differ] - case.min_brightness).max(initial=0.0)
        print(
            f"  wgsl and masknmf: {differ.sum()} signals on the other side of min_brightness than float64, within "
            f"{distance:.2e} of it"
        )
    keep_results.append(
        {
            "case": name,
            "kept wgsl": keep_wgsl.size,
            "kept masknmf": keep_masknmf.size,
            "kept float64": int(keep64.sum()),
            "same as masknmf": np.array_equal(keep_wgsl, keep_masknmf),
        }
    )
    if keep_wgsl.size == n:
        continue

    # the kept signals and their residual correlation images, W gathered from those of the merged signals
    kept = merged.index_select(keep_wgsl)
    keep_torch = torch.from_numpy(keep_wgsl)
    a_kept = torch.index_select(a_merged, 1, keep_torch).coalesce()
    c_kept = c_merged[:, keep_wgsl]
    blocks_kept = _compute_hals_schedule(
        torch.index_select(a_merged.bool(), 1, keep_torch).coalesce(), "cpu", frame_batch_size=10**9
    )
    selection = (
        is_identical(kept.get_a(), a_kept),
        np.array_equal(kept.get_c(), c_kept),
        get_groups(kept) == [block.tolist() for block in blocks_kept],
    )
    print(f"  kept signals identical to masknmf's (a, c, groups): {selection}")
    same_selection &= all(selection)
    correlation.update_residual(kept, keep_wgsl)
    with contextlib.redirect_stdout(io.StringIO()):
        residual_kept, _ = _compute_residual_correlation_image(
            u,
            v,
            ring_torch,
            a_kept,
            torch.from_numpy(c_kept),
            (height, width),
            uv_norms=uv_norms,
            blocks=blocks_kept,
            noise_std=noise_std,
            batch_size=1000,
        )
    kept_support_masknmf = residual_kept.resid_corr_img_support_values.coalesce()
    masknmf_rows, masknmf_cols = kept_support_masknmf.indices().numpy()
    order = np.argsort(masknmf_cols * n_pixels + masknmf_rows)
    mean64, normalizer64, kept_support64 = residual_float64(a_kept, c_kept, blocks_kept)
    for quantity, wgsl, masknmf_value, reference in (
        ("mean", correlation.get_resid_corr_img_mean(), residual_kept.resid_corr_img_mean.numpy(), mean64),
        (
            "normalizer",
            correlation.get_resid_corr_img_normalizer(),
            residual_kept.resid_corr_img_normalizer.numpy(),
            normalizer64,
        ),
        (
            "support values",
            correlation.get_resid_corr_img_support_values(),
            kept_support_masknmf.values().numpy()[order],
            kept_support64,
        ),
    ):
        residual_results.append(
            {
                "case": name,
                "quantity": quantity,
                "wgsl": relative_error(wgsl, reference),
                "torch_float32": relative_error(masknmf_value, reference),
            }
        )

# index_select with a mask_ab smaller than the support of a: the entries of a above the median of its values
above = a_merged.values() > a_merged.values().median()
mask_ab = torch.sparse_coo_tensor(
    a_merged.indices()[:, above], torch.ones(int(above.sum()), dtype=torch.bool), a_merged.shape
).coalesce()
masked = SignalBuffers(gpu_compression, a_merged, c_merged, np.zeros(n_pixels, dtype=np.float32), mask_ab=mask_ab)
subset = np.flatnonzero(np.arange(n) % 3 != 0)
selected = masked.index_select(subset)
subset_torch = torch.from_numpy(subset)
blocks_selected = _compute_hals_schedule(
    torch.index_select(mask_ab, 1, subset_torch).coalesce(), "cpu", frame_batch_size=10**9
)
selection = (
    is_identical(selected.get_a(), torch.index_select(a_merged, 1, subset_torch).coalesce()),
    np.array_equal(selected.get_c(), c_merged[:, subset]),
    get_groups(selected) == [block.tolist() for block in blocks_selected],
)
print(f"index_select with a mask_ab: {subset.size} of {n} signals identical to masknmf's (a, c, groups): {selection}")
same_selection &= all(selection)

keep_df = pd.DataFrame(keep_results)
print(keep_df.to_string(index=False))
print(f"brightnesses identical to masknmf's: {same_brightness}")
residual_df = pd.DataFrame(residual_results)
print("residual correlation images of the kept signals, errors relative to masknmf's formulas in float64")
print(residual_df.to_string(index=False, float_format="%.2e"))

if not keep_df["same as masknmf"].all() or not same_brightness:
    raise AssertionError("WGSL deletion keeps different signals than masknmf")
if not same_selection:
    raise AssertionError("WGSL index_select differs from masknmf's")
if not (residual_df["wgsl"] <= 2 * residual_df["torch_float32"]).all():
    raise AssertionError("WGSL residual correlation images of the kept signals differ from masknmf")
print("passed")
