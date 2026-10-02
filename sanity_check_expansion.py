"""
Compare the WGSL mask expansion of the support update (CorrelationImages.expand_masks) against masknmf's
_mask_expansion_routine, with the real compression results U, V and the signals of sanity_check_deletion.py kept at
the deletion threshold 0.2, with a ring term of rank 40 from the WGSL fluctuating background update.

The dilated supports must be the same as masknmf's sparse_dilation_routine gives, and masknmf's new masks must be its
residual correlation images above its thresholds at them. The images at the dilated pixels must be at least as close
to masknmf's formulas in float64 as masknmf's float32 values are, within a factor of 2. The entries where WGSL or
masknmf and float64 are on different sides of the threshold are reported with their distance to it, and the entries
where the WGSL and masknmf masks differ. a, mask_ab and the groups of the new signals must be what masknmf constructs
from the WGSL masks, and c unchanged.
"""

import contextlib
import io
import types

import numpy as np
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import (
    DemixingState,
    _compute_hals_schedule,
    _compute_residual_correlation_image,
    sparse_dilation_routine,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._compression import compute_cell_tiles
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages, dilate_supports
from project._merge import SignalMerger

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"
n_signals = 400
support_threshold = 0.9

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

# a ring term for the merged signals, the uv_norms of the pass with it, the deletion at 0.2
background = FluctuatingBaseline(gpu_compression, u)
background.set_signals(merged)
background.update(40)
correlation.set_uv_norms(merged)
rrp = merged.ring_rank_padded
ring_left = read_buffer(merged.buffers["ring_left"], np.float32, (rank, rrp))
ring_right = read_buffer(merged.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))
ring_term = (ring_left[:, :40].copy(), ring_right[:40, :n_frames].copy())
ring_torch = tuple(torch.from_numpy(q) for q in ring_term)
correlation.update_residual(merged, preserved)
keep = correlation.get_signals_to_keep(0.2)
kept = merged.index_select(keep)
correlation.update_residual(kept, keep)
n = kept.n_signals
print(f"{merged.n_signals} signals after the merge, {n} after the deletion")

# the WGSL expansion and its images at the dilated pixels
expanded = correlation.expand_masks(support_threshold)
a_kept = kept.get_a()
c_kept = kept.get_c()
kept_pixels, kept_signals = a_kept.indices().numpy()
order = np.lexsort((kept_pixels, kept_signals))
dilated_pixels, dilated_signals, on_support = dilate_supports(
    kept_pixels[order], kept_signals[order], n, (height, width)
)
n_dilated = dilated_pixels.size
dilated_ptr = np.concatenate([[0], np.cumsum(np.bincount(dilated_signals, minlength=n))])
values_wgsl = read_buffer(correlation._buffers["dilated_values"], np.float32, (n_dilated,)).copy()
dilated_keys = dilated_signals * n_pixels + dilated_pixels

# masknmf's dilation, residual correlation images and expansion
masknmf_rows, masknmf_cols, masknmf_signals = (
    x.numpy() for x in sparse_dilation_routine(height, width, a_kept.bool().coalesce(), expansion_radius=5)
)
same_dilation = np.array_equal(
    np.unique(masknmf_signals * n_pixels + masknmf_rows * width + masknmf_cols), dilated_keys
)
blocks = _compute_hals_schedule(a_kept.bool().coalesce(), "cpu", frame_batch_size=10**9)
with contextlib.redirect_stdout(io.StringIO()):
    residual, _ = _compute_residual_correlation_image(
        u,
        v,
        ring_torch,
        a_kept,
        torch.from_numpy(c_kept),
        (height, width),
        uv_norms=None,
        blocks=blocks,
        noise_std=noise_std,
        batch_size=1000,
    )
values_masknmf = np.empty(n_dilated, dtype=np.float32)
batch = 32
for start in range(0, n, batch):
    images = residual.getitem_tensor(slice(start, min(n, start + batch))).numpy()
    sl = slice(dilated_ptr[start], dilated_ptr[min(n, start + batch)])
    values_masknmf[sl] = images[dilated_signals[sl] - start, dilated_pixels[sl] // width, dilated_pixels[sl] % width]
state = types.SimpleNamespace(frame_batch_size=100, device="cpu", shape=(height, width, n_frames))
state.greedy_connected_comps = types.MethodType(DemixingState.greedy_connected_comps, state)
mask_masknmf, _ = DemixingState._mask_expansion_routine(
    state, support_threshold, a_kept.bool().coalesce(), a_kept, residual
)
mask_rows, mask_cols = mask_masknmf.indices().numpy()
in_mask_masknmf = np.isin(dilated_keys, mask_cols * n_pixels + mask_rows)
support_masknmf = residual.resid_corr_img_support_values.coalesce()
max_masknmf = (
    torch.zeros(n)
    .scatter_reduce_(0, support_masknmf.indices()[1], support_masknmf.values(), "amax", include_self=False)
    .numpy()
)
masknmf_consistent = np.array_equal(
    in_mask_masknmf, values_masknmf > (max_masknmf * np.float32(support_threshold))[dilated_signals]
)


def residual_float64(a_signals: torch.Tensor, c_signals: np.ndarray, blocks) -> tuple:
    """masknmf's _compute_residual_correlation_image in float64 with the ring term: mean, normalizer, support values
    [n_entries] in the order of the entries sorted by (signal, pixel), and c~"""
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
    return mean, normalizer, support, c_tilde


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

# the images at the dilated pixels in float64: the support values on the supports, elsewhere
# (u_p^T W'_i - sum_s a_ps c_s . c~_i - mean_p sum of c~_i) / normalizer_p
mean64, normalizer64, support64, c_tilde64 = residual_float64(a_kept, c_kept, blocks)
L, R = (q.astype(np.float64) for q in ring_term)
w64 = V @ c_tilde64 - L @ (R @ c_tilde64)
products64 = c_kept.astype(np.float64).T @ c_tilde64
a64 = sp.csr_matrix((a_kept.values().numpy().astype(np.float64), (kept_pixels, kept_signals)), shape=(n_pixels, n))
values64 = np.empty(n_dilated)
for i in range(n):
    p = dilated_pixels[dilated_ptr[i] : dilated_ptr[i + 1]]
    numerator = u_csr[p] @ w64[:, i] - a64[p] @ products64[:, i] - mean64[p] * c_tilde64[:, i].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        values64[dilated_ptr[i] : dilated_ptr[i + 1]] = np.where(normalizer64[p] > 0, numerator / normalizer64[p], 0.0)
values64[on_support] = support64
max64 = np.maximum.reduceat(support64, np.concatenate([[0], np.cumsum(np.bincount(kept_signals, minlength=n))])[:-1])
in_mask64 = values64 > (max64 * support_threshold)[dilated_signals]


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


# the WGSL masks, as expand_masks stored them, among the dilated entries: its images above its thresholds
mask_pixels, mask_signals = expanded._mask_entries
in_mask_wgsl = np.isin(dilated_keys, mask_signals * n_pixels + mask_pixels)
thresholds_wgsl = read_buffer(correlation._signal_maxima, np.float32, (n, 2))[:, 0] * np.float32(support_threshold)
wgsl_consistent = np.array_equal(in_mask_wgsl, values_wgsl > thresholds_wgsl[dilated_signals])
print(f"dilated supports identical to masknmf's: {same_dilation} ({n_dilated} entries)")
print(f"masknmf's masks are its images above its thresholds: {masknmf_consistent}, the WGSL ones: {wgsl_consistent}")
error_wgsl, error_masknmf = relative_error(values_wgsl, values64), relative_error(values_masknmf, values64)
print(f"images at the dilated pixels, errors relative to float64: wgsl {error_wgsl:.2e}, masknmf {error_masknmf:.2e}")
for name, in_mask in (("wgsl", in_mask_wgsl), ("masknmf", in_mask_masknmf)):
    differ = in_mask != in_mask64
    distance = np.abs(values64 - (max64 * support_threshold)[dilated_signals])[differ].max(initial=0.0)
    print(
        f"{name}: {differ.sum()} of {n_dilated} entries on the other side of the threshold than float64, within "
        f"{distance:.2e} of it, {in_mask.sum()} in the masks"
    )
print(f"masks of wgsl and masknmf differ at {np.sum(in_mask_wgsl != in_mask_masknmf)} entries")

# what masknmf constructs from the WGSL masks: a with zeros at the new pixels, the groups of the masks
mask_wgsl = torch.sparse_coo_tensor(
    np.stack([mask_pixels, mask_signals]), torch.ones(mask_pixels.size), a_kept.shape
).coalesce()
a_expected = torch.sparse_coo_tensor(
    torch.cat([a_kept.indices(), mask_wgsl.indices()], dim=1),
    torch.cat([a_kept.values(), torch.zeros(mask_pixels.size)]),
    a_kept.shape,
).coalesce()
a_wgsl = expanded.get_a()
same_a = np.array_equal(a_wgsl.indices().numpy(), a_expected.indices().numpy()) and np.array_equal(
    a_wgsl.values().numpy(), a_expected.values().numpy()
)
s = expanded.structures
groups = [
    s["group_signals"][s["group_ptr"][g] : s["group_ptr"][g + 1]].tolist() for g in range(s["group_ptr"].size - 1)
]
same_groups = groups == [block.tolist() for block in _compute_hals_schedule(mask_wgsl, "cpu", frame_batch_size=10**9)]
same_c = np.array_equal(expanded.get_c(), c_kept)
print(f"a, groups and c of the new signals as masknmf constructs them: {same_a}, {same_groups}, {same_c}")
print(f"{a_kept.values().numel()} entries of a before, {a_wgsl.values().numel()} after")

if not (same_dilation and masknmf_consistent):
    raise AssertionError("masknmf's expansion is not what this test assumes")
if error_wgsl > 2 * error_masknmf:
    raise AssertionError("WGSL residual correlation images at the dilated pixels differ from masknmf")
if not (wgsl_consistent and same_a and same_groups and same_c):
    raise AssertionError("WGSL expanded signals differ from masknmf's construction")
print("passed")
