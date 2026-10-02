"""
Compare the WGSL residual correlation images (CorrelationImages.update_residual) against masknmf's
_compute_residual_correlation_image: the residual mean and normalizer of each pixel and the values of each signal's
image on its support. With the real compression results U, V and the signals of sanity_check_merge.py after the WGSL
merge: without a ring term, W carried over from the merge test, and with a ring term from the WGSL fluctuating
background update of the merged signals, W reused. Both use the same robust noise term and HALS groups.

Each quantity is compared to masknmf in float32 and to masknmf's formulas in float64. The WGSL values must be at least
as close to the float64 values as masknmf's float32 values are, within a factor of 2.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import _compute_hals_schedule, _compute_residual_correlation_image

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
print(f"{merged.n_signals} signals after the merge")

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


def residual_float64(a_merged: torch.Tensor, c_merged: np.ndarray, ring_term, blocks) -> tuple:
    """masknmf's _compute_residual_correlation_image in float64: mean, normalizer, support values [n_entries] in the
    order of the entries sorted by (signal, pixel)"""
    n = c_merged.shape[1]
    A = sp.csr_matrix(
        (a_merged.values().numpy().astype(np.float64), a_merged.indices().numpy()), shape=(n_pixels, n)
    )
    C = c_merged.astype(np.float64)
    if ring_term is None:
        L, R = np.zeros((rank, 1)), np.zeros((1, n_frames))
    else:
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


results = []
background = FluctuatingBaseline(gpu_compression, u)
for case in ("no ring term, W carried over", "ring rank 40, W reused"):
    if case.startswith("ring"):
        background.set_signals(merged)
        background.update(40)
    correlation.set_uv_norms(merged)
    correlation.update_residual(merged, preserved if case.startswith("no ring") else None)

    rrp = merged.ring_rank_padded
    ring_term = None
    if rrp > 0:
        ring_term = (
            read_buffer(merged.buffers["ring_left"], np.float32, (rank, rrp))[:, :40].copy(),
            read_buffer(merged.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
                :40, :n_frames
            ].copy(),
        )
    a_merged = merged.get_a()
    c_merged = merged.get_c()
    blocks = _compute_hals_schedule(a_merged.bool(), "cpu", frame_batch_size=10**9)
    residual, _ = _compute_residual_correlation_image(
        u,
        v,
        None if ring_term is None else tuple(torch.from_numpy(q) for q in ring_term),
        a_merged,
        torch.from_numpy(c_merged),
        (height, width),
        uv_norms=None,
        blocks=blocks,
        noise_std=torch.full((n_pixels,), noise),
        batch_size=1000,
    )
    # masknmf's support values in (pixel, signal) order, the WGSL ones in (signal, pixel) order
    support_masknmf = residual.resid_corr_img_support_values.coalesce()
    masknmf_rows, masknmf_cols = support_masknmf.indices().numpy()
    order = np.argsort(masknmf_cols * n_pixels + masknmf_rows)
    support_masknmf = support_masknmf.values().numpy()[order]

    mean64, normalizer64, support64 = residual_float64(a_merged, c_merged, ring_term, blocks)
    for name, wgsl, masknmf_value, reference in (
        ("mean", correlation.get_resid_corr_img_mean(), residual.resid_corr_img_mean.numpy(), mean64),
        (
            "normalizer",
            correlation.get_resid_corr_img_normalizer(),
            residual.resid_corr_img_normalizer.numpy(),
            normalizer64,
        ),
        ("support values", correlation.get_resid_corr_img_support_values(), support_masknmf, support64),
    ):
        results.append(
            {
                "case": case,
                "quantity": name,
                "wgsl": relative_error(wgsl, reference),
                "torch_float32": relative_error(masknmf_value, reference),
            }
        )
        print(results[-1])

df = pd.DataFrame(results)
print("errors relative to masknmf's formulas in float64")
print(df.to_string(index=False, float_format="%.2e"))

if not (df["wgsl"] <= 2 * df["torch_float32"]).all():
    raise AssertionError("WGSL residual correlation images differ from masknmf")
print("passed")
