"""
Compare the per-dataset and per-pass parts of the WGSL correlation images (CorrelationImages) against masknmf, with
the real compression results U, V: the robust noise term (DemixingState._sketch_robust_variance_term) for masknmf's
frame batch sizes, the mean and normalizer of the standard correlation images (_compute_standard_correlation_image) and
the norms uv_norms of _compute_residual_correlation_image, without a ring term and with one from the WGSL fluctuating
background update of synthetic footprints and traces. The frames of the noise term are drawn from a seeded generator
and given to both.

Each quantity is compared to masknmf in float32 and to the same formulas in float64, computed from the Gram matrices
of the rows of V for the columns of U of each cell. The WGSL values must be at least as close to the float64 values
as masknmf's float32 values are, within a factor of 2.
"""

import types

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import (
    DemixingState,
    _compute_residual_correlation_image,
    _compute_standard_correlation_image,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._compression import compute_cell_tiles
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
n_frames, height, width = compression["shape"]
rank = v.shape[0]
n_pixels = height * width

gpu_compression = CompressionBuffers(dmr_path)
correlation = CorrelationImages(gpu_compression)

frames = np.random.default_rng(0).choice(n_frames, size=5000, replace=False)

# float64 references from the Gram matrices of the cells
cell_size, cell_col_ptr, cell_cols, tiles = compute_cell_tiles(u, (height, width))
n_cells_x = width // cell_size
tiles = tiles.astype(np.float64)
v64 = v.numpy().astype(np.float64)


def cell_forms(vv: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """per pixel u_p^T (vv vv^T) u_p and u_p^T vv w, vv [rank, n]"""
    x = np.zeros(n_pixels)
    y = np.zeros(n_pixels)
    for c in range(cell_col_ptr.size - 1):
        cols = cell_cols[cell_col_ptr[c] : cell_col_ptr[c + 1]]
        uc = tiles[cell_col_ptr[c] * cell_size**2 : cell_col_ptr[c + 1] * cell_size**2].reshape(cols.size, -1).T
        rows = (c // n_cells_x) * cell_size + np.arange(cell_size)[:, None]
        pixels = (rows * width + (c % n_cells_x) * cell_size + np.arange(cell_size)[None, :]).ravel()
        vc = vv[cols]
        x[pixels] = np.einsum("pk,kl,pl->p", uc, vc @ vc.T, uc)
        y[pixels] = uc @ (vc @ w)
    return x, y


def robust_noise_float64(frame_batch_size: int) -> float:
    last = np.zeros(frames.size)
    last[(-(-frames.size // frame_batch_size) - 1) * frame_batch_size :] = 1.0
    sq, last_sum = cell_forms(v64[:, frames], last)
    return float(np.quantile(np.sqrt(np.maximum(sq / frames.size - (last_sum / frames.size) ** 2, 0.0)), 0.1))


def robust_noise_masknmf(frame_batch_size: int) -> float:
    """DemixingState._sketch_robust_variance_term on the attributes it reads, np.random.choice returning frames"""
    state = types.SimpleNamespace(
        v=v,
        u_sparse=u,
        device="cpu",
        frame_batch_size=frame_batch_size,
        factorized_ring_term=None,
        pmd_obj=types.SimpleNamespace(shape=(n_frames, height, width)),
    )
    choice = np.random.choice
    np.random.choice = lambda *args, **kwargs: frames
    try:
        noise = DemixingState._sketch_robust_variance_term(state)
    finally:
        np.random.choice = choice
    return float(noise.flatten()[0])


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


results = []
d64, uv_sum64 = cell_forms(v64, np.ones(n_frames))

for frame_batch_size in (5000, 2000):
    noise64 = robust_noise_float64(frame_batch_size)
    noise_masknmf = robust_noise_masknmf(frame_batch_size)
    noise_wgsl = correlation.set_robust_noise(frames, frame_batch_size)
    results.append(
        {
            "quantity": f"robust noise, frame_batch_size {frame_batch_size}",
            "wgsl": abs(noise_wgsl - noise64) / noise64,
            "torch_float32": abs(noise_masknmf - noise64) / noise64,
        }
    )
    print(results[-1])

# the standard correlation images with each implementation's own noise term, of the last frame batch size
mean64 = uv_sum64 / n_frames
normalizer64 = np.sqrt(-2 * uv_sum64 * mean64 + mean64**2 * n_frames + d64 + n_frames * noise64**2)
standard = _compute_standard_correlation_image(
    u,
    v,
    torch.zeros(n_frames, 1),
    (height, width),
    noise_std=torch.full((height, width), noise_masknmf),
    frame_batch_size=1000,
)
results.append(
    {
        "quantity": "std_corr_img_mean",
        "wgsl": relative_error(correlation.get_std_corr_img_mean(), mean64),
        "torch_float32": relative_error(standard.std_corr_img_mean.numpy(), mean64),
    }
)
print(results[-1])
results.append(
    {
        "quantity": "std_corr_img_normalizer",
        "wgsl": relative_error(correlation.get_std_corr_img_normalizer(), normalizer64),
        "torch_float32": relative_error(standard.std_corr_img_normalizer.numpy(), normalizer64),
    }
)
print(results[-1])

# a ring term from the fluctuating background update of synthetic footprints and traces, as in benchmark_hals.py
rng = np.random.default_rng(0)
rows, cols = np.mgrid[:height, :width]
entries = []
for i in range(600):
    r0, c0 = rng.uniform(12, height - 12), rng.uniform(12, width - 12)
    radius = rng.uniform(3, 7)
    d2 = (rows - r0) ** 2 + (cols - c0) ** 2
    mask = d2 <= radius**2
    pixels = (rows[mask] * width + cols[mask]).astype(np.int64)
    entries.append((pixels, np.full(pixels.size, i), np.exp(-d2[mask] / (2 * (radius / 2) ** 2))))
pixels, signal_indices, values = (np.concatenate(x) for x in zip(*entries))
a = torch.sparse_coo_tensor(np.stack([pixels, signal_indices]), values.astype(np.float32), (n_pixels, 600)).coalesce()
c = np.abs(rng.standard_normal((n_frames, 600))).astype(np.float32)
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32))
background = FluctuatingBaseline(gpu_compression, u)
background.set_signals(signals)
ring_rank = background.update(40)
rrp = signals.ring_rank_padded
q0 = read_buffer(signals.buffers["ring_left"], np.float32, (rank, rrp))[:, :ring_rank].copy()
q1 = read_buffer(signals.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
    :ring_rank, :n_frames
].copy()

# masknmf's uv_norms do not depend on a and c, one signal keeps its residual image cheap
a_one = torch.sparse_coo_tensor(np.array([[0], [0]]), np.array([1.0], np.float32), (n_pixels, 1)).coalesce()
u_csr = sp.csr_matrix((u.values().numpy().astype(np.float64), u.indices().numpy()), shape=(n_pixels, rank))
for name, ring_term in (("uv_norms, no ring term", None), (f"uv_norms, ring rank {ring_rank}", (q0, q1))):
    if ring_term is None:
        reference = np.sqrt(d64)
        gpu_signals = SignalBuffers(
            gpu_compression, a_one, np.zeros((n_frames, 1), np.float32), np.zeros(n_pixels, np.float32)
        )
    else:
        q0_64, q1_64 = q0.astype(np.float64), q1.astype(np.float64)
        left = u_csr @ q0_64
        right = u_csr @ (v64 @ q1_64.T)
        reference = np.sqrt(
            np.maximum(d64 - 2 * np.sum(left * right, axis=1) + np.sum((left @ (q1_64 @ q1_64.T)) * left, axis=1), 0.0)
        )
        gpu_signals = signals
    correlation.set_uv_norms(gpu_signals)
    _, uv_norms_masknmf = _compute_residual_correlation_image(
        u,
        v,
        None if ring_term is None else tuple(torch.from_numpy(q) for q in ring_term),
        a_one,
        torch.zeros(n_frames, 1),
        (height, width),
        noise_std=torch.full((n_pixels,), noise_masknmf),
        batch_size=1000,
    )
    results.append(
        {
            "quantity": name,
            "wgsl": relative_error(correlation.get_uv_norms(), reference),
            "torch_float32": relative_error(uv_norms_masknmf.numpy().ravel(), reference),
        }
    )
    print(results[-1])

df = pd.DataFrame(results)
print("errors relative to the same formulas in float64")
print(df.to_string(index=False, float_format="%.2e"))

if not (df["wgsl"] <= 2 * df["torch_float32"]).all():
    raise AssertionError("WGSL correlation image normalizers differ from masknmf")
print("passed")
