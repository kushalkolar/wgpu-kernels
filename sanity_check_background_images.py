"""
Compare the WGSL background-to-signal correlation images (CorrelationImages.set_background_images) against masknmf's
_compute_standard_correlation_image of U q0 q1 at the end of a pass, with the real compression results U, V and the ring
terms of ranks 40 and 9 from the WGSL fluctuating background update of the signals of sanity_check_merge.py.

The mean and normalizer of U q0 q1 at each pixel are compared to masknmf in float32 and to the same formulas in
float64. The WGSL values must be at least as close to the float64 values as masknmf's float32 values are, within a
factor of 2, at the pixels where the float64 radicand of the normalizer is positive. masknmf's normalizer is nan where
its radicand is negative, the WGSL one 0.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import _compute_standard_correlation_image

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages

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
background = FluctuatingBaseline(gpu_compression, u)
background.set_signals(signals)


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


results = []
for ring_rank in (40, 9):
    background.update(ring_rank)
    correlation.set_background_images(signals)
    rrp = signals.ring_rank_padded
    q0 = read_buffer(signals.buffers["ring_left"], np.float32, (rank, rrp))[:, :ring_rank].copy()
    q1 = read_buffer(signals.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
        :ring_rank, :n_frames
    ].copy()

    standard = _compute_standard_correlation_image(
        u, torch.from_numpy(q0) @ torch.from_numpy(q1), torch.from_numpy(c), (height, width), frame_batch_size=1000
    )

    # float64: mean = U q0 q1 1 / T, radicand = -2 (U q0 q1 1) mean + T mean^2 + diag(U q0 H q0^T U^T)
    w = u_csr @ q0.astype(np.float64)
    q1_64 = q1.astype(np.float64)
    uv_sum = w @ q1_64.sum(axis=1)
    mean64 = uv_sum / n_frames
    radicand64 = -2 * uv_sum * mean64 + n_frames * mean64**2 + np.einsum("pj,jk,pk->p", w, q1_64 @ q1_64.T, w)
    valid = radicand64 > 0
    normalizer64 = np.sqrt(np.where(valid, radicand64, 0.0))

    normalizer_wgsl = correlation.get_bkgd_corr_img_normalizer()
    normalizer_masknmf = standard.std_corr_img_normalizer.numpy()
    print(
        f"ring rank {ring_rank}: float64 radicand <= 0 at {np.sum(~valid)} pixels, WGSL normalizer 0 at "
        f"{np.sum(normalizer_wgsl == 0)}, masknmf's nan at {np.sum(np.isnan(normalizer_masknmf))}"
    )
    for quantity, wgsl, masknmf_value, reference in (
        ("mean", correlation.get_bkgd_corr_img_mean(), standard.std_corr_img_mean.numpy(), mean64),
        ("normalizer", normalizer_wgsl[valid], normalizer_masknmf[valid], normalizer64[valid]),
    ):
        results.append(
            {
                "ring_rank": ring_rank,
                "quantity": quantity,
                "wgsl": relative_error(wgsl, reference),
                "torch_float32": relative_error(masknmf_value, reference),
            }
        )

df = pd.DataFrame(results)
print("errors relative to the same formulas in float64")
print(df.to_string(index=False, float_format="%.2e"))

if not (df["wgsl"] <= 2 * df["torch_float32"]).all():
    raise AssertionError("WGSL background correlation images differ from masknmf")
print("passed")
