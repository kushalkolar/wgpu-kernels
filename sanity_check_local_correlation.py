"""
Compare the WGSL local correlation image (LocalCorrelationImage) against masknmf's get_local_correlation_structure, with
the real compression results U, V, the signals of sanity_check_merge.py and the ring term of rank 40 from the WGSL
fluctuating background update, for the MAD thresholds 1 (masknmf's default, the end of a pass), 2 with the sign
positive, and 0. masknmf runs on the last 3 rows of cells and the row above them, its image compared on those 3 rows of
cells, the WGSL image computed for the whole fov.

The images are compared to masknmf in float32 and to its algorithm in float64 (exact lower medians). The WGSL values
must be at least as close to float64 as masknmf's float32 values are, within a factor of 2. The statistics of the
traces of the last row of cells (median, threshold, mean and inverse normalizer) are compared to float64 at the pixels
with kept frames (masknmf's mean is nan at the others, the WGSL one 0, the correlations 0 in both). With a threshold
of 0 the mean is over all frames, where the residual sums to nearly 0 and the float32 sums are at their rounding.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import get_local_correlation_structure

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages
from project._local_correlation import LocalCorrelationImage

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
cell_size = gpu_compression.cell_size
correlation = CorrelationImages(gpu_compression)
noise = correlation.set_robust_noise(np.random.default_rng(0).choice(n_frames, size=5000, replace=False))

# the signals of sanity_check_merge.py and a ring term for them
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
background.update(40)
rrp = signals.ring_rank_padded
q0 = read_buffer(signals.buffers["ring_left"], np.float32, (rank, rrp))[:, :40].copy()
q1 = read_buffer(signals.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
    :40, :n_frames
].copy()

# the crop: the row above the last 3 rows of cells, then those
first_row = height - 3 * cell_size - 1
crop_pixels = torch.arange(first_row * width, n_pixels)
u_crop = torch.index_select(u, 0, crop_pixels).coalesce()
a_crop = torch.index_select(a, 0, crop_pixels).coalesce()
crop_height = height - first_row

# float64: the residual traces row by row
V = v.numpy().astype(np.float64)
Q0, Q1 = q0.astype(np.float64), q1.astype(np.float64)
C_t = c.T.astype(np.float64)
A = a_csr.astype(np.float64)


def residual_row(row: int) -> np.ndarray:
    """[width, n_frames] the residual U V - a c^T - U q0 q1 of a row of pixels in float64"""
    u_row = u_csr[row * width : (row + 1) * width]
    return u_row @ V - (u_row @ Q0) @ Q1 - A[row * width : (row + 1) * width] @ C_t


def normalized_row(r: np.ndarray, mad_threshold: int, sign: str) -> tuple[np.ndarray, np.ndarray]:
    """masknmf's thresholded, normalized traces y [width, n_frames] (0 at the frames not kept) and the statistics
    (median, threshold, mean, inv) [width, 4]"""
    k = (n_frames - 1) // 2
    keep = np.ones(r.shape, dtype=bool)
    median = np.zeros(r.shape[0])
    threshold = np.zeros(r.shape[0])
    if mad_threshold != 0:
        median = np.partition(r, k, axis=1)[:, k]
        d = np.abs(r - median[:, None])
        threshold = np.partition(d, k, axis=1)[:, k] * mad_threshold
        keep = d >= threshold[:, None]
        if sign == "positive":
            keep &= r > median[:, None]
        elif sign == "negative":
            keep &= r < median[:, None]
    # nan without kept frames, as masknmf's nanmean
    with np.errstate(invalid="ignore"):
        mean = np.where(keep, r, 0).sum(axis=1) / keep.sum(axis=1)
    y = np.where(keep, r - mean[:, None], 0.0)
    divisor = np.sqrt((y**2).sum(axis=1) + n_frames * np.float64(noise) ** 2)
    inv = np.where(divisor < 1e-6, 0.0, 1.0 / divisor)
    return y * inv[:, None], np.stack([median, threshold, mean, inv], axis=1)


def image_float64(mad_threshold: int, sign: str) -> tuple[np.ndarray, np.ndarray]:
    """masknmf's local correlation image of the crop in float64, [crop_height, width], and the statistics of the last
    row of cells"""
    sums = np.zeros((crop_height, width))
    counts = np.zeros((crop_height, width))
    previous = None
    last_stats = []
    for i, row in enumerate(range(first_row, height)):
        y, row_stats = normalized_row(residual_row(row), mad_threshold, sign)
        if row >= height - cell_size:
            last_stats.append(row_stats)
        horizontal = (y[1:] * y[:-1]).sum(axis=1)
        pairs = [(i, slice(1, None), i, slice(None, -1), horizontal)]
        if previous is not None:
            pairs += [
                (i, slice(None), i - 1, slice(None), (y * previous).sum(axis=1)),
                (i, slice(1, None), i - 1, slice(None, -1), (y[1:] * previous[:-1]).sum(axis=1)),
                (i, slice(None, -1), i - 1, slice(1, None), (y[:-1] * previous[1:]).sum(axis=1)),
            ]
        for row_p, cols_p, row_q, cols_q, rho in pairs:
            sums[row_p, cols_p] += rho
            counts[row_p, cols_p] += 1
            sums[row_q, cols_q] += rho
            counts[row_q, cols_q] += 1
        previous = y
    return sums / counts, np.concatenate(last_stats)


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


local = LocalCorrelationImage(gpu_compression)
results = []
for mad_threshold, sign in ((1, "unconstrained"), (2, "positive"), (0, "unconstrained")):
    local.compute(noise, mad_threshold, sign, signals)
    image_wgsl = local.get_image()[first_row + 1 :]
    stats_wgsl = read_buffer(local._stats, np.float32, (cell_size * width, 4))
    image_masknmf = get_local_correlation_structure(
        u_crop,
        v,
        (crop_height, width, n_frames),
        mad_threshold,
        torch.full((crop_height, width), noise),
        batch_size=10000,
        a=a_crop,
        c=torch.from_numpy(c),
        fluctuating_background_term1=torch.from_numpy(q0),
        fluctuating_background_term2=torch.from_numpy(q1),
        sign=sign,
    )[1:]
    image64, stats64 = image_float64(mad_threshold, sign)
    image64 = image64[1:]
    case = f"mad_threshold {mad_threshold}, {sign}"
    results.append(
        {
            "case": case,
            "quantity": "image",
            "wgsl": relative_error(image_wgsl, image64),
            "torch_float32": relative_error(image_masknmf, image64),
        }
    )
    # the pixels with kept frames, the others have masknmf's mean nan and correlations 0
    kept = np.isfinite(stats64[:, 2])
    print(f"{case}: {np.sum(~kept)} pixels of the last row of cells without kept frames")
    for k, quantity in enumerate(("median", "threshold", "mean", "inv")):
        if mad_threshold == 0 and quantity in ("median", "threshold"):
            continue
        results.append(
            {
                "case": case,
                "quantity": f"last row of cells: {quantity}",
                "wgsl": relative_error(stats_wgsl[kept, k], stats64[kept, k]),
                "torch_float32": np.nan,
            }
        )
    print(results[-1])

df = pd.DataFrame(results)
print("errors relative to masknmf's algorithm in float64")
print(df.to_string(index=False, float_format="%.2e"))

images = df[df["quantity"] == "image"]
if not (images["wgsl"] <= 2 * images["torch_float32"]).all():
    raise AssertionError("WGSL local correlation images differ from masknmf")
print("passed")
