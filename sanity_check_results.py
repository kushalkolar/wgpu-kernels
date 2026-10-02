"""
Check the export of the end-of-pass results to masknmf's DemixingResults (get_demixing_results), with the real
compression results U, V, the signals of sanity_check_merge.py and the ring term of rank 40 from the WGSL fluctuating
background update, as in the other end-of-pass checks. The reference is a DemixingResults built from masknmf's own
image routines on the same inputs (_compute_standard_correlation_image of U V and of U q0 q1,
_compute_residual_correlation_image), given the WGSL global residual correlation image and multiunit terms, which have
their own checks.

Asserted: every exported tensor is bitwise equal to the GPU data it came from; shapes, dtypes and sparse index sets
are those of the reference; masknmf's PMD, AC, background, residual and multiunit arrays and the ROI averages are
bitwise identical between the two, since they only use shared tensors; an export -> from_hdf5 round trip is bitwise.
Reported: the differences of the stored image tensors and of the standard, residual and background-to-signal
correlation image arrays from the reference, whose accuracy the component checks cover.
"""

import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.demixing_arrays import ResidCorrMode
from masknmf.demixing.demixing_results import DemixingResults
from masknmf.demixing.signal_demixer import (
    _compute_hals_schedule,
    _compute_residual_correlation_image,
    _compute_standard_correlation_image,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages
from project._local_correlation import LocalCorrelationImage
from project._results import get_demixing_results

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"
export_path = "./sanity_check_results.hdf5"
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

# the end of a pass on the GPU, then the export
correlation.set_uv_norms(signals)
correlation.get_merge_pairs(signals)
correlation.update_residual(signals)
correlation.set_background_images(signals)
local = LocalCorrelationImage(gpu_compression)
local.compute(noise, 1, "unconstrained", signals)
multiunit = background.get_multiunit_factorization()
start = time.perf_counter()
results = get_demixing_results(compression, signals, correlation, local, multiunit)
print(f"get_demixing_results: {time.perf_counter() - start:.2f} s")

# the reference from masknmf's routines
c_torch = torch.from_numpy(c)
q0_torch, q1_torch = torch.from_numpy(q0), torch.from_numpy(q1)
standard = _compute_standard_correlation_image(
    u, v, c_torch, (height, width), noise_std=torch.full((height, width), noise), frame_batch_size=1000
)
residual, _ = _compute_residual_correlation_image(
    u,
    v,
    (q0_torch, q1_torch),
    a,
    c_torch,
    (height, width),
    uv_norms=None,
    blocks=_compute_hals_schedule(a.bool(), "cpu", frame_batch_size=10**9),
    noise_std=torch.full((n_pixels,), noise),
    batch_size=1000,
)
background_images = _compute_standard_correlation_image(
    u, q0_torch @ q1_torch, c_torch, (height, width), frame_batch_size=1000
)
start = time.perf_counter()
reference = DemixingResults(
    (n_frames, height, width),
    u,
    v,
    a,
    c_torch,
    mean_img=compression["mean_img"],
    var_img=compression["var_img"],
    u_local_projector=compression["u_local_projector"],
    factorized_bkgd_term1=q0_torch,
    factorized_bkgd_term2=q1_torch,
    b=torch.zeros(n_pixels),
    std_corr_img_mean=standard.std_corr_img_mean,
    std_corr_img_normalizer=standard.std_corr_img_normalizer,
    resid_corr_img_support_values=residual.resid_corr_img_support_values,
    resid_corr_img_mean=residual.resid_corr_img_mean,
    resid_corr_img_normalizer=residual.resid_corr_img_normalizer,
    bkgd_corr_img_mean=background_images.std_corr_img_mean,
    bkgd_corr_img_normalizer=background_images.std_corr_img_normalizer,
    global_residual_correlation_image=torch.from_numpy(local.get_image()),
    multiunit_basis_term1=torch.from_numpy(multiunit[0]),
    multiunit_basis_term2=torch.from_numpy(multiunit[1]),
    device="cpu",
)
print(f"DemixingResults of the reference (its ROI averages included): {time.perf_counter() - start:.2f} s")


def same_bits(x, y) -> bool:
    """bitwise equality of two dense or sparse COO tensors or arrays, with their shapes and dtypes"""
    x, y = (torch.as_tensor(t) for t in (x, y))
    if x.is_sparse != y.is_sparse or x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.is_sparse:
        x, y = x.coalesce(), y.coalesce()
        return same_bits(x.indices(), y.indices()) and same_bits(x.values(), y.values())
    x, y = (np.ascontiguousarray(t.numpy()).reshape(-1).view(np.uint8) for t in (x, y))
    return np.array_equal(x, y)


def relative_difference(x, reference) -> float:
    """max difference relative to the max of the reference"""
    x, reference = np.asarray(x), np.asarray(reference)
    return float(np.abs(x - reference).max() / np.abs(reference).max())


# 1. the exported tensors and the GPU data, the support values from the order of the entries of a
s = signals.structures
entry_signals = np.repeat(np.arange(n_signals), np.diff(s["a_ptr"]))
order = np.lexsort((entry_signals, s["a_pixels"]))
support = torch.sparse_coo_tensor(
    np.stack([s["a_pixels"][order].astype(np.int64), entry_signals[order]]),
    correlation.get_resid_corr_img_support_values()[order],
    (n_pixels, n_signals),
)
gpu_data = pd.DataFrame(
    [
        ("u", results.u, u),
        ("v", results.v, v),
        ("mean_img", results.mean_img, compression["mean_img"]),
        ("var_img", results.var_img, compression["var_img"]),
        ("u_local_projector", results.u_local_projector, compression["u_local_projector"].coalesce()),
        ("a", results.a, a),
        ("c", results.c, c),
        ("b", results.b, np.zeros(n_pixels, dtype=np.float32)),
        ("factorized_bkgd_term1", results.factorized_bkgd_term1, q0),
        ("factorized_bkgd_term2", results.factorized_bkgd_term2, q1),
        ("std_corr_img_mean", results.std_corr_img_mean, correlation.get_std_corr_img_mean()),
        ("std_corr_img_normalizer", results.std_corr_img_normalizer, correlation.get_std_corr_img_normalizer()),
        ("resid_corr_img_support_values", results.resid_corr_img_support_values, support),
        ("resid_corr_img_mean", results.resid_corr_img_mean, correlation.get_resid_corr_img_mean()),
        ("resid_corr_img_normalizer", results.resid_corr_img_normalizer, correlation.get_resid_corr_img_normalizer()),
        ("bkgd_corr_img_mean", results.bkgd_corr_img_mean, correlation.get_bkgd_corr_img_mean()),
        ("bkgd_corr_img_normalizer", results.bkgd_corr_img_normalizer, correlation.get_bkgd_corr_img_normalizer()),
        ("global_residual_correlation_image", results.global_residual_correlation_image, local.get_image()),
        ("multiunit_basis_term1", results.multiunit_basis_term1, multiunit[0]),
        ("multiunit_basis_term2", results.multiunit_basis_term2, multiunit[1]),
    ],
    columns=["name", "exported", "source"],
)
gpu_data["identical"] = [same_bits(row.exported, row.source) for row in gpu_data.itertuples()]
print("exported tensors bitwise equal to their source")
print(gpu_data[["name", "identical"]].to_string(index=False))

# 2. shapes, dtypes and sparse index sets of everything DemixingResults serializes
layouts = []
for name in sorted(DemixingResults._serialized):
    x, r = getattr(results, name), getattr(reference, name)
    if name == "shape":
        layouts.append((name, tuple(x) == tuple(r), "", ""))
        continue
    same_layout = x.shape == r.shape and x.dtype == r.dtype and x.is_sparse == r.is_sparse
    if same_layout and x.is_sparse:
        same_layout = same_bits(x.coalesce().indices(), r.coalesce().indices())
    layouts.append((name, same_layout, tuple(x.shape), str(x.dtype)))
layouts = pd.DataFrame(layouts, columns=["name", "same_as_reference", "shape", "dtype"])
print("shapes, dtypes and sparse indices compared to the reference")
print(layouts.to_string(index=False))

# 3. masknmf's arrays of the shared tensors
frames = [0, n_frames // 2, n_frames - 1]
arrays = pd.DataFrame(
    [
        (name, same_bits(getattr(results, name)[frames], getattr(reference, name)[frames]))
        for name in (
            "pmd_array",
            "ac_array",
            "fluctuating_background_array",
            "residual_array",
            "multiunit_background_array",
        )
    ]
    # the static background is an image
    + [("static_background_array", same_bits(results.static_background_array[:], reference.static_background_array[:]))]
    + [
        (name, same_bits(getattr(results, name), getattr(reference, name)))
        for name in ("pmd_roi_averages", "fluctuating_background_roi_averages", "residual_roi_averages")
    ],
    columns=["name", "identical"],
)
print(f"arrays at frames {frames} and ROI averages compared to the reference")
print(arrays.to_string(index=False))

# 4. the round trip through an hdf5 file
if os.path.exists(export_path):
    os.remove(export_path)
start = time.perf_counter()
results.export(export_path)
loaded = DemixingResults.from_hdf5(export_path)
print(f"export and from_hdf5: {time.perf_counter() - start:.2f} s")
round_trip = []
for name in sorted(DemixingResults._serialized):
    x, y = getattr(results, name), getattr(loaded, name)
    round_trip.append(
        (name, tuple(int(i) for i in x) == tuple(int(i) for i in y) if name == "shape" else same_bits(x, y))
    )
round_trip = pd.DataFrame(round_trip, columns=["name", "identical"])
print("export -> from_hdf5")
print(round_trip.to_string(index=False))
os.remove(export_path)


# reported: the image tensors and arrays computed by the two implementations
def stored(demixing_results: DemixingResults, name: str) -> np.ndarray:
    """a stored tensor, the values of a sparse one"""
    x = getattr(demixing_results, name)
    return (x.coalesce().values() if x.is_sparse else x).numpy()


def array_difference(ours, theirs, n_items: int, batch: int = 100) -> float:
    """max difference of the images of all items, relative to the max of the reference's"""
    difference, reference_max = 0.0, 0.0
    for first in range(0, n_items, batch):
        items = list(range(first, min(first + batch, n_items)))
        x, r = ours[items], theirs[items]
        difference = max(difference, float(np.abs(x - r).max()))
        reference_max = max(reference_max, float(np.abs(r).max()))
    return difference / reference_max


differences = [
    (name, relative_difference(stored(results, name), stored(reference, name)))
    for name in (
        "std_corr_img_mean",
        "std_corr_img_normalizer",
        "resid_corr_img_mean",
        "resid_corr_img_normalizer",
        "resid_corr_img_support_values",
        "bkgd_corr_img_mean",
        "bkgd_corr_img_normalizer",
    )
]
image_arrays = {
    "standard_correlation_images": (results.standard_correlation_images, reference.standard_correlation_images),
    # DemixingResults' residual images, without the support values
    "residual_correlation_images": (results.residual_correlation_images, reference.residual_correlation_images),
    "background_to_signal_correlation_image": (
        results.background_to_signal_correlation_image,
        reference.background_to_signal_correlation_image,
    ),
}
for name, (ours, theirs) in image_arrays.items():
    differences.append((name, array_difference(ours, theirs, n_signals)))
# with the support values
ours, theirs = image_arrays["residual_correlation_images"]
ours.mode = theirs.mode = ResidCorrMode.DEFAULT
differences.append(("residual_correlation_images, mode DEFAULT", array_difference(ours, theirs, n_signals)))
differences = pd.DataFrame(differences, columns=["name", "difference"])
print("differences from the reference, relative to its max")
print(differences.to_string(index=False, float_format="%.2e"))

if not (
    gpu_data["identical"].all()
    and layouts["same_as_reference"].all()
    and arrays["identical"].all()
    and round_trip["identical"].all()
):
    raise AssertionError("the exported DemixingResults differ")
print("passed")
