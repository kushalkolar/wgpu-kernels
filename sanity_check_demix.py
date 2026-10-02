"""
Compare the WGSL demixing loop (Demixer.demix) against masknmf's DemixingState.demix: a pass with masknmf's defaults
(25 iterations) from the signals of sanity_check_merge.py and one more signal of one pixel, the most active one outside
their footprints, whose trace is minus the pixel's trace in U V, so that the first spatial update sets its footprint to
0 and deletes it. On the compression results cut to their first 2000 frames, with masknmf on CUDA if torch has a GPU,
and a frame_batch_size of 500 for both (masknmf's default of 5000 needs more than the 6 GB of the A1000 here, the
results do not depend on it for these signals). In lockstep: before each iteration masknmf's state is set to the WGSL
one (a, c, mask_ab and its blocks, the ring term and the background rank), then masknmf's loop body runs on it with
the random matrices of the WGSL ring model (drawn on the CPU for both) and the same frames of the robust noise term, so
that both start each iteration from the same inputs.

Asserted: the defaults of demix are masknmf's. After each iteration: the numbers of signals, the supports of a, the
groups and the background rank are masknmf's, and a, c and the ring term are within 1e-3 of masknmf's, relative to
their max (the float32 differences of one iteration are at most ~2e-5 in the component checks, errors in the steps of
the loop 1e-2 and above). c is compared by the contribution of each signal, ||a_i|| |c_i|: a footprint that the spatial
update sets nearly to 0 gives a large c_i = projection / ||a_i||^2, in which the float32 rounding of a_i is amplified.
The entries of mask_ab that differ, threshold decisions of the mask expansion whose accuracy sanity_check_expansion.py
compares to float64, are reported. At the end of the pass, from the final state: the supports of the residual
correlation images and the rank of the multiunit factorization are masknmf's, and the WGSL normalizers are 0 where
masknmf's are nan (a negative radicand); the differences of the images, of the global residual correlation image on
the last 3 rows of cells and of the multiunit factorization are reported, their accuracy is covered by the component
checks. Then the deletion of a signal whose trace is set to 0 must give masknmf's a, c, mask_ab and blocks. Last, a
pass on all frames without masknmf, the random matrices drawn on the GPU, is timed.
"""

import contextlib
import inspect
import io
import os
import time
from unittest import mock

import numpy as np
import pandas as pd
import pygfx
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.compression import PMDArray
from masknmf.demixing.background_estimation import RingModel
from masknmf.demixing.initialization_results import InitializationResults
from masknmf.demixing.signal_demixer import (
    DemixingState,
    _compute_standard_correlation_image,
    delete_comp,
    get_local_correlation_structure,
)
from masknmf.utils._serialization import load_dict, save_dict

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers
from project._background import FluctuatingBaseline
from project._correlation import CorrelationImages
from project._demixer import Demixer

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)
device = pygfx.renderers.wgpu.get_shared().device
masknmf_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"masknmf on {torch.cuda.get_device_name() if masknmf_device == 'cuda' else 'the CPU'}")

dmr_path = "./demix_new.hdf5"
n_crop_frames = 2000
crop_path = f"./demix_new_{n_crop_frames}_frames.hdf5"
n_signals = 400
frame_batch_size = 500
tolerance = 1e-3

# the compression results cut to their first n_crop_frames frames, written once
if not os.path.exists(crop_path):
    d = load_dict(dmr_path, "DemixingResults")
    save_dict(
        {
            "shape": np.array([n_crop_frames, *(int(i) for i in d["shape"][1:])]),
            "u": d["u"].coalesce(),
            "v": d["v"][:, :n_crop_frames].contiguous(),
            "pmd_mean_img": d["pmd_mean_img"],
            "pmd_var_img": d["pmd_var_img"],
            "pmd_u_projector": d["pmd_u_projector"].coalesce(),
        },
        crop_path,
        "DemixingResults",
    )
    del d

# the defaults of masknmf's demix, those of the loop body below
masknmf_defaults = {
    name: p.default
    for name, p in inspect.signature(DemixingState.demix).parameters.items()
    if p.default is not inspect.Parameter.empty
}
wgsl_defaults = {
    name: p.default
    for name, p in inspect.signature(Demixer.demix).parameters.items()
    if p.default is not inspect.Parameter.empty
}
shared_defaults = sorted(set(masknmf_defaults) & set(wgsl_defaults))
same_defaults = all(masknmf_defaults[name] == wgsl_defaults[name] for name in shared_defaults)
print(f"defaults of demix identical to masknmf's: {same_defaults} ({', '.join(shared_defaults)})")
maxiter = masknmf_defaults["maxiter"]


def make_signals(compression: dict, gpu_compression: CompressionBuffers) -> tuple[torch.Tensor, np.ndarray, int]:
    """the signals of sanity_check_merge.py and the one-pixel signal at the pixel with the largest norm of U V - mean
    outside their footprints, a [n_pixels, n_signals + 1], c [n_frames, n_signals + 1] and the pixel"""
    u = compression["u"].coalesce()
    v = compression["v"]
    _, height, width = compression["shape"]
    n_pixels = height * width
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
    u_csr = sp.csr_matrix((u.values().numpy().astype(np.float64), u.indices().numpy()), shape=(n_pixels, v.shape[0]))
    a_csr = sp.csr_matrix((values, (pixels, signal_indices)), shape=(n_pixels, n_signals))
    weights = np.asarray((a_csr.T @ u_csr).todense()) / np.asarray(a_csr.sum(axis=0)).T
    c = (torch.from_numpy(weights.astype(np.float32)) @ v).numpy().T

    images = CorrelationImages(gpu_compression)
    images.set_robust_noise(np.arange(min(5000, v.shape[1])))
    activity = images.get_std_corr_img_normalizer()
    activity[pixels] = 0
    p = int(np.argmax(activity))
    trace = -np.asarray(u_csr[p] @ v.numpy()).ravel()
    a = torch.sparse_coo_tensor(
        np.stack([np.append(pixels, p), np.append(signal_indices, n_signals)]),
        np.append(values, 1.0).astype(np.float32),
        (n_pixels, n_signals + 1),
    ).coalesce()
    c = np.ascontiguousarray(np.concatenate([c, trace[:, None]], axis=1).astype(np.float32))
    return a, c, p


compression = load_compression(crop_path)
u = compression["u"].coalesce()
v = compression["v"]
n_frames, height, width = compression["shape"]
rank = v.shape[0]
n_pixels = height * width
gpu_compression = CompressionBuffers(crop_path)
cell_size = gpu_compression.cell_size
n_frames_padded = gpu_compression.n_frames_padded
a, c, p = make_signals(compression, gpu_compression)
print(f"{n_frames} frames, one-pixel signal at row {p // width}, column {p % width}")

# the random matrices of the WGSL ring model and multiunit factorization drawn on the CPU, for masknmf's torch.randn
omegas = []
omega_rng = np.random.default_rng(5)
encode_omega = FluctuatingBaseline._encode_omega


def encode_shared_omega(self, encoder, k, k8, omega, seed):
    if omega is None:
        omega = omega_rng.standard_normal((n_frames, k)).astype(np.float32)
        omegas.append(omega)
    return encode_omega(self, encoder, k, k8, omega, seed)


def shared_randn(*size, device=None, **kwargs):
    omega = omegas.pop(0)
    if omega.shape != tuple(size):
        raise AssertionError(f"masknmf draws {tuple(size)}, the WGSL ring model drew {omega.shape}")
    return torch.from_numpy(omega).to(device)


FluctuatingBaseline._encode_omega = encode_shared_omega


def get_state(demixer: Demixer) -> dict:
    """the WGSL state as masknmf's tensors"""
    signals = demixer.signals
    a = signals.get_a()
    if signals._mask_entries is None:
        mask_indices = a.indices()
    else:
        mask_indices = torch.from_numpy(np.stack(signals._mask_entries))
    mask = torch.sparse_coo_tensor(mask_indices, torch.ones(mask_indices.shape[1]), a.shape).coalesce()
    ring = signals.get_factorized_ring_term()
    if ring is None:
        ring = (np.zeros((rank, 1), dtype=np.float32), np.zeros((1, n_frames), dtype=np.float32))
    s = signals.structures
    ptr = s["group_ptr"].astype(np.int64)
    return {
        "a": a,
        "c": signals.get_c(),
        "mask": mask,
        "ring": ring,
        "background_rank": demixer.background_rank,
        "groups": [s["group_signals"][ptr[g] : ptr[g + 1]].tolist() for g in range(ptr.size - 1)],
    }


def set_state(state: DemixingState, ours: dict):
    """masknmf's state from the WGSL one, its blocks from the mask as masknmf's update_hals_scheduler computes them"""
    state.a = ours["a"].clone().to(masknmf_device)
    state.c = torch.from_numpy(ours["c"].copy()).to(masknmf_device)
    state.mask_ab = ours["mask"].clone().to(masknmf_device)
    state.factorized_ring_term = tuple(torch.from_numpy(q.copy()).to(masknmf_device) for q in ours["ring"])
    state.background_rank = ours["background_rank"]
    state.update_hals_scheduler()
    state.standard_correlation_image.c = state.c


def relative_difference(x, reference) -> float:
    """max difference relative to the max of the reference"""
    x, reference = (t.cpu() if isinstance(t, torch.Tensor) else t for t in (x, reference))
    x, reference = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.abs(x - reference).max() / np.abs(reference).max())


probe = np.random.default_rng(1).standard_normal((n_frames, 16))


def ring_term_difference(q, q_ref) -> float:
    """max difference of q0 q1 applied to probe, relative to the max of the reference"""
    x = q[0].astype(np.float64) @ (q[1].astype(np.float64) @ probe)
    ref = q_ref[0].astype(np.float64) @ (q_ref[1].astype(np.float64) @ probe)
    return float(np.abs(x - ref).max() / np.abs(ref).max())


def contribution_difference(c, c_ref, a_ref: torch.Tensor) -> float:
    """max over the signals of ||a_i|| max |c_i - c_ref_i|, relative to the max of ||a_i|| max |c_ref_i|"""
    c, c_ref = (np.asarray(x, dtype=np.float64) for x in (c, c_ref))
    signals = a_ref.indices()[1].numpy()
    norms = np.sqrt(np.bincount(signals, a_ref.values().numpy().astype(np.float64) ** 2, minlength=c_ref.shape[1]))
    return float((norms * np.abs(c - c_ref).max(axis=0)).max() / (norms * np.abs(c_ref).max(axis=0)).max())


def get_entries(x: torch.Tensor) -> set:
    """the (pixel, signal) coordinates of the entries of a coalesced sparse tensor"""
    return set(zip(*x.indices().tolist()))


def compare(state: DemixingState, ours: dict, iteration: int) -> dict:
    """the WGSL state after an iteration and masknmf's after its loop body"""
    masknmf_a = state.a.coalesce().cpu()
    masknmf_mask = state.mask_ab.coalesce().cpu()
    masknmf_ring = tuple(q.cpu().numpy() for q in state.factorized_ring_term)
    same_n = masknmf_a.shape == ours["a"].shape
    same_support = same_n and torch.equal(masknmf_a.indices(), ours["a"].indices())
    same_ring_rank = masknmf_ring[0].shape[1] == ours["ring"][0].shape[1]
    return {
        "iteration": iteration,
        "n_signals": ours["a"].shape[1],
        "same_n_signals": same_n,
        "same_supports": same_support,
        "same_groups": [block.tolist() for block in state.blocks] == ours["groups"],
        "background_rank": ours["background_rank"],
        "same_background_rank": state.background_rank == ours["background_rank"],
        "ring_rank": ours["ring"][0].shape[1],
        "same_ring_rank": same_ring_rank,
        "a": relative_difference(ours["a"].values(), masknmf_a.values()) if same_support else np.nan,
        "c": contribution_difference(ours["c"], state.c.cpu(), masknmf_a) if same_n else np.nan,
        "ring_term": ring_term_difference(ours["ring"], masknmf_ring) if same_ring_rank else np.nan,
        "mask_entries_differing": len(get_entries(masknmf_mask) ^ get_entries(ours["mask"])) if same_n else np.nan,
    }


# masknmf's DemixingState from the same signals, the robust noise term from all frames (fewer than 5000)
pmd = PMDArray.from_tensors(
    (n_frames, height, width),
    u,
    v,
    compression["mean_img"],
    compression["var_img"],
    u_local_projector=compression["u_local_projector"],
    device=masknmf_device,
)
init = InitializationResults(a, a.bool().coalesce(), torch.from_numpy(c), torch.zeros(n_pixels, 1))
quiet = contextlib.redirect_stdout(io.StringIO())
with quiet:
    state = DemixingState(pmd, init, frame_batch_size=frame_batch_size, device=masknmf_device)
    # the start of masknmf's demix, lines 3448-3459
    state.detrender = None
    state.precompute_quantities()
    state.W = RingModel(state.shape[0], state.shape[1], masknmf_defaults["ring_radius"], state.device, state.data_order)
    state.update_hals_scheduler()
    state.initialize_standard_correlation_image()
    state.compute_residual_correlation_image()

# its schedules, lines 3461-3478, for a support_threshold of one value
support_threshold = [masknmf_defaults["support_threshold"]] * maxiter
min_brightness = masknmf_defaults["min_brightness"]
min_brightness_list = [None] * maxiter if min_brightness is None else np.linspace(0, min_brightness, maxiter)
background_enabled = False


def masknmf_iteration(iteration: int):
    """masknmf's loop body, lines 3493-3535, without reassign_background"""
    global background_enabled
    d = masknmf_defaults
    state.static_baseline_update()
    if d["ring_model_start_pt"] is not None and iteration >= d["ring_model_start_pt"]:
        background_enabled = True
    if background_enabled:
        state.fluctuating_baseline_update(downsampling_factor=d["background_downsampling_factor"])
    state.spatial_update(plot_en=False)
    state.static_baseline_update()
    state.temporal_update(denoise=False, plot_en=False, c_nonneg=d["c_nonneg"])
    if d["update_frequency"] and ((iteration + 1) % d["update_frequency"] == 0):
        state.standard_correlation_image.c = state.c
        original_shape = state.a.shape[1]
        state.merge_signals(d["merge_threshold"], d["merge_overlap_threshold"], False)
        if state.a.shape[1] < original_shape:
            state.update_hals_scheduler()
        state.support_update_routine(
            support_threshold[iteration], d["deletion_threshold"], min_brightness=min_brightness_list[iteration]
        )
        state.update_hals_scheduler()


# the lockstep pass
demixer = Demixer(gpu_compression, compression, frame_batch_size)
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32), frame_batch_size=frame_batch_size)
passes = demixer.demix(signals)
results = []
start = time.perf_counter()
for iteration in range(maxiter):
    if iteration == 0:
        before = {
            "a": a,
            "c": c,
            "mask": torch.sparse_coo_tensor(a.indices(), torch.ones(a.indices().shape[1]), a.shape).coalesce(),
            "ring": (np.zeros((rank, 1), dtype=np.float32), np.zeros((1, n_frames), dtype=np.float32)),
            "background_rank": None,
        }
    else:
        before = get_state(demixer)
    next(passes)
    set_state(state, before)
    with quiet, mock.patch("torch.randn", shared_randn):
        masknmf_iteration(iteration)
    if omegas:
        raise AssertionError(f"masknmf drew {len(omegas)} random matrices fewer than the WGSL ring model")
    ours = get_state(demixer)
    if iteration == 0:
        # no other footprint has its pixel before the first mask expansion
        one_pixel_deleted = not bool((ours["a"].indices()[0] == p).any())
        print(f"one-pixel signal deleted in the first iteration: {one_pixel_deleted}")
    results.append(compare(state, ours, iteration))
    print(f"{time.perf_counter() - start:.0f} s: {results[-1]}")

table = pd.DataFrame(results)
print(
    "WGSL demixing loop compared to masknmf's loop body from the same state: a, c (by contribution) and the ring term"
)
print("relative to their max, and the number of entries of mask_ab in only one of the two")
print(table.to_string(index=False, float_format="%.2e"))
same_columns = [name for name in table.columns if name.startswith("same_")]
loop_passed = (
    same_defaults
    and table[same_columns].all().all()
    and (table[["a", "c", "ring_term"]] <= tolerance).all().all()
    and one_pixel_deleted
)

# the end of the pass, masknmf's lines 3537-3565 on the same final state
final = get_state(demixer)
if next(passes, None) is not None:
    raise AssertionError("demix yielded more than maxiter iterations")
set_state(state, final)
with quiet, mock.patch("torch.randn", shared_randn):
    state.standard_correlation_image.c = state.c
    state.compute_residual_correlation_image()
    background_images = _compute_standard_correlation_image(
        state.u_sparse,
        state.factorized_ring_term[0] @ state.factorized_ring_term[1],
        state.c,
        (state.d1, state.d2),
        data_order=state.data_order,
        frame_batch_size=state.frame_batch_size,
        device=state.device,
    )
    term1, term2 = state.extract_multiunit_factorization()
correlation = demixer._correlation
residual = state.residual_correlation_image
masknmf_support = residual.resid_corr_img_support_values.coalesce().cpu()
s = demixer.signals.structures
entry_signals = np.repeat(np.arange(demixer.signals.n_signals), np.diff(s["a_ptr"]))
order = np.lexsort((entry_signals, s["a_pixels"]))
same_support_values = torch.equal(masknmf_support.indices(), final["a"].indices())

# the global residual correlation image on the last 3 rows of cells and the row above, as sanity_check_local_correlation
first_row = height - 3 * cell_size - 1
crop_pixels = torch.arange(first_row * width, n_pixels, device=masknmf_device)
q0, q1 = state.factorized_ring_term
with quiet:
    image_masknmf = get_local_correlation_structure(
        torch.index_select(state.u_sparse, 0, crop_pixels).coalesce(),
        state.v,
        (height - first_row, width, n_frames),
        1,
        state.robust_noise_term[first_row:],
        batch_size=10000,
        a=torch.index_select(state.a, 0, crop_pixels).coalesce(),
        c=state.c,
        fluctuating_background_term1=q0,
        fluctuating_background_term2=q1,
        sign=masknmf_defaults["sign"],
    )[1:]
term1_wgsl, term2_wgsl = demixer._multiunit
same_multiunit_rank = term1_wgsl.shape[1] == term1.shape[1]


def normalizer_difference(normalizer: np.ndarray, normalizer_masknmf: torch.Tensor, name: str) -> tuple[float, bool]:
    """the difference where masknmf's normalizer is finite, and whether the WGSL one is 0 where masknmf's is nan"""
    normalizer_masknmf = normalizer_masknmf.cpu().numpy()
    finite = np.isfinite(normalizer_masknmf)
    zero_where_nan = bool(np.all(normalizer[~finite] == 0))
    print(f"{name}: masknmf's nan at {np.count_nonzero(~finite)} pixels, the WGSL one 0 there: {zero_where_nan}")
    return relative_difference(normalizer[finite], normalizer_masknmf[finite]), zero_where_nan


resid_normalizer, resid_zero_where_nan = normalizer_difference(
    correlation.get_resid_corr_img_normalizer(), residual.resid_corr_img_normalizer, "resid_corr_img_normalizer"
)
bkgd_normalizer, bkgd_zero_where_nan = normalizer_difference(
    correlation.get_bkgd_corr_img_normalizer(), background_images.std_corr_img_normalizer, "bkgd_corr_img_normalizer"
)
outputs = pd.DataFrame(
    [
        (
            "resid_corr_img_mean",
            relative_difference(correlation.get_resid_corr_img_mean(), residual.resid_corr_img_mean),
        ),
        ("resid_corr_img_normalizer, where masknmf's is finite", resid_normalizer),
        (
            "resid_corr_img_support_values",
            (
                relative_difference(correlation.get_resid_corr_img_support_values()[order], masknmf_support.values())
                if same_support_values
                else np.nan
            ),
        ),
        (
            "bkgd_corr_img_mean",
            relative_difference(correlation.get_bkgd_corr_img_mean(), background_images.std_corr_img_mean),
        ),
        ("bkgd_corr_img_normalizer, where masknmf's is finite", bkgd_normalizer),
        (
            "global residual correlation image, last 3 rows of cells",
            relative_difference(demixer._local_correlation.get_image()[first_row + 1 :], image_masknmf),
        ),
        (
            "multiunit term1 term2",
            (
                ring_term_difference((term1_wgsl, term2_wgsl), (term1.cpu().numpy(), term2.cpu().numpy()))
                if same_multiunit_rank
                else np.nan
            ),
        ),
    ],
    columns=["output", "difference"],
)
print(
    f"end of the pass: supports of the residual images identical: {same_support_values}, multiunit ranks "
    f"{term1_wgsl.shape[1]} (wgsl), {term1.shape[1]} (masknmf)"
)
print("differences from masknmf's outputs on the same final state, relative to their max")
print(outputs.to_string(index=False, float_format="%.2e"))

# the deletion of a signal whose trace is 0, at the end of masknmf's temporal_update, lines 3105-3123
deleted_signal = 0
device.queue.write_buffer(
    demixer.signals.buffers["temporal_demixed"],
    4 * deleted_signal * n_frames_padded,
    np.zeros(n_frames_padded, dtype=np.float32),
)
set_state(state, get_state(demixer))
encoder = demixer._new_encoder()
sums = demixer._encode_c_sums(encoder)
demixer._submit(encoder)
deleted = demixer._delete_zero_signals(sums)
temp = torch.squeeze(torch.sum(state.c, dim=0) == 0).long()
with quiet:
    if torch.sum(temp):
        state.a, state.c, state.standard_correlation_image, state.mask_ab = delete_comp(
            state.a,
            state.c,
            state.standard_correlation_image,
            state.mask_ab,
            temp,
            "zero c!",
            False,
            order=state.data_order,
        )
        state.update_hals_scheduler()
after = get_state(demixer)
masknmf_a = state.a.coalesce().cpu()
zero_c = {
    "deleted": deleted and after["a"].shape[1] == final["a"].shape[1] - 1,
    "a": torch.equal(masknmf_a.indices(), after["a"].indices())
    and np.array_equal(masknmf_a.values().numpy(), after["a"].values().numpy()),
    "c": np.array_equal(state.c.cpu().numpy(), after["c"]),
    "mask_ab": torch.equal(state.mask_ab.coalesce().cpu().indices(), after["mask"].indices()),
    "blocks": [block.tolist() for block in state.blocks] == after["groups"],
}
print(f"deletion of a signal with c = 0 identical to masknmf's: {zero_c}")

# a timed pass on all frames, the random matrices drawn on the GPU
FluctuatingBaseline._encode_omega = encode_omega
del demixer, signals, gpu_compression, state, pmd
compression = load_compression(dmr_path)
gpu_compression = CompressionBuffers(dmr_path)
a, c, _ = make_signals(compression, gpu_compression)
demixer = Demixer(gpu_compression, compression, frame_batch_size)
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32), frame_batch_size=frame_batch_size)
times = []
start = time.perf_counter()
for _ in demixer.demix(signals):
    times.append(time.perf_counter() - start)
total = time.perf_counter() - start
iteration_times = np.diff(np.concatenate([[0.0], times]))
print(
    f"timed pass on {compression['shape'][0]} frames: {total:.2f} s, iterations {iteration_times.sum():.2f} s "
    f"(median {np.median(iteration_times) * 1000:.0f} ms, max {iteration_times.max() * 1000:.0f} ms), "
    f"end of the pass {total - times[-1]:.2f} s, {demixer.signals.n_signals} signals"
)

if not (
    loop_passed
    and same_support_values
    and same_multiunit_rank
    and resid_zero_where_nan
    and bkgd_zero_where_nan
    and all(zero_c.values())
):
    raise AssertionError("the WGSL demixing loop differs from masknmf's")
print("passed")
