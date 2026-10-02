"""
Compare the WGSL superpixel initialization (Demixer.initialize_signals) against masknmf's superpixel_init, with
masknmf's defaults of InitializingState and a frame_batch_size of 500 for both (as sanity_check_demix.py), masknmf on
CUDA if torch has a GPU: the first initialization on U V, and a second one on the residual of the signals of a demix
pass from the first. masknmf's functions get the WGSL local correlation image (which has its own check, masknmf
rounds its values to ~7.6e-6) and the WGSL state before each step, so that each step starts from the same inputs.

Asserted for each case: the peaks are masknmf's (find_local_peaks_2d); after the HALS update of superpixel_init
(spatial_temporal_ini_uv), a and c of the superpixels are within 1e-3 of masknmf's relative to their max, c by the
contribution of each signal ||a_i|| |c_i| (see sanity_check_demix.py); the pure superpixels selected from the WGSL
traces by successive projection are those masknmf's successive_projection selects from them; the signals returned
start with the signals, unchanged, and continue with the pure superpixels of the update; without carry_background they
have no ring term, with it the ring term of the signals. Reported: the pure superpixels masknmf selects from its own
update, and the times of initialize_signals.
"""

import contextlib
import io
import time

import numpy as np
import pandas as pd
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import (
    find_local_peaks_2d,
    search_superpixel_in_range,
    spatial_temporal_ini_uv,
    successive_projection,
    superpixel_adapter,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._demixer import Demixer

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)
masknmf_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"masknmf on {torch.cuda.get_device_name() if masknmf_device == 'cuda' else 'the CPU'}")

dmr_path = "./demix_new.hdf5"
frame_batch_size = 500
tolerance = 1e-3
# masknmf's defaults of InitializingState._initialize_signals_superpixels
mad_correlation_threshold = 0.9
min_peak_distance = 3
residual_threshold = 0.3
patch_size = (100, 100)

compression = load_compression(dmr_path)
u = compression["u"].coalesce().to(masknmf_device)
v = compression["v"].to(masknmf_device)
n_frames, height, width = compression["shape"]
n_pixels = height * width
gpu_compression = CompressionBuffers(dmr_path)
demixer = Demixer(gpu_compression, compression, frame_batch_size)
quiet = contextlib.redirect_stdout(io.StringIO())


def contribution_difference(a, c, a_ref, c_ref) -> float:
    """max over the signals of ||a_i|| max |c_i - c_ref_i|, relative to the max of ||a_i|| max |c_ref_i|"""
    c, c_ref = (np.asarray(x, dtype=np.float64) for x in (c, c_ref))
    signals = a_ref.indices()[1].numpy()
    norms = np.sqrt(np.bincount(signals, a_ref.values().numpy().astype(np.float64) ** 2, minlength=c_ref.shape[1]))
    return float((norms * np.abs(c - c_ref).max(axis=0)).max() / (norms * np.abs(c_ref).max(axis=0)).max())


def relative_difference(x, reference) -> float:
    """max difference relative to the max of the reference"""
    x, reference = np.asarray(x, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.abs(x - reference).max() / np.abs(reference).max())


def masknmf_pure(peaks: np.ndarray, c_superpixels: torch.Tensor) -> np.ndarray:
    """masknmf's pure superpixels (superpixel_init) from the traces [n_frames, n_superpixels] of the superpixels at
    peaks, indices of the superpixels"""
    peak_coords = torch.from_numpy(np.stack([peaks // width, peaks % width], axis=1)).to(masknmf_device)
    _, connectivity, _ = superpixel_adapter(peak_coords, (height, width, n_frames), "C")
    pure = []
    for i in range(-(-height // patch_size[0])):
        for j in range(-(-width // patch_size[1])):
            rows = slice(i * patch_size[0], min((i + 1) * patch_size[0], height))
            cols = slice(j * patch_size[1], min((j + 1) * patch_size[1], width))
            labels, m = search_superpixel_in_range(connectivity[rows, cols], c_superpixels)
            selected = successive_projection(m, m.shape[1], residual_threshold, device=masknmf_device)
            if len(selected) > 0:
                pure.append(labels[torch.from_numpy(selected).to(labels.device)])
    return (torch.unique(torch.hstack(pure)) - 1).cpu().numpy()


def compare_initialization(case: str, carry_background: bool) -> dict:
    """initialize_signals and masknmf's steps of superpixel_init on the same inputs"""
    previous = demixer.signals
    start = time.perf_counter()
    initialized = demixer.initialize_signals(
        mad_correlation_threshold=mad_correlation_threshold,
        min_peak_distance=min_peak_distance,
        residual_threshold=residual_threshold,
        patch_size=patch_size,
        carry_background=carry_background,
    )
    seconds = time.perf_counter() - start
    image = demixer._local_correlation.get_image()

    # the steps of initialize_signals, from the same signals and image
    signals = previous
    if previous is not None and not carry_background and previous.ring_rank > 0:
        signals = demixer._without_ring_term(previous)
    n_old = 0 if signals is None else signals.n_signals
    peaks = demixer._get_superpixels(mad_correlation_threshold, min_peak_distance)
    superpixels = demixer._update_superpixels(signals, peaks)
    pure = demixer._select_pure_superpixels(superpixels, n_old, peaks, patch_size, residual_threshold)
    a_superpixels = superpixels.get_a()
    c_superpixels = superpixels.get_c()[:, n_old:]
    new = torch.arange(n_old, superpixels.n_signals)
    a_new = torch.index_select(a_superpixels, 1, new).coalesce()

    # masknmf: the peaks, the HALS update from the same signals and peaks, the selection from the WGSL traces and from
    # its own
    with quiet:
        peak_coords, _ = find_local_peaks_2d(
            torch.from_numpy(image).to(masknmf_device),
            kernel_radius=min_peak_distance,
            correlation_cutoff=mad_correlation_threshold,
            exclude_border=True,
        )
        peaks_masknmf = (peak_coords[:, 0] * width + peak_coords[:, 1]).cpu().numpy()
        a_ini, _, _ = superpixel_adapter(peak_coords, (height, width, n_frames), "C")
        a_previous = None if signals is None else signals.get_a().to(masknmf_device)
        c_previous = None if signals is None else torch.from_numpy(signals.get_c()).to(masknmf_device)
        c_masknmf, a_masknmf = spatial_temporal_ini_uv(
            u, v, (height, width, n_frames), a_ini, a=a_previous, c=c_previous, frame_batch_size=frame_batch_size
        )
        pure_from_wgsl = masknmf_pure(peaks, torch.from_numpy(c_superpixels).to(masknmf_device))
        pure_masknmf = masknmf_pure(peaks, c_masknmf)
    a_masknmf = a_masknmf.coalesce().cpu()
    c_masknmf = c_masknmf.cpu().numpy()
    same_update_support = torch.equal(a_masknmf.indices(), a_new.indices())

    # the signals returned: the signals as they were, then the pure superpixels of the update
    a_initialized = initialized.get_a()
    c_initialized = initialized.get_c()
    a_expected = (
        torch.index_select(a_superpixels, 1, torch.from_numpy(n_old + pure)).coalesce()
        if signals is None
        else torch.cat(
            [signals.get_a(), torch.index_select(a_superpixels, 1, torch.from_numpy(n_old + pure))], dim=1
        ).coalesce()
    )
    c_expected = (
        c_superpixels[:, pure] if signals is None else np.concatenate([signals.get_c(), c_superpixels[:, pure]], axis=1)
    )
    if carry_background and previous is not None and previous.ring_rank > 0:
        same_ring_term = (
            all(
                initialized.buffers[name] is previous.buffers[name]
                for name in ("ring_left", "ring_right", "ring_right_t")
            )
            and initialized.ring_rank == previous.ring_rank
        )
    else:
        same_ring_term = initialized.ring_rank == 0
    return {
        "case": case,
        "seconds": seconds,
        "superpixels": peaks.size,
        "pure": pure.size,
        "same_peaks": np.array_equal(peaks, peaks_masknmf),
        "same_update_support": same_update_support,
        "a": relative_difference(a_new.values(), a_masknmf.values()) if same_update_support else np.nan,
        "c": contribution_difference(a_new, c_superpixels, a_masknmf, c_masknmf) if same_update_support else np.nan,
        "same_pure_from_wgsl_traces": np.array_equal(pure, pure_from_wgsl),
        "pure_masknmf_own_traces": pure_masknmf.size,
        "pure_differing_masknmf_own_traces": np.setxor1d(pure, pure_masknmf).size,
        "same_returned": torch.equal(a_initialized.indices(), a_expected.indices())
        and np.array_equal(a_initialized.values().numpy(), a_expected.values().numpy())
        and np.array_equal(c_initialized, c_expected),
        "same_ring_term": same_ring_term,
    }


results = [compare_initialization("first, U V", False)]
print(results[-1])
signals = demixer.initialize_signals()
start = time.perf_counter()
for _ in demixer.demix(signals):
    pass
print(f"demix pass: {time.perf_counter() - start:.1f} s, {signals.n_signals} -> {demixer.signals.n_signals} signals")
results.append(compare_initialization("second, residual", False))
print(results[-1])
results.append(compare_initialization("second, residual with the ring term", True))
print(results[-1])

table = pd.DataFrame(results)
print("WGSL superpixel initialization compared to masknmf's steps from the same inputs, a and c relative to max")
print(table.to_string(index=False, float_format="%.2e"))
same_columns = [name for name in table.columns if name.startswith("same_")]
if not (table[same_columns].all().all() and (table[["a", "c"]] <= tolerance).all().all()):
    raise AssertionError("the WGSL superpixel initialization differs from masknmf's")
print("passed")
