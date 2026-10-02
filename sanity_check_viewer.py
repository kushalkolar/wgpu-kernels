"""
Compare the frames of the viewer (DemixingFrames) against masknmf's PMDArray, ACArray and ResidualArray without
rescaling (DemixingResults' default, as masknmf's demixing GUIs show them), constructed from the same tensors, for 6
frames spread across the movie: without signals, with the signals of Demixer.initialize_signals (b of the
initialization, no ring term) and with the signals at the end of a demix pass (b = 0, with the ring term). masknmf on
CUDA if torch has a GPU. Without signals masknmf's arrays get one signal without entries, b = 0 and no ring term.

Asserted: the PMD, AC and residual frames are within 1e-5 of masknmf's relative to the max of the PMD frame, the scale
of the three; the traces of SignalBuffers.get_trace, which the viewer shows, are those of get_c, bitwise. Reported: the
errors of the frames and of masknmf's float32 arrays against float64, the time of a frame.
"""

import time

import numpy as np
import pandas as pd
import torch

import fastplotlib as fpl
from masknmf.compression.pmd_array import PMDArray
from masknmf.demixing.demixing_arrays.ac_array import ACArray
from masknmf.demixing.demixing_arrays.fluctuating_background_array import FluctuatingBackgroundArray
from masknmf.demixing.demixing_arrays.residual_array import ResidualArray
from masknmf.demixing.demixing_arrays.static_baseline import StaticBackgroundArray

from project import select_adapter, load_compression, CompressionBuffers
from project._demixer import Demixer
from project._viewer import DemixingFrames

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)
masknmf_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"masknmf on {torch.cuda.get_device_name() if masknmf_device == 'cuda' else 'the CPU'}")

dmr_path = "./demix_new.hdf5"
tolerance = 1e-5

compression = load_compression(dmr_path)
n_frames, height, width = compression["shape"]
u = compression["u"].coalesce().to(masknmf_device)
v = compression["v"].to(masknmf_device)
var_img = compression["var_img"].to(masknmf_device)
pmd_array = PMDArray.from_tensors(
    (n_frames, height, width),
    u,
    v,
    compression["mean_img"].to(masknmf_device),
    var_img,
    device=masknmf_device,
    rescale=False,
)
u64 = u.double()
gpu_compression = CompressionBuffers(dmr_path)
frames = DemixingFrames(gpu_compression)
frame_indices = np.linspace(0, n_frames - 1, 6).astype(int)


def compare_frames(case: str, signals) -> list[dict]:
    """the frames of DemixingFrames with signals and masknmf's arrays of the same tensors, both against float64"""
    frames.set_signals(signals)
    if signals is None:
        a = torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (height * width, 1))
        c = torch.zeros((n_frames, 1))
        b = torch.zeros(height * width)
        ring = None
    else:
        a, c, b = signals.get_a(), torch.from_numpy(signals.get_c()), torch.from_numpy(signals.get_b())
        ring = signals.get_factorized_ring_term()
    # without a ring term the zeros of DemixingState
    q0, q1 = (torch.zeros((u.shape[1], 1)), torch.zeros((1, n_frames))) if ring is None else map(torch.from_numpy, ring)
    a, c, b, q0, q1 = (x.to(masknmf_device) for x in (a, c, b, q0, q1))
    ac_array = ACArray.from_tensors((height, width), a, c, var_img)
    residual_array = ResidualArray(
        pmd_array,
        ac_array,
        FluctuatingBackgroundArray.from_tensors((height, width), u, q0, q1, var_img),
        StaticBackgroundArray.from_tensors(b.reshape(height, width), var_img),
    )

    # frame, (ours vs masknmf, ours vs float64, masknmf vs float64), (PMD, AC, residual)
    errors = np.zeros((frame_indices.size, 3, 3))
    for i, t in enumerate(frame_indices):
        frames.t = t
        ours = frames.get_frames()
        theirs = [array[t].reshape(height, width) for array in (pmd_array, ac_array, residual_array)]
        pmd64 = torch.sparse.mm(u64, v[:, [t]].double())
        ac64 = torch.sparse.mm(a.double(), c[[t]].double().T)
        residual64 = pmd64 - torch.sparse.mm(u64, q0.double() @ q1[:, [t]].double()) - ac64 - b.double()[:, None]
        exact = [x.reshape(height, width).cpu().numpy() for x in (pmd64, ac64, residual64)]
        scale = np.abs(theirs[0]).max()
        for j in range(3):
            errors[i, :, j] = [
                np.abs(ours[j] - theirs[j]).max() / scale,
                np.abs(ours[j] - exact[j]).max() / scale,
                np.abs(theirs[j] - exact[j]).max() / scale,
            ]
    errors = errors.max(axis=0)
    return [
        {
            "case": case,
            "frame": name,
            "vs masknmf": errors[0, j],
            "ours vs float64": errors[1, j],
            "masknmf vs float64": errors[2, j],
        }
        for j, name in enumerate(("PMD", "AC", "residual"))
    ]


results = compare_frames("no signals", None)
demixer = Demixer(gpu_compression, compression)
signals = demixer.initialize_signals()
results += compare_frames(f"initialized, {signals.n_signals} signals", signals)
start = time.perf_counter()
for _ in demixer.demix(signals):
    pass
print(f"demix pass: {time.perf_counter() - start:.1f} s, ring rank {demixer.signals.ring_rank}")
signals = demixer.signals
results += compare_frames(f"after a pass, {signals.n_signals} signals", signals)
table = pd.DataFrame(results)
print("frames relative to the max of masknmf's PMD frame, max over the frames")
print(table.to_string(index=False, float_format="%.2e"))

start = time.perf_counter()
for t in range(100):
    frames.t = t
frames.get_frames()
print(f"frame: {(time.perf_counter() - start) * 10:.2f} ms")

c = signals.get_c()
same_traces = all(np.array_equal(signals.get_trace(i), c[:, i]) for i in (0, signals.n_signals // 2, signals.n_signals - 1))
print(f"get_trace equal to get_c: {same_traces}")

if not ((table["vs masknmf"] <= tolerance).all() and same_traces):
    raise AssertionError("the viewer's frames differ from masknmf's arrays or get_trace from get_c")
print("passed")
