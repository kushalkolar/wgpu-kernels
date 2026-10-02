"""
Time each kernel of one HALS iteration (spatial and temporal update) on synthetic spatial footprints and
temporal traces with the real compression results U, V. The iteration is submitted as one command buffer,
and each dispatch is timed on the GPU with timestamps at the beginning and end of its compute pass. Also
prints the wall time of an iteration without timestamps.

For the memory-bound kernels the bytes they move are computed from the index structures, and compared to
the measured bandwidth of a copy kernel on the same device. For the matrix products the flops are
compared to the measured throughput of FMA chains.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pygfx
import torch

import fastplotlib as fpl

from project import select_adapter, CompressionBuffers
from project._hals import SignalBuffers, HALS
from project._roofline import measure_bandwidth, measure_fma_throughput, time_compute_passes

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
pygfx.renderers.wgpu.enable_wgpu_features("timestamp-query")

dmr_path = "./demix_new.hdf5"
n_signals = 1500
ring_rank = 168
n_iterations = 5

compression = CompressionBuffers(dmr_path)
height, width = compression.fov_shape

# synthetic spatial footprints: gaussian discs, overlapping
rng = np.random.default_rng(0)
rows, cols = np.mgrid[:height, :width]
entries = []
for i in range(n_signals):
    r0, c0 = rng.uniform(12, height - 12), rng.uniform(12, width - 12)
    radius = rng.uniform(3, 7)
    d2 = (rows - r0) ** 2 + (cols - c0) ** 2
    mask = d2 <= radius**2
    pixels = (rows[mask] * width + cols[mask]).astype(np.int64)
    values = np.exp(-d2[mask] / (2 * (radius / 2) ** 2)).astype(np.float32)
    entries.append((pixels, np.full(pixels.size, i), values))

pixels, signals, values = (np.concatenate(x) for x in zip(*entries))
a = torch.sparse_coo_tensor(
    np.stack([pixels, signals]), values, (height * width, n_signals)
).coalesce()
c = np.abs(rng.standard_normal((compression.n_frames, n_signals))).astype(np.float32)
b = np.zeros(height * width, dtype=np.float32)
q0 = (0.01 * rng.standard_normal((compression.rank, ring_rank))).astype(np.float32)
q1 = rng.standard_normal((ring_rank, compression.n_frames)).astype(np.float32)

gpu_signals = SignalBuffers(compression, a, c, b, factorized_ring_term=(q0, q1))
hals = HALS(compression, gpu_signals)
device = pygfx.renderers.wgpu.get_shared().device


def record_iteration(command_encoder):
    hals.spatial_update(command_encoder)
    hals.temporal_update(command_encoder=command_encoder)


# warm up, compiles the pipelines
time_compute_passes(record_iteration)

# GPU time of the dispatches, summed by shader label
timings = defaultdict(float)
spans = []
for _ in range(n_iterations):
    ms_by_label, span = time_compute_passes(record_iteration)
    for label, ms in ms_by_label.items():
        timings[label] += ms
    spans.append(span)

wall_times = []
for _ in range(n_iterations):
    t0 = time.perf_counter()
    encoder = device.create_command_encoder()
    record_iteration(encoder)
    device.queue.submit([encoder.finish()])
    device._poll_wait()
    wall_times.append(1e3 * (time.perf_counter() - t0))

# bytes moved by the memory-bound kernels, flops of the matrix products
s = gpu_signals.structures
row_bytes = compression.n_frames_padded * 4
block_col_ptr, _, _ = compression.get_block_structure()
n_block_cols = np.diff(block_col_ptr)
diff_items = hals.diff_structures["items"]
n_edges = s["neighbors"].size
n_partial_rows = hals.superblock_structures["n_partial_rows"]
rrp = gpu_signals.ring_rank_padded

work_bytes = {
    # rows of V and rows of c of each work item
    "spatial_hals_diff": (diff_items[:, 8].sum() + diff_items[:, 9].sum()) * row_bytes,
    # V once, partial rows written
    "temporal_hals_partials": (n_block_cols.sum() + n_partial_rows) * row_bytes,
    # c_i, the neighbors' c_j, the ring term and the partial rows read, c_i written
    "temporal_hals": (3 * n_signals + n_edges + n_partial_rows) * row_bytes,
}
work_flops = {
    "gemm_nt ring_c": 2 * n_signals * rrp * compression.n_frames_padded,
    "gemm_nt ring_term": 2 * n_signals * rrp * compression.n_frames_padded,
    "gemm_nn ring_term": 2 * n_signals * rrp * compression.n_frames_padded,
}

bandwidth = measure_bandwidth()
fma_throughput = measure_fma_throughput()
print(f"measured bandwidth: {bandwidth:.1f} GB/s, FMA throughput: {fma_throughput:.2f} TFLOP/s")

rows = []
for kernel, t in sorted(timings.items(), key=lambda x: -x[1]):
    ms = t / n_iterations
    if kernel in work_bytes:
        floor = work_bytes[kernel] / (bandwidth * 1e9) * 1e3
    elif kernel in work_flops:
        floor = work_flops[kernel] / (fma_throughput * 1e12) * 1e3
    else:
        floor = np.nan
    rows.append(
        {
            "device": adapter.info.device,
            "kernel": kernel,
            "ms_per_iteration": ms,
            "floor_ms": floor,
            "efficiency": floor / ms,
            "n_signals": n_signals,
            "ring_rank": ring_rank,
        }
    )

df = pd.DataFrame(rows)
print(df.drop(columns="device").to_string(index=False, float_format="%.2f"))
print(
    f"per iteration: dispatches {df['ms_per_iteration'].sum():.2f} ms, first to last compute pass "
    f"{np.mean(spans):.2f} ms, wall time without timestamps {np.median(wall_times):.2f} ms"
)

if not Path(__file__).parent.joinpath("benchmark_hals.csv").is_file():
    df.to_csv("benchmark_hals.csv", index=False)
else:
    df.to_csv("benchmark_hals.csv", index=False, header=False, mode="a")
