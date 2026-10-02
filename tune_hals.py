"""
Sweep the pipeline constants of the memory-bound HALS kernels, one kernel at a time with the other
kernels at their defaults, and time the kernel for each setting on the GPU with timestamps: the median of
n_iterations runs, after running the update for warm_up_seconds so that every setting is timed at a sustained
GPU clock. Settings whose workgroup memory does not fit the device limit, or for which spatial_hals_diff would
stage more than 8 vec4s per invocation, are skipped.
"""

import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pygfx
import torch

import fastplotlib as fpl

from project import select_adapter, CompressionBuffers
from project._hals import SignalBuffers, HALS
from project._roofline import time_compute_passes

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
pygfx.renderers.wgpu.enable_wgpu_features("timestamp-query")

dmr_path = "./demix_new.hdf5"
n_signals = 1500
ring_rank = 168
n_iterations = 10
warm_up_seconds = 0.5

compression = CompressionBuffers(dmr_path)
height, width = compression.fov_shape
device = pygfx.renderers.wgpu.get_shared().device
max_workgroup_bytes = device.limits["max-compute-workgroup-storage-size"]

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
max_rows = 4 * compression.max_block_cols


def diff_staged_per_invocation(chunk4):
    return -(-max_rows * chunk4 // 256) + -(-32 * chunk4 // 256)


def diff_workgroup_bytes(chunk4):
    v_rows = -(-max_rows * chunk4 // 256) * 256 // chunk4
    c_rows = -(-32 * chunk4 // 256) * 256 // chunk4
    return (max(v_rows * (chunk4 + 1), 512) + c_rows * (chunk4 + 1)) * 16


# (kernels whose times are summed, update function name, list of tuning dicts). The superblock size also
# sets the number of partial rows that temporal_hals reads
sweeps = [
    (
        ("temporal_hals_partials", "temporal_hals"),
        "temporal_update",
        [
            {"partials_wg_size": wg, "partials_superblock_size": size}
            for wg, size in product((64, 128, 256), (2, 3, 4))
        ],
    ),
    (
        ("spatial_hals_diff",),
        "spatial_update",
        [
            {"diff_chunk4": chunk4}
            for chunk4 in (4, 8, 16, 32, 64)
            if diff_workgroup_bytes(chunk4) <= max_workgroup_bytes
            and diff_staged_per_invocation(chunk4) <= 8
        ],
    ),
    (
        ("temporal_hals",),
        "temporal_update",
        [{"temporal_hals_wg_size": wg} for wg in (64, 128, 256)],
    ),
]

results = []
for kernels, update, configs in sweeps:
    kernel = " + ".join(kernels)
    for tuning in configs:
        hals = HALS(compression, gpu_signals, tuning=tuning)
        run = getattr(hals, update)
        # the first run compiles the pipelines
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < warm_up_seconds:
            run()
            device._poll_wait()
        timings = [
            time_compute_passes(lambda encoder: run(command_encoder=encoder))[0]
            for _ in range(n_iterations)
        ]
        ms = np.median([sum(t[k] for k in kernels) for t in timings])
        print(f"{kernel:26s} {tuning}: {ms:.2f} ms")
        results.append(
            {"device": adapter.info.device, "kernel": kernel, "tuning": str(tuning), "ms_per_iteration": ms}
        )
        del hals

df = pd.DataFrame(results)
print()
for kernel, group in df.groupby("kernel", sort=False):
    best = group.loc[group["ms_per_iteration"].idxmin()]
    print(f"best {kernel}: {best['tuning']} {best['ms_per_iteration']:.2f} ms")

if not Path(__file__).parent.joinpath("tune_hals.csv").is_file():
    df.to_csv("tune_hals.csv", index=False)
else:
    df.to_csv("tune_hals.csv", index=False, header=False, mode="a")
