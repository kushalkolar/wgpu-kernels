"""
Time the fluctuating background update (FluctuatingBaseline.update, masknmf's ring model) on the synthetic spatial
footprints and temporal traces of benchmark_hals.py with the real compression results U, V: for a given background
rank, and with the rank estimated first from the sketch of 305 columns, as after each support update. Each compute
pass is timed on the GPU with timestamps, the wall time of an update includes the QR and eigendecompositions on the
CPU and the transfers around them.
"""

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pygfx
import torch
import wgpu
from wgpu.backends.wgpu_native._ffi import lib

import fastplotlib as fpl

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers
from project._background import FluctuatingBaseline
from project._roofline import _TimestampEncoder

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
pygfx.renderers.wgpu.enable_wgpu_features("timestamp-query")
device = pygfx.renderers.wgpu.get_shared().device

dmr_path = "./demix_new.hdf5"
n_signals = 1500
background_rank = 168
n_updates = 3

compression = CompressionBuffers(dmr_path)
height, width = compression.fov_shape

# synthetic spatial footprints as in benchmark_hals.py: gaussian discs, overlapping
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
a = torch.sparse_coo_tensor(np.stack([pixels, signals]), values, (height * width, n_signals)).coalesce()
c = np.abs(rng.standard_normal((compression.n_frames, n_signals))).astype(np.float32)
gpu_signals = SignalBuffers(compression, a, c, np.zeros(height * width, dtype=np.float32))

# GPU time of the compute passes, summed by label
pass_ms = defaultdict(float)


class TimedFluctuatingBaseline(FluctuatingBaseline):
    """writes timestamps at the beginning and end of each compute pass"""

    def _new_encoder(self):
        self._query_set = device.create_query_set(type="timestamp", count=1024)
        return _TimestampEncoder(device.create_command_encoder(), self._query_set)

    def _submit(self, encoder):
        n = 2 * len(encoder.labels)
        resolved = device.create_buffer(
            size=8 * max(n, 2), usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC
        )
        if n > 0:
            encoder.resolve_query_set(self._query_set, 0, n, resolved, 0)
        device.queue.submit([encoder.finish()])
        if n > 0:
            ms_per_tick = lib.wgpuQueueGetTimestampPeriod(device.queue._internal) / 1e6
            ticks = np.frombuffer(device.queue.read_buffer(resolved, 0, 8 * n), dtype=np.uint64).astype(np.int64)
            for label, pass_ticks in zip(encoder.labels, ticks[1::2] - ticks[::2]):
                pass_ms[label] += pass_ticks * ms_per_tick


# CPU time of the factorizations
numpy_ms = {"ms": 0.0}
for name in ("qr", "eigh"):

    def timed(*args, _f=getattr(np.linalg, name), **kwargs):
        t0 = time.perf_counter()
        out = _f(*args, **kwargs)
        numpy_ms["ms"] += 1e3 * (time.perf_counter() - t0)
        return out

    setattr(np.linalg, name, timed)

u = load_compression(dmr_path)["u"]
timed_update = TimedFluctuatingBaseline(compression, u)
timed_update.set_signals(gpu_signals)
update = FluctuatingBaseline(compression, u)
update.set_signals(gpu_signals)

for rank in (background_rank, None):
    # warm up, compiles the pipelines
    timed_update.update(rank)
    update.update(rank)
    pass_ms.clear()
    for i in range(n_updates):
        timed_update.update(rank, seed=i)
    wall, numpy_times = [], []
    for i in range(n_updates):
        numpy_ms["ms"] = 0.0
        t0 = time.perf_counter()
        estimated = update.update(rank, seed=i)
        device._poll_wait()
        wall.append(1e3 * (time.perf_counter() - t0))
        numpy_times.append(numpy_ms["ms"])
    description = f"given rank {rank}" if rank is not None else f"rank estimated, {estimated}"
    print(f"=== {description}: wall time {np.median(wall):.1f} ms, numpy qr + eigh {np.median(numpy_times):.1f} ms")
    df = pd.DataFrame(
        [{"pass": label, "ms_per_update": ms / n_updates} for label, ms in pass_ms.items()]
    ).sort_values("ms_per_update", ascending=False)
    print(df.to_string(index=False, float_format="%.2f"))
    print(f"compute passes {df['ms_per_update'].sum():.1f} ms")
