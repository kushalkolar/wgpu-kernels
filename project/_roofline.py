import time
from collections import defaultdict
from typing import Callable

import numpy as np
import wgpu
from wgpu.backends.wgpu_native._ffi import lib
import pygfx

from ._hals import create_buffer


class _TimestampEncoder:
    """forwards to a command encoder, and writes timestamps at the beginning and end of each compute pass"""

    def __init__(self, encoder: wgpu.GPUCommandEncoder, query_set: wgpu.GPUQuerySet):
        self._encoder = encoder
        self._query_set = query_set
        # label of each compute pass
        self.labels = []

    def begin_compute_pass(self, *, label: str = "") -> wgpu.GPUComputePassEncoder:
        index = 2 * len(self.labels)
        self.labels.append(label)
        return self._encoder.begin_compute_pass(
            label=label,
            timestamp_writes={
                "query_set": self._query_set,
                "beginning_of_pass_write_index": index,
                "end_of_pass_write_index": index + 1,
            },
        )

    def __getattr__(self, name):
        return getattr(self._encoder, name)


def time_compute_passes(
    record: Callable[[wgpu.GPUCommandEncoder], None], max_passes: int = 2048
) -> tuple[dict[str, float], float]:
    """
    Submit the commands that ``record(encoder)`` records into a command encoder, with a GPU timestamp at the
    beginning and end of each compute pass. Needs the "timestamp-query" feature.

    Returns the time of the compute passes in ms summed by pass label, and the time from the beginning of the
    first pass to the end of the last one.
    """
    device = pygfx.renderers.wgpu.get_shared().device
    query_set = device.create_query_set(type="timestamp", count=2 * max_passes)
    resolved = device.create_buffer(
        size=16 * max_passes,
        usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
    )

    encoder = _TimestampEncoder(device.create_command_encoder(), query_set)
    record(encoder)
    n = 2 * len(encoder.labels)
    encoder.resolve_query_set(query_set, 0, n, resolved, 0)
    device.queue.submit([encoder.finish()])

    # the timestamps are in ticks of the GPU's timestamp counter: wgpu-native does not enable wgpu-core's
    # conversion to ns, and wgpu-py does not wrap the native function that returns the ns per tick
    ms_per_tick = lib.wgpuQueueGetTimestampPeriod(device.queue._internal) / 1e6
    ticks = np.frombuffer(device.queue.read_buffer(resolved, 0, 8 * n), dtype=np.uint64)
    ticks = ticks.astype(np.int64)
    ms_by_label = defaultdict(float)
    for label, pass_ticks in zip(encoder.labels, ticks[1::2] - ticks[::2]):
        ms_by_label[label] += pass_ticks * ms_per_tick

    return dict(ms_by_label), (ticks[-1] - ticks[0]) * ms_per_tick


def _time(device: wgpu.GPUDevice, pipeline, bind_group, n_workgroups: int, n: int = 10) -> float:
    """median time of a dispatch in seconds, waiting for the GPU after each one"""

    def run():
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(n_workgroups)
        compute_pass.end()
        device.queue.submit([encoder.finish()])
        device._poll_wait()

    run()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def measure_bandwidth(n_bytes: int = 512 * 2**20) -> float:
    """memory bandwidth of a vec4 copy kernel, bytes read + written, in GB/s"""
    device = pygfx.renderers.wgpu.get_shared().device
    src = create_buffer(device, np.zeros(n_bytes // 4, np.float32))
    dst = create_buffer(device, np.zeros(n_bytes // 4, np.float32))

    code = """
    @group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
    @group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
        for (var i = gid.x; i < arrayLength(&src); i += nwg.x * 256u) {
            dst[i] = src[i];
        }
    }
    """
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": device.create_shader_module(code=code), "entry_point": "main"}
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": src}},
            {"binding": 1, "resource": {"buffer": dst}},
        ],
    )
    t = _time(device, pipeline, bind_group, 4096)
    return 2 * n_bytes / t / 1e9


def measure_fma_throughput(n_iterations: int = 4096, n_workgroups: int = 4096) -> float:
    """float32 throughput of independent vec4 FMA chains, in TFLOP/s"""
    device = pygfx.renderers.wgpu.get_shared().device
    out = create_buffer(device, np.zeros(n_workgroups * 256 * 4, np.float32))

    code = f"""
    @group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3u) {{
        var x0 = vec4<f32>(f32(gid.x));
        var x1 = x0 + 1.0; var x2 = x0 + 2.0; var x3 = x0 + 3.0;
        var x4 = x0 + 4.0; var x5 = x0 + 5.0; var x6 = x0 + 6.0; var x7 = x0 + 7.0;
        let m = vec4<f32>(0.9999);
        let c = vec4<f32>(0.0001);
        for (var i = 0u; i < {n_iterations}u; i++) {{
            x0 = fma(x0, m, c); x1 = fma(x1, m, c); x2 = fma(x2, m, c); x3 = fma(x3, m, c);
            x4 = fma(x4, m, c); x5 = fma(x5, m, c); x6 = fma(x6, m, c); x7 = fma(x7, m, c);
        }}
        out[gid.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    }}
    """
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": device.create_shader_module(code=code), "entry_point": "main"}
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": {"buffer": out}}],
    )
    t = _time(device, pipeline, bind_group, n_workgroups)
    return n_workgroups * 256 * n_iterations * 8 * 4 * 2 / t / 1e12
