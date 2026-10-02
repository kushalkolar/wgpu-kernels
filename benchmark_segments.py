"""
Read bandwidth of the access pattern that spatial_hals_diff uses to stage the rows of V: each workgroup reads all
frames of a block of rows_per_workgroup consecutive rows of V, one segment of the frames of every row at a time,
with a barrier after each segment. The segment length is swept from 64 B to whole rows, with 16 B of workgroup
memory and with as much as spatial_hals_diff allocates, which limits the number of workgroups per CU. This
separates the cost of short segments from the cost of low occupancy.

Every dispatch reads all of V once and is timed on the GPU with timestamps.
"""

import numpy as np
import pandas as pd
import pygfx

import fastplotlib as fpl

from project import select_adapter, CompressionBuffers
from project._hals import create_empty_buffer
from project._spmv import ComputeShader
from project._roofline import measure_bandwidth, time_compute_passes

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
pygfx.renderers.wgpu.enable_wgpu_features("timestamp-query")

dmr_path = "./demix_new.hdf5"
n_iterations = 5
wg_size = 128
# the rows of V of a block of U
rows_per_workgroup = 19
# workgroup memory of spatial_hals_diff with diff_chunk4 = 16, from RADV_DEBUG=shaderstats on the Radeon 780M
diff_workgroup_bytes = 17920

compression = CompressionBuffers(dmr_path)
device = pygfx.renderers.wgpu.get_shared().device
n4 = compression.n_frames_padded // 4
n_workgroups = -(-compression.rank // rows_per_workgroup)
out = create_empty_buffer(device, n_workgroups * wg_size * 16)

wgsl = """
// V, [rank, n_frames_padded]
@group(0) @binding(0)
var<storage, read> temporal_compressed: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> out: array<vec4<f32>>;

override n4: u32;
override rank: u32;
override rows_per_workgroup: u32;
// segment length, in vec4s
override seg4: u32;
// workgroup memory in vec4s, only there to limit the number of workgroups per CU
override pad4: u32;

const wg_size: u32 = 128u;

var<workgroup> pad: array<vec4<f32>, pad4>;

@compute @workgroup_size(wg_size)
fn read_segments(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let row0 = wid.x * rows_per_workgroup;
    var acc = vec4<f32>(0.0);
    for (var t0 = 0u; t0 < n4; t0 += seg4) {
        for (var j = lid; j < rows_per_workgroup * seg4; j += wg_size) {
            let row = row0 + j / seg4;
            let t = t0 + j % seg4;
            if (row < rank && t < n4) {
                acc += temporal_compressed[row * n4 + t];
            }
        }
        workgroupBarrier();
    }
    pad[lid % pad4] = acc;
    workgroupBarrier();
    out[wid.x * wg_size + lid] = pad[(lid + 1u) % pad4];
}
"""

v_bytes = compression.rank * n4 * 16
bandwidth = measure_bandwidth()
print(f"measured copy bandwidth (read + write): {bandwidth:.1f} GB/s")

results = []
for pad_bytes in (16, diff_workgroup_bytes):
    for seg4 in (4, 8, 16, 32, 64, 128, 256, n4):
        shader = ComputeShader(wgsl, entry_point="read_segments")
        for name, value in {
            "n4": n4,
            "rank": compression.rank,
            "rows_per_workgroup": rows_per_workgroup,
            "seg4": seg4,
            "pad4": pad_bytes // 16,
        }.items():
            shader.set_constant(name, value)
        shader.set_resource(0, compression.temporal_compressed)
        shader.set_resource(1, out)

        # warm up, compiles the pipeline
        time_compute_passes(lambda encoder: shader.encode(encoder, n_workgroups))
        ms = np.median(
            [
                time_compute_passes(lambda encoder: shader.encode(encoder, n_workgroups))[0]["read_segments"]
                for _ in range(n_iterations)
            ]
        )
        results.append(
            {
                "workgroup_memory_bytes": pad_bytes,
                "segment_bytes": "whole row" if seg4 == n4 else 16 * seg4,
                "ms": ms,
                "read_GB_s": v_bytes / (ms * 1e6),
            }
        )
        print(results[-1])

df = pd.DataFrame(results)
print()
print(df.to_string(index=False, float_format="%.1f"))
