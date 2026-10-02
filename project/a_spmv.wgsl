// out[p] = x[p] - sum over the entries e of a at the pixel p of a_values[e] (sums[s] / n_frames), s the signal of e: x
// minus a times the means of the traces, from their sums over the frames. One invocation per pixel, over its entries
// in the pixel-major view of a (see compute_signal_structures).

@group(0) @binding(0)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(2)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(3)
var<storage, read> a_values: array<f32>;
@group(0) @binding(4)
var<storage, read> sums: array<f32>;
@group(0) @binding(5)
var<storage, read> x: array<f32>;
@group(0) @binding(6)
var<storage, read_write> out: array<f32>;

override n_pixels: u32;
override n_frames: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn a_spmv(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let p = gid.y * nwg.x * wg_size + gid.x;
    if (p >= n_pixels) {
        return;
    }
    var sum = 0.0;
    for (var e = pixel_ptr[p]; e < pixel_ptr[p + 1u]; e++) {
        sum = fma(a_values[pixel_entries[e]], sums[pixel_signals[e]] / f32(n_frames), sum);
    }
    out[p] = x[p] - sum;
}
