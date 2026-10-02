// For each signal i, the products of the raw traces with its standardized trace: products[e] = c_s . c~_i for each
// edge e = (i, s) of a graph of the signals (neighbors[neighbor_ptr[i]:neighbor_ptr[i + 1]]), the overlap graph of a or
// its dilated version (compute_dilated_overlap_graph), and self_products[i] = c_i . c~_i if compute_self_products. c
// and c~ are [n_signals, n4] vec4s, 0 beyond the frames.
//
// One workgroup per signal, as temporal_gram.wgsl: each invocation accumulates a strided set of vec4s over time, then
// workgroup_sum, for up to 4 neighbors per pass over c~_i.

@group(0) @binding(0)
var<storage, read> c: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> c_tilde: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(3)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(4)
var<storage, read_write> products: array<f32>;
@group(0) @binding(5)
var<storage, read_write> self_products: array<f32>;
// x: n_signals, a uniform since it changes with the signals
@group(0) @binding(6)
var<uniform> n_signals: vec4<u32>;

override n4: u32;
override compute_self_products: bool = true;

const wg_size: u32 = 256u;

fn hsum(v: vec4<f32>) -> f32 {
    return (v.x + v.y) + (v.z + v.w);
}

@compute @workgroup_size(wg_size)
fn trace_products(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.y * nwg.x + wid.x;
    if (i >= n_signals.x) {
        return;
    }
    let row_i = i * n4;

    if (compute_self_products) {
        var own = vec4<f32>();
        for (var t = lid; t < n4; t += wg_size) {
            own = fma(c[row_i + t], c_tilde[row_i + t], own);
        }
        let total = workgroup_sum(hsum(own), lid);
        if (lid == 0u) {
            self_products[i] = total;
        }
    }

    let start = neighbor_ptr[i];
    let end = neighbor_ptr[i + 1u];
    for (var e0 = start; e0 < end; e0 += 4u) {
        // clamp to the last edge, the extra results are not written
        let row0 = neighbors[e0] * n4;
        let row1 = neighbors[min(e0 + 1u, end - 1u)] * n4;
        let row2 = neighbors[min(e0 + 2u, end - 1u)] * n4;
        let row3 = neighbors[min(e0 + 3u, end - 1u)] * n4;
        var acc0 = vec4<f32>();
        var acc1 = vec4<f32>();
        var acc2 = vec4<f32>();
        var acc3 = vec4<f32>();
        for (var t = lid; t < n4; t += wg_size) {
            let x = c_tilde[row_i + t];
            acc0 = fma(c[row0 + t], x, acc0);
            acc1 = fma(c[row1 + t], x, acc1);
            acc2 = fma(c[row2 + t], x, acc2);
            acc3 = fma(c[row3 + t], x, acc3);
        }
        let dots = vec4<f32>(
            workgroup_sum(hsum(acc0), lid),
            workgroup_sum(hsum(acc1), lid),
            workgroup_sum(hsum(acc2), lid),
            workgroup_sum(hsum(acc3), lid),
        );
        if (lid < 4u && e0 + lid < end) {
            products[e0 + lid] = dots[lid];
        }
    }
}
