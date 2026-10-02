// For each signal i: sum over time of c_i, c_i . c_i, and c_i . c_j for each signal j > i whose support
// overlaps the support of i. Written to both edges (i, j) and (j, i) of the overlap graph.
//
// One workgroup per signal. Each invocation accumulates a strided set of vec4s over time (Harris
// "Optimizing Parallel Reduction in CUDA", algorithm cascading), then workgroup_sum. Up to 4 neighbors
// are accumulated per pass over c_i so that c_i is read once for typical neighbor counts.

// c, [n_signals, n_frames_padded]
@group(0) @binding(0)
var<storage, read> temporal_demixed: array<vec4<f32>>;
// edges (i, j) with j > i, as indices into neighbors, grouped by i
@group(0) @binding(1)
var<storage, read> upper_ptr: array<u32>;
@group(0) @binding(2)
var<storage, read> upper_edges: array<u32>;
// overlap graph, neighbors of signal i are neighbors[neighbor_ptr[i]:neighbor_ptr[i + 1]]
@group(0) @binding(3)
var<storage, read> neighbors: array<u32>;
// index of edge (j, i) for each edge (i, j)
@group(0) @binding(4)
var<storage, read> neighbor_reverse: array<u32>;

@group(0) @binding(5)
var<storage, read_write> c_sum: array<f32>;
@group(0) @binding(6)
var<storage, read_write> c_sq: array<f32>;
// c_i . c_j for each edge (i, j)
@group(0) @binding(7)
var<storage, read_write> gram: array<f32>;

override n_frames_padded: u32;
const wg_size: u32 = 256u;

fn hsum(v: vec4<f32>) -> f32 {
    return v.x + v.y + v.z + v.w;
}

@compute @workgroup_size(wg_size)
fn temporal_gram(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.y * nwg.x + wid.x;
    if (i >= arrayLength(&c_sum)) {
        return;
    }

    let n4 = n_frames_padded / 4u;
    let row_i = i * n4;

    var s = vec4<f32>(0.0);
    var sq = vec4<f32>(0.0);
    for (var t = lid; t < n4; t += wg_size) {
        let x = temporal_demixed[row_i + t];
        s += x;
        sq = fma(x, x, sq);
    }

    let total_s = workgroup_sum(hsum(s), lid);
    let total_sq = workgroup_sum(hsum(sq), lid);
    if (lid == 0u) {
        c_sum[i] = total_s;
        c_sq[i] = total_sq;
    }

    let start = upper_ptr[i];
    let end = upper_ptr[i + 1u];

    for (var e0 = start; e0 < end; e0 += 4u) {
        // clamp to the last edge, the extra results are not written
        let e1 = min(e0 + 1u, end - 1u);
        let e2 = min(e0 + 2u, end - 1u);
        let e3 = min(e0 + 3u, end - 1u);
        let row0 = neighbors[upper_edges[e0]] * n4;
        let row1 = neighbors[upper_edges[e1]] * n4;
        let row2 = neighbors[upper_edges[e2]] * n4;
        let row3 = neighbors[upper_edges[e3]] * n4;

        var acc0 = vec4<f32>(0.0);
        var acc1 = vec4<f32>(0.0);
        var acc2 = vec4<f32>(0.0);
        var acc3 = vec4<f32>(0.0);
        for (var t = lid; t < n4; t += wg_size) {
            let x = temporal_demixed[row_i + t];
            acc0 = fma(x, temporal_demixed[row0 + t], acc0);
            acc1 = fma(x, temporal_demixed[row1 + t], acc1);
            acc2 = fma(x, temporal_demixed[row2 + t], acc2);
            acc3 = fma(x, temporal_demixed[row3 + t], acc3);
        }

        let dots = array<f32, 4>(
            workgroup_sum(hsum(acc0), lid),
            workgroup_sum(hsum(acc1), lid),
            workgroup_sum(hsum(acc2), lid),
            workgroup_sum(hsum(acc3), lid),
        );

        if (lid < 4u && e0 + lid < end) {
            let e = upper_edges[e0 + lid];
            gram[e] = dots[lid];
            gram[neighbor_reverse[e]] = dots[lid];
        }
    }
}
