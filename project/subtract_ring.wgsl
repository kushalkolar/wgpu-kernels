// out = V - q0 q1, V [rank, n4] and q1 [ring_rank_padded, n4] vec4s, q0 [rank, ring_rank_padded]
//
// One invocation per vec4 of frames and block of rows_per_invocation rows, so that each vec4 of q1 is read once for
// the block. The values of q0 are the same for the whole workgroup.

@group(0) @binding(0)
var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> q0: array<f32>;
@group(0) @binding(2)
var<storage, read> q1: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read_write> out: array<vec4<f32>>;

override rank: u32;
override n4: u32;
override ring_rank_padded: u32;

const wg_size: u32 = 256u;
const rows_per_invocation: u32 = 8u;

// workgroup x indexes the chunks of wg_size vec4s of frames, workgroup y the blocks of rows
@compute @workgroup_size(wg_size)
fn subtract_ring(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let t4 = wid.x * wg_size + lid;
    let first_row = wid.y * rows_per_invocation;
    if (t4 >= n4) {
        return;
    }
    var acc: array<vec4<f32>, rows_per_invocation>;
    for (var i = 0u; i < rows_per_invocation; i++) {
        acc[i] = v[min(first_row + i, rank - 1u) * n4 + t4];
    }
    for (var r = 0u; r < ring_rank_padded; r++) {
        let x = q1[r * n4 + t4];
        for (var i = 0u; i < rows_per_invocation; i++) {
            acc[i] = fma(vec4<f32>(-q0[min(first_row + i, rank - 1u) * ring_rank_padded + r]), x, acc[i]);
        }
    }
    for (var i = 0u; i < rows_per_invocation; i++) {
        if (first_row + i < rank) {
            out[(first_row + i) * n4 + t4] = acc[i];
        }
    }
}
