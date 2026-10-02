// x = q0 q1[:, t], the ring term of the frame t on the columns of U: q0 [rank, ring_rank4] and q1^T
// [n_frames_padded, ring_rank4] vec4s. One invocation per row of q0, the row t of q1^T is the same for all.

@group(0) @binding(0)
var<storage, read> q0: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> q1_t: array<vec4<f32>>;
// uniforms, since t changes every frame and the ring rank with the signals
struct Frame {
    t: u32,
    // ring_rank_padded / 4
    ring_rank4: u32,
}
@group(0) @binding(2)
var<uniform> frame: Frame;
@group(0) @binding(3)
var<storage, read_write> x: array<f32>;

override rank: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn ring_frame(@builtin(global_invocation_id) gid: vec3u) {
    let k = gid.x;
    if (k >= rank) {
        return;
    }
    var sum = vec4<f32>(0.0);
    for (var j = 0u; j < frame.ring_rank4; j++) {
        sum = fma(q0[k * frame.ring_rank4 + j], q1_t[frame.t * frame.ring_rank4 + j], sum);
    }
    x[k] = (sum.x + sum.y) + (sum.z + sum.w);
}
