// cumulator[i, t] = sum over the (signal, block) pairs of i of partial[pair, t] - ring_term[i, t] - atb_i
// over one chunk of frames, the "cumulator" of masknmf's temporal_update_hals:
//
//   a_i^T U V - (a_i^T U q0) q1 - a_i^T b
//
// ring_term = ring_w q1 is computed by gemm_nt into the cumulator buffer beforehand, and is replaced in
// place by the cumulator. Frames beyond n_frames are set to zero.

struct Chunk {
    start4: u32,
    size4: u32,
}

@group(0) @binding(0)
var<uniform> chunk: Chunk;
// (signal, block) pairs of signal i are pair_ptr[i]:pair_ptr[i + 1]
@group(0) @binding(1)
var<storage, read> pair_ptr: array<u32>;
// [n_pairs, chunk.size4] vec4s
@group(0) @binding(2)
var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read> atb: array<f32>;
// [n_signals, n_frames_padded], holds ring_w q1 before this kernel
@group(0) @binding(4)
var<storage, read_write> cumulator: array<vec4<f32>>;

override n_frames: u32;
override n_frames_padded: u32;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size)
fn temporal_hals_cumulator(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = wid.y;
    let local4 = wid.x * wg_size + lid;
    if (local4 >= chunk.size4) {
        return;
    }
    let t4 = chunk.start4 + local4;
    let index = i * (n_frames_padded / 4u) + t4;

    var sum = vec4<f32>(0.0);
    for (var pair = pair_ptr[i]; pair < pair_ptr[i + 1u]; pair++) {
        sum += partial[pair * chunk.size4 + local4];
    }

    let frames = vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u);
    let value = sum - cumulator[index] - vec4<f32>(atb[i]);
    cumulator[index] = select(vec4<f32>(0.0), value, frames < vec4<u32>(n_frames));
}
