// masknmf's standardized temporal traces, one workgroup per signal i of c [n_signals, n4] vec4s:
//
//     c~_i = (c_i - m_i) / ||c_i - m_i||, 0 where the norm is 0 (masknmf's nan_to_num)
//
// over the frames t < n_frames, 0 in the padding, with m_i the mean of c_i. stats[i] = (m_i, ||c_i - m_i||, sum of
// c~_i, sum of c_i) and max_abs[i] = max of |c_i|. The sums are pairwise as in row_sums.wgsl: one vec4 per invocation
// and a tree through workgroup memory for each chunk of 4 wg_size frames, then a tree over the chunks.

@group(0) @binding(0)
var<storage, read> c: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> c_tilde: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read_write> stats: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read_write> max_abs: array<f32>;

override n_frames: u32;
// vec4s per row of c and c_tilde, at most wg_size^2
override n4: u32;

const wg_size: u32 = 256u;

var<workgroup> chunk_sums: array<f32, wg_size>;

// the frames t4 * 4:t4 * 4 + 4 of the row that are frames of the movie
fn frame_mask(t4: u32) -> vec4<bool> {
    return vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n_frames);
}

// tree over the sums of the chunks
fn sum_chunks(n_chunks: u32, lid: u32) -> f32 {
    workgroupBarrier();
    return workgroup_sum(select(0.0, chunk_sums[lid], lid < n_chunks), lid);
}

@compute @workgroup_size(wg_size)
fn standardize(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let signal = wid.x;
    let row = signal * n4;
    let n_chunks = (n4 + wg_size - 1u) / wg_size;

    // the sum of c_i and the max of |c_i|
    var lane_max = 0.0;
    for (var chunk = 0u; chunk < n_chunks; chunk++) {
        let t4 = chunk * wg_size + lid;
        let x = select(vec4<f32>(), c[row + min(t4, n4 - 1u)], frame_mask(t4));
        let s = workgroup_sum((x.x + x.y) + (x.z + x.w), lid);
        if (lid == 0u) {
            chunk_sums[chunk] = s;
        }
        let a = abs(x);
        lane_max = max(lane_max, max(max(a.x, a.y), max(a.z, a.w)));
    }
    let sum = sum_chunks(n_chunks, lid);
    let mean = sum / f32(n_frames);
    let c_max = workgroup_max(lane_max, lid);

    // norm of c_i - m_i
    for (var chunk = 0u; chunk < n_chunks; chunk++) {
        let t4 = chunk * wg_size + lid;
        let x = select(vec4<f32>(), c[row + min(t4, n4 - 1u)] - mean, frame_mask(t4));
        let s = workgroup_sum((x.x * x.x + x.y * x.y) + (x.z * x.z + x.w * x.w), lid);
        if (lid == 0u) {
            chunk_sums[chunk] = s;
        }
    }
    let norm = sqrt(sum_chunks(n_chunks, lid));

    // c~_i and its sum
    for (var chunk = 0u; chunk < n_chunks; chunk++) {
        let t4 = chunk * wg_size + lid;
        let x = select(vec4<f32>(), (c[row + min(t4, n4 - 1u)] - mean) / norm, frame_mask(t4) & vec4<bool>(norm > 0.0));
        if (t4 < n4) {
            c_tilde[row + t4] = x;
        }
        let s = workgroup_sum((x.x + x.y) + (x.z + x.w), lid);
        if (lid == 0u) {
            chunk_sums[chunk] = s;
        }
    }
    let tilde_sum = sum_chunks(n_chunks, lid);

    if (lid == 0u) {
        stats[signal] = vec4<f32>(mean, norm, tilde_sum, sum);
        max_abs[signal] = c_max;
    }
}
