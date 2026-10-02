// out[e] = number of bits set in both bits[i] and bits[j] for each pair e = (i, j), the pixels where both thresholded
// images are above the threshold (a pair (i, i) gives the area of image i). bits is [n_signals, n_row_words] from
// standard_image_bitmasks.wgsl. One workgroup per pair: each invocation counts a strided set of vec4s of words, then
// workgroup_sum, exact since the counts are integers below 2^24.

@group(0) @binding(0)
var<storage, read> bits: array<vec4<u32>>;
@group(0) @binding(1)
var<storage, read> pairs: array<vec2<u32>>;
@group(0) @binding(2)
var<storage, read_write> out: array<u32>;
// x: the number of pairs
@group(0) @binding(3)
var<uniform> n_pairs: vec4<u32>;

// words per signal, a multiple of 4
override n_row_words: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn bitmask_overlaps(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let e = wid.y * nwg.x + wid.x;
    if (e >= n_pairs.x) {
        return;
    }
    let pair = pairs[e];
    let n4 = n_row_words / 4u;
    var count = 0u;
    for (var t = lid; t < n4; t += wg_size) {
        let x = bits[pair.x * n4 + t] & bits[pair.y * n4 + t];
        count += (countOneBits(x.x) + countOneBits(x.y)) + (countOneBits(x.z) + countOneBits(x.w));
    }
    let total = workgroup_sum(f32(count), lid);
    if (lid == 0u) {
        out[e] = u32(total);
    }
}
