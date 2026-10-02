// masknmf's MAD threshold and normalization of each pixel's trace r in local correlation images
// (get_local_correlation_structure, threshold_data_inplace): the lower median m of r over the frames (torch.median),
// the lower median MAD of |r - m|, the frames kept, |r - m| >= threshold = mad_threshold MAD (and r > m or r < m for
// the signs positive and negative, all frames for a mad_threshold of 0), the mean of r over them and
// inv = 1 / sqrt(sum over them of (r - mean)^2 + noise_term), 0 where the square root is below 1e-6.
// stats[q] = (m, threshold, mean, inv) for the rows q of traces.
//
// One workgroup per trace, held in registers as n_slots vec4s of u32 keys per invocation: an order-preserving map of
// the floats, the frames beyond the movie mapped to the largest key. The k-th smallest key is found from the highest
// bit down, a bit set if at most k keys are below the prefix with it set, one workgroup count per bit. After
// compaction_bits bits the keys left in the range are moved to workgroup memory, one per invocation, if there are at
// most wg_size of them, so that each later count compares one key. For the MAD the keys are replaced by those of
// |r - m| and a mask of the frames with r < m, and the mean and normalizer take r as m +- |r - m|, which is r up to
// the rounding of r - m. The workgroup counts go through subgroups, of at least 32 invocations.

@group(0) @binding(0)
var<storage, read> traces: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> stats: array<vec4<f32>>;
// x: the number of traces, y: the bits of the noise term T s^2; uniforms, since they change with the band and the pass
@group(0) @binding(2)
var<uniform> sizes: vec4<u32>;

override n_frames: u32;
// vec4s per row of traces
override n4: u32;
// vec4s of frames per invocation, at least n4 / wg_size and at most max_slots
override n_slots: u32;
override mad_threshold: f32;
// 0: unconstrained, 1: positive, 2: negative
override sign: u32;

const wg_size: u32 = 1024u;
const max_slots: u32 = 8u;
const tol: f32 = 1e-6;
const no_key: u32 = 0xffffffffu;
// the bits of the k-th smallest key determined before the compaction
const compaction_bits: u32 = 12u;

var<private> keys: array<vec4<u32>, max_slots>;
// the frames with r < m, bit 4 j + i for component i of slot j
var<private> below_median: u32;
// (subgroup_invocation_id, subgroup_size, subgroup_id)
var<private> subgroup: vec3<u32>;

var<workgroup> subgroup_sums: array<f32, wg_size / 32u>;
var<workgroup> active_keys: array<u32, wg_size>;
var<workgroup> n_active: atomic<u32>;

// the sum over the workgroup: subgroupAdd, then every subgroup adds the subgroups' sums, at most one per invocation
fn workgroup_sum(x: f32) -> f32 {
    // subgroup_sums may still be read from the previous call
    workgroupBarrier();
    let s = subgroupAdd(x);
    if (subgroup.x == 0u) {
        subgroup_sums[subgroup.z] = s;
    }
    workgroupBarrier();
    let n_subgroups = wg_size / subgroup.y;
    return subgroupAdd(select(0.0, subgroup_sums[min(subgroup.x, wg_size / 32u - 1u)], subgroup.x < n_subgroups));
}

// the frames of the movie among the vec4 j of this invocation
fn frame_mask(j: u32, lid: u32) -> vec4<bool> {
    return vec4<u32>(4u * (lid + wg_size * j)) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n_frames);
}

// increasing with the float
fn to_key(x: vec4<f32>) -> vec4<u32> {
    let b = bitcast<vec4<u32>>(x);
    return select(b | vec4<u32>(0x80000000u), ~b, (b & vec4<u32>(0x80000000u)) != vec4<u32>());
}

fn from_key(k: vec4<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(select(~k, k & vec4<u32>(0x7fffffffu), (k & vec4<u32>(0x80000000u)) != vec4<u32>()));
}

// the k-th smallest of the keys
fn kth_smallest(k: u32, lid: u32) -> f32 {
    var prefix = 0u;
    // the keys below prefix and in [prefix, prefix + 2^bit)
    var below = 0.0;
    var in_range = f32(n_frames);
    var compacted = false;
    var own = no_key;
    // the keys below prefix at the compaction
    var base = 0.0;
    for (var bit = 32u; bit > 0u; bit--) {
        if (bit == 32u - compaction_bits) {
            if (lid == 0u) {
                atomicStore(&n_active, 0u);
            }
            workgroupBarrier();
            compacted = in_range <= f32(wg_size);
            base = below;
            if (compacted) {
                for (var j = 0u; j < n_slots; j++) {
                    let x = keys[j];
                    for (var i = 0u; i < 4u; i++) {
                        if ((x[i] >> bit) == (prefix >> bit) && x[i] != no_key) {
                            active_keys[atomicAdd(&n_active, 1u)] = x[i];
                        }
                    }
                }
            }
            workgroupBarrier();
            if (compacted && lid < atomicLoad(&n_active)) {
                own = active_keys[lid];
            }
        }
        let candidate = prefix | (1u << (bit - 1u));
        var count = 0.0;
        if (compacted) {
            count = select(0.0, 1.0, own < candidate);
        } else {
            var counts = vec4<u32>();
            for (var j = 0u; j < n_slots; j++) {
                counts += select(vec4<u32>(), vec4<u32>(1u), keys[j] < vec4<u32>(candidate));
            }
            count = f32((counts.x + counts.y) + (counts.z + counts.w));
        }
        let total = workgroup_sum(count) + select(0.0, base, compacted);
        let lower = total - below;
        if (total <= f32(k)) {
            prefix = candidate;
            below = total;
            in_range -= lower;
        } else {
            in_range = lower;
        }
    }
    return from_key(vec4<u32>(prefix)).x;
}

// the frames of slot j with r < m
fn below_mask(j: u32) -> vec4<bool> {
    return ((vec4<u32>(below_median >> (4u * j)) >> vec4<u32>(0u, 1u, 2u, 3u)) & vec4<u32>(1u)) != vec4<u32>();
}

// r of slot j: from the keys, or m +- |r - m| when they hold |r - m|
fn trace(j: u32, median: f32) -> vec4<f32> {
    let x = from_key(keys[j]);
    if (mad_threshold == 0.0) {
        return x;
    }
    return vec4<f32>(median) + select(x, -x, below_mask(j));
}

fn kept(j: u32, lid: u32, threshold: f32) -> vec4<bool> {
    var keep = frame_mask(j, lid);
    if (mad_threshold != 0.0) {
        let d = from_key(keys[j]);
        keep &= d >= vec4<f32>(threshold);
        if (sign == 1u) {
            keep &= !below_mask(j) & (d > vec4<f32>());
        } else if (sign == 2u) {
            keep &= below_mask(j);
        }
    }
    return keep;
}

@compute @workgroup_size(wg_size)
fn trace_statistics(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_id) subgroup_index: u32,
) {
    subgroup = vec3<u32>(subgroup_invocation, subgroup_size, subgroup_index);
    let q = wid.y * nwg.x + wid.x;
    if (q >= sizes.x) {
        return;
    }
    for (var j = 0u; j < n_slots; j++) {
        let t4 = lid + wg_size * j;
        let x = traces[q * n4 + min(t4, n4 - 1u)];
        keys[j] = select(vec4<u32>(no_key), to_key(x), frame_mask(j, lid));
    }

    var median = 0.0;
    var threshold = 0.0;
    if (mad_threshold != 0.0) {
        let k = (n_frames - 1u) / 2u;
        median = kth_smallest(k, lid);
        for (var j = 0u; j < n_slots; j++) {
            let x = from_key(keys[j]);
            let below = x < vec4<f32>(median);
            below_median |= dot(select(vec4<u32>(), vec4<u32>(1u, 2u, 4u, 8u), below), vec4<u32>(1u)) << (4u * j);
            keys[j] = select(vec4<u32>(no_key), to_key(abs(x - median)), frame_mask(j, lid));
        }
        threshold = kth_smallest(k, lid) * mad_threshold;
    }

    // the mean over the kept frames, then the sum of squares around it
    var count = vec4<f32>();
    var sum = vec4<f32>();
    for (var j = 0u; j < n_slots; j++) {
        let keep = kept(j, lid, threshold);
        count += select(vec4<f32>(), vec4<f32>(1.0), keep);
        sum += select(vec4<f32>(), trace(j, median), keep);
    }
    let n_kept = workgroup_sum((count.x + count.y) + (count.z + count.w));
    let total = workgroup_sum((sum.x + sum.y) + (sum.z + sum.w));
    let mean = select(0.0, total / n_kept, n_kept > 0.0);
    var squares = vec4<f32>();
    for (var j = 0u; j < n_slots; j++) {
        let y = trace(j, median) - mean;
        squares += select(vec4<f32>(), y * y, kept(j, lid, threshold));
    }
    let divisor = sqrt(workgroup_sum((squares.x + squares.y) + (squares.z + squares.w)) + bitcast<f32>(sizes.y));
    if (lid == 0u) {
        stats[q] = vec4<f32>(median, threshold, mean, select(1.0 / divisor, 0.0, divisor < tol));
    }
}
