// masknmf's rank_1_NMF_fit for each component of its merge graph (merge_components): the rank-1 nonnegative fit s t^T
// of the members' a_j c_j^T, with s over the union of the members' pixels,
//
//     s = mean of the a_j, mask = s > 0, then 5 times
//     t = relu(sum_j (a_j . s) c_j / ||s||^2),    s = relu(sum_j a_j (c_j . t) / ||t||^2) mask
//
// norms of 0 as 1. Component g has the members members[member_ptr[g]:member_ptr[g + 1]] in masknmf's order and the
// union pixels union_ptr[g]:union_ptr[g + 1]; entries[entry_ptr[g] + u K + j] is the index in a_values of member j at
// union pixel u, none if the member has no entry there. s and the mask are over the union pixels, t is row g of t_out,
// 0 beyond the frames. One workgroup per component: strided sums of each invocation, then workgroup_sum. Each
// invocation reads back only the entries of s, mask and t that it wrote, the dot products of the members are shared
// through workgroup memory.

@group(0) @binding(0)
var<storage, read> member_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> members: array<u32>;
@group(0) @binding(2)
var<storage, read> union_ptr: array<u32>;
@group(0) @binding(3)
var<storage, read> entry_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> entries: array<u32>;
@group(0) @binding(5)
var<storage, read> a_values: array<f32>;
// [n_signals, n4]
@group(0) @binding(6)
var<storage, read> c: array<vec4<f32>>;
@group(0) @binding(7)
var<storage, read_write> s: array<f32>;
@group(0) @binding(8)
var<storage, read_write> mask: array<f32>;
// [n_components, n4]
@group(0) @binding(9)
var<storage, read_write> t_out: array<vec4<f32>>;

override n_frames: u32;
// vec4s per row of c and t_out
override n4: u32;
// at least the number of members of each component
override max_members: u32;

const wg_size: u32 = 256u;
const none: u32 = 0xffffffffu;

// a_j . s and c_j . t of the members
var<workgroup> spatial_dots: array<f32, max_members>;
var<workgroup> temporal_dots: array<f32, max_members>;

// a_j at union pixel u of the component
fn a_value(e0: u32, n_members: u32, u: u32, j: u32) -> f32 {
    let index = entries[e0 + u * n_members + j];
    return select(0.0, a_values[min(index, arrayLength(&a_values) - 1u)], index != none);
}

// the frames t4 * 4:t4 * 4 + 4 of the row that are frames of the movie
fn frame_mask(t4: u32) -> vec4<bool> {
    return vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n_frames);
}

@compute @workgroup_size(wg_size)
fn rank_1_nmf_fit(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let g = wid.x;
    let m0 = member_ptr[g];
    let n_members = member_ptr[g + 1u] - m0;
    let u0 = union_ptr[g];
    let n_union = union_ptr[g + 1u] - u0;
    let e0 = entry_ptr[g];
    let row_t = g * n4;

    // the mean of the members and the mask
    let inv_k = 1.0 / f32(n_members);
    for (var u = lid; u < n_union; u += wg_size) {
        var x = 0.0;
        for (var j = 0u; j < n_members; j++) {
            x = fma(a_value(e0, n_members, u, j), inv_k, x);
        }
        s[u0 + u] = x;
        mask[u0 + u] = select(0.0, 1.0, x > 0.0);
    }

    for (var iteration = 0u; iteration < 5u; iteration++) {
        // a_j . s and ||s||^2, workgroup_sum's barriers order the writes of spatial_dots before their reads
        for (var j = 0u; j < n_members; j++) {
            var x = 0.0;
            for (var u = lid; u < n_union; u += wg_size) {
                x = fma(a_value(e0, n_members, u, j), s[u0 + u], x);
            }
            let dot_j = workgroup_sum(x, lid);
            if (lid == 0u) {
                spatial_dots[j] = dot_j;
            }
        }
        var x = 0.0;
        for (var u = lid; u < n_union; u += wg_size) {
            x = fma(s[u0 + u], s[u0 + u], x);
        }
        var spatial_norm = workgroup_sum(x, lid);
        spatial_norm = select(spatial_norm, 1.0, spatial_norm == 0.0);

        // t
        for (var t4 = lid; t4 < n4; t4 += wg_size) {
            var y = vec4<f32>();
            for (var j = 0u; j < n_members; j++) {
                y = fma(vec4<f32>(spatial_dots[j]), c[members[m0 + j] * n4 + t4], y);
            }
            t_out[row_t + t4] = select(vec4<f32>(), max(y / spatial_norm, vec4<f32>()), frame_mask(t4));
        }

        // c_j . t and ||t||^2
        for (var j = 0u; j < n_members; j++) {
            var y = vec4<f32>();
            for (var t4 = lid; t4 < n4; t4 += wg_size) {
                y = fma(c[members[m0 + j] * n4 + t4], t_out[row_t + t4], y);
            }
            let dot_j = workgroup_sum((y.x + y.y) + (y.z + y.w), lid);
            if (lid == 0u) {
                temporal_dots[j] = dot_j;
            }
        }
        var y = vec4<f32>();
        for (var t4 = lid; t4 < n4; t4 += wg_size) {
            let t = t_out[row_t + t4];
            y = fma(t, t, y);
        }
        var temporal_norm = workgroup_sum((y.x + y.y) + (y.z + y.w), lid);
        temporal_norm = select(temporal_norm, 1.0, temporal_norm == 0.0);

        // s
        for (var u = lid; u < n_union; u += wg_size) {
            var z = 0.0;
            for (var j = 0u; j < n_members; j++) {
                z = fma(a_value(e0, n_members, u, j), temporal_dots[j], z);
            }
            s[u0 + u] = max(z / temporal_norm * mask[u0 + u], 0.0);
        }
    }
}
