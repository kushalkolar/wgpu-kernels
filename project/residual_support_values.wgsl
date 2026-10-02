// masknmf's residual correlation image of each signal on its support (_compute_residual_correlation_image): for each
// entry (p, i) of a, with the sums over the signals s with an entry at p and the signals j of i's group among them,
//
//     r = x_i - sum_s a_ps c_s . c~_i - mean_p s_i,        masknmf's cumulator / n_i
//     value = (r + sum_j a_pj (c_j - m_j) . c~_i) / sqrt(norm2_p + a_pi^2 n_i^2 + 2 n_i r a_pi + noise_term)
//
// with (c_j - m_j) . c~_i = c_j . c~_i - m_j s_i, x_i and mean_p, norm2_p from residual_pixels.wgsl, the products of
// trace_products.wgsl and (m, n, s) of standardize.wgsl. The numerator adds back the part of i's group, the denominator
// only i's, as masknmf does. Where masknmf's value is nan (norm2_p < 0, a negative radicand, or 0 / 0) it is 0, and
// where it is +-inf (a nonzero numerator over 0) the largest finite value of that sign, as masknmf's nan_to_num gives
// them. One workgroup per signal, a strided set of its entries per invocation.
//
// For the support update (masknmf's _flag_components_for_deletion, _mask_expansion_routine) also the max of each
// signal's values, 0 for a signal without entries, and its brightness max |a_i| max |c_i|.

@group(0) @binding(0)
var<storage, read> a_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> a_pixels: array<u32>;
@group(0) @binding(2)
var<storage, read> a_values: array<f32>;
// the pixel-major view of a, see compute_signal_structures
@group(0) @binding(3)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(5)
var<storage, read> pixel_signals: array<u32>;
// the group of each signal
@group(0) @binding(6)
var<storage, read> signal_groups: array<u32>;
// (m, n, sum of c~, sum of c) per signal
@group(0) @binding(7)
var<storage, read> stats: array<vec4<f32>>;
// max |c| per signal
@group(0) @binding(8)
var<storage, read> max_abs: array<f32>;
@group(0) @binding(9)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(10)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(11)
var<storage, read> products: array<f32>;
@group(0) @binding(12)
var<storage, read> self_products: array<f32>;
@group(0) @binding(13)
var<storage, read> mean: array<f32>;
@group(0) @binding(14)
var<storage, read> norm2: array<f32>;
@group(0) @binding(15)
var<storage, read> x_entries: array<f32>;
@group(0) @binding(16)
var<storage, read_write> support_values: array<f32>;
// (max of the values, brightness) per signal
@group(0) @binding(17)
var<storage, read_write> signal_maxima: array<vec2<f32>>;

struct Sizes {
    n_signals: u32,
    // T s^2, the robust noise term
    noise_term: f32,
}

// uniforms, since they change with the signals and passes
@group(0) @binding(18)
var<uniform> sizes: Sizes;

const wg_size: u32 = 256u;
// the largest finite f32
const f32_max: f32 = 0x1.fffffep+127f;

// c_s . c~_i: s is i or a neighbor of i when both have an entry at a pixel
fn product(i: u32, s: u32) -> f32 {
    if (s == i) {
        return self_products[i];
    }
    var lo = neighbor_ptr[i];
    var hi = neighbor_ptr[i + 1u];
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (neighbors[mid] < s) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return products[lo];
}

@compute @workgroup_size(wg_size)
fn residual_support_values(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.y * nwg.x + wid.x;
    if (i >= sizes.n_signals) {
        return;
    }
    let stats_i = stats[i];
    let group = signal_groups[i];
    var value_max = -f32_max;
    var a_max = 0.0;
    for (var e = a_ptr[i] + lid; e < a_ptr[i + 1u]; e += wg_size) {
        let p = a_pixels[e];
        let a_i = a_values[e];
        var a_c = 0.0;
        var group_c = 0.0;
        for (var e2 = pixel_ptr[p]; e2 < pixel_ptr[p + 1u]; e2++) {
            let s = pixel_signals[e2];
            let a_s = a_values[pixel_entries[e2]];
            let c_s = product(i, s);
            a_c = fma(a_s, c_s, a_c);
            if (signal_groups[s] == group) {
                group_c = fma(a_s, c_s - stats[s].x * stats_i.z, group_c);
            }
        }
        let r = x_entries[e] - a_c - mean[p] * stats_i.z;
        var d2 = norm2[p];
        d2 += (a_i * a_i) * (stats_i.y * stats_i.y);
        d2 += 2.0 * ((stats_i.y * r) * a_i);
        d2 += sizes.noise_term;
        let numerator = r + group_c;
        let over_zero = select(select(0.0, -f32_max, numerator < 0.0), f32_max, numerator > 0.0);
        let value = select(0.0, select(over_zero, numerator / sqrt(d2), d2 > 0.0), norm2[p] >= 0.0 && d2 >= 0.0);
        support_values[e] = value;
        value_max = max(value_max, value);
        a_max = max(a_max, abs(a_i));
    }

    let signal_value_max = workgroup_max(value_max, lid);
    let signal_a_max = workgroup_max(a_max, lid);
    if (lid == 0u) {
        let has_entries = a_ptr[i + 1u] > a_ptr[i];
        signal_maxima[i] = vec2<f32>(select(0.0, signal_value_max, has_entries), signal_a_max * max_abs[i]);
    }
}
