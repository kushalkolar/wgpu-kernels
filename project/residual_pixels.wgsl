// masknmf's per-pixel parts of the residual correlation images (_compute_residual_correlation_image), of the
// residual R = U V' - a c^T with V' = V - q0 q1. For each pixel p, with u_p the row of U for p and the sums over the
// signals i, s with an entry at p:
//
//     mean_p = u_p^T v' / T - sum_i a_pi m_i
//     norm2_p = -2 (u_p^T v') mean_p + 2 (sum_i a_pi S_i) mean_p + T mean_p^2
//               - 2 sum_i a_pi u_p^T V' c_i + sum_i a_pi sum_s a_ps c_s . c_i + uv_norms_p^2
//     normalizer_p = sqrt(norm2_p + noise_term)
//
// in masknmf's order of the terms, v' = V' 1 the sums of the rows of V'. The traces enter through their standardized
// versions (standardize.wgsl: m the mean, n the norm of c - m, S the sum of c): u_p^T V' c_i = n_i x_i + m_i u_p^T v'
// with x_i = u_p^T W'_i, W' = V' c~, and c_s . c_i = n_i (c_s . c~_i) + m_i S_s with the products of
// trace_products.wgsl. x_i is also written for each entry of a, for residual_support_values.wgsl. The normalizer is 0
// where masknmf's is nan (norm2_p + noise_term < 0). One workgroup per cell, one pixel per invocation.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<f32>;
// W' [rank, sizes.w_stride]
@group(0) @binding(3)
var<storage, read> w: array<f32>;
// v' [rank]
@group(0) @binding(4)
var<storage, read> v_sums: array<f32>;
// the pixel-major view of a, see compute_signal_structures
@group(0) @binding(5)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(6)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(7)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(8)
var<storage, read> a_values: array<f32>;
// (m, n, sum of c~, S) per signal
@group(0) @binding(9)
var<storage, read> stats: array<vec4<f32>>;
@group(0) @binding(10)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(11)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(12)
var<storage, read> products: array<f32>;
@group(0) @binding(13)
var<storage, read> self_products: array<f32>;
@group(0) @binding(14)
var<storage, read> uv_norms: array<f32>;
@group(0) @binding(15)
var<storage, read_write> mean: array<f32>;
@group(0) @binding(16)
var<storage, read_write> normalizer: array<f32>;
@group(0) @binding(17)
var<storage, read_write> norm2: array<f32>;
// x_i of each entry of a, signal-major
@group(0) @binding(18)
var<storage, read_write> x_entries: array<f32>;

struct Sizes {
    w_stride: u32,
    // T s^2, the robust noise term
    noise_term: f32,
}

// uniforms, since they change with the signals and passes
@group(0) @binding(19)
var<uniform> sizes: Sizes;

override cell_size: u32;
// number of cells along the fov width
override n_cells_x: u32;
override n_frames: f32;

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

@compute @workgroup_size(cell_size * cell_size)
fn residual_pixels(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_cell_pixels = cell_size * cell_size;
    let cell = wid.x;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    let width = n_cells_x * cell_size;
    let row = (cell / n_cells_x) * cell_size + lid / cell_size;
    let p = row * width + (cell % n_cells_x) * cell_size + lid % cell_size;
    // tile column k of this pixel is tiles[t + k * n_cell_pixels]
    let t = col_start * n_cell_pixels + lid;

    var y = 0.0;
    for (var k = 0u; k < n_cols; k++) {
        y = fma(tiles[t + k * n_cell_pixels], v_sums[cell_cols[col_start + k]], y);
    }

    let e_start = pixel_ptr[p];
    let e_end = pixel_ptr[p + 1u];
    var a_mean = 0.0;
    var a_sum = 0.0;
    for (var e = e_start; e < e_end; e++) {
        let a = a_values[pixel_entries[e]];
        let s = stats[pixel_signals[e]];
        a_mean = fma(a, s.x, a_mean);
        a_sum = fma(a, s.w, a_sum);
    }
    let m = y / n_frames - a_mean;

    var n = -2.0 * (y * m);
    n += 2.0 * (a_sum * m);
    n += n_frames * (m * m);

    // the signals' terms
    var cross = 0.0;
    var quadratic = 0.0;
    for (var e = e_start; e < e_end; e++) {
        let entry = pixel_entries[e];
        let i = pixel_signals[e];
        let a_i = a_values[entry];
        let stats_i = stats[i];
        var x = 0.0;
        for (var k = 0u; k < n_cols; k++) {
            x = fma(tiles[t + k * n_cell_pixels], w[cell_cols[col_start + k] * sizes.w_stride + i], x);
        }
        x_entries[entry] = x;
        cross = fma(a_i, fma(stats_i.y, x, stats_i.x * y), cross);

        var a_c = 0.0;
        for (var e2 = e_start; e2 < e_end; e2++) {
            let s = pixel_signals[e2];
            let c_s_c_i = fma(stats_i.y, product(i, s), stats_i.x * stats[s].w);
            a_c = fma(a_values[pixel_entries[e2]], c_s_c_i, a_c);
        }
        quadratic = fma(a_i, a_c, quadratic);
    }
    n += -2.0 * cross;
    n += quadratic;
    n += uv_norms[p] * uv_norms[p];

    mean[p] = m;
    norm2[p] = n;
    normalizer[p] = select(0.0, sqrt(n + sizes.noise_term), n + sizes.noise_term > 0.0);
}
