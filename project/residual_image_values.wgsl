// masknmf's residual correlation image of each signal (ResidualCorrelationImages.getitem_tensor) at given pixels: for
// each entry (p, i) of a list of (pixel, signal) coordinates sorted by signal, the support value of the entry (p, i) of
// a if a has one (residual_support_values.wgsl), otherwise
//
//     (u_p^T W'_i - sum_s a_ps c_s . c~_i - mean_p s_i) / normalizer_p,        0 where normalizer_p is 0
//
// with W' = V' c~, mean_p and normalizer_p from residual_pixels.wgsl (0 where masknmf's normalizer is nan, which its
// nan_to_num turns into 0 as it does the values over a normalizer of 0), s_i the sum of c~_i (standardize.wgsl) and
// the sum over the signals s with an entry of a at p, whose products trace_products.wgsl computes for a graph with
// them among the neighbors of i (compute_dilated_overlap_graph). One workgroup per signal, a strided set of its
// entries per invocation.

// the entries of signal i are entry_pixels[entry_ptr[i]:entry_ptr[i + 1]]
@group(0) @binding(0)
var<storage, read> entry_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> entry_pixels: array<u32>;
// the index of the entry of a at each entry, no_entry if there is none
@group(0) @binding(2)
var<storage, read> a_entries: array<u32>;
@group(0) @binding(3)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(5)
var<storage, read> tiles: array<f32>;
// W' [rank, sizes.w_stride]
@group(0) @binding(6)
var<storage, read> w: array<f32>;
@group(0) @binding(7)
var<storage, read> mean: array<f32>;
@group(0) @binding(8)
var<storage, read> normalizer: array<f32>;
// (m, n, sum of c~, sum of c) per signal
@group(0) @binding(9)
var<storage, read> stats: array<vec4<f32>>;
// the pixel-major view of a, see compute_signal_structures
@group(0) @binding(10)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(11)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(12)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(13)
var<storage, read> a_values: array<f32>;
@group(0) @binding(14)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(15)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(16)
var<storage, read> products: array<f32>;
@group(0) @binding(17)
var<storage, read> support_values: array<f32>;
@group(0) @binding(18)
var<storage, read_write> values: array<f32>;

struct Sizes {
    n_signals: u32,
    w_stride: u32,
}

// uniforms, since they change with the signals
@group(0) @binding(19)
var<uniform> sizes: Sizes;

override cell_size: u32;
// number of cells along the fov width
override n_cells_x: u32;

const wg_size: u32 = 256u;
const no_entry: u32 = 0xffffffffu;

// c_s . c~_i for a neighbor s of i
fn product(i: u32, s: u32) -> f32 {
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
fn residual_image_values(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.y * nwg.x + wid.x;
    if (i >= sizes.n_signals) {
        return;
    }
    let n_cell_pixels = cell_size * cell_size;
    let width = n_cells_x * cell_size;
    let tilde_sum = stats[i].z;
    for (var e = entry_ptr[i] + lid; e < entry_ptr[i + 1u]; e += wg_size) {
        let entry = a_entries[e];
        if (entry != no_entry) {
            values[e] = support_values[entry];
            continue;
        }

        // u_p^T W'_i over the columns of U of the cell of p
        let p = entry_pixels[e];
        let row = p / width;
        let col = p % width;
        let cell = (row / cell_size) * n_cells_x + col / cell_size;
        let col_start = cell_col_ptr[cell];
        let n_cols = cell_col_ptr[cell + 1u] - col_start;
        // tile column k of this pixel is tiles[t + k * n_cell_pixels]
        let t = col_start * n_cell_pixels + (row % cell_size) * cell_size + col % cell_size;
        var x = 0.0;
        for (var k = 0u; k < n_cols; k++) {
            x = fma(tiles[t + k * n_cell_pixels], w[cell_cols[col_start + k] * sizes.w_stride + i], x);
        }

        // the signals with an entry at p, which i has not
        var a_c = 0.0;
        for (var e2 = pixel_ptr[p]; e2 < pixel_ptr[p + 1u]; e2++) {
            a_c = fma(a_values[pixel_entries[e2]], product(i, pixel_signals[e2]), a_c);
        }
        let numerator = x - a_c - mean[p] * tilde_sum;
        values[e] = select(0.0, numerator / normalizer[p], normalizer[p] > 0.0);
    }
}
