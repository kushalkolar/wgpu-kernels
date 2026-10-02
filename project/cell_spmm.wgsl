// out[j, p] = sum_k U[p, k] phi[k, j] - sum_s a[p, s] psi[s, j], out is [n_j, n_pixels] (a column per row)
//
// U is stored as one dense tile per cell (see _compression.py), a is the pixel-major view of the spatial footprints
// (see compute_signal_structures in _hals.py), the a term only with the override with_signals. One workgroup per cell
// and chunk of 8 columns j: the rows of phi of the cell's columns of U are staged through workgroup memory and each
// invocation computes 4 horizontally adjacent pixels of the cell, as in uv_frame.wgsl. Workgroup x indexes the chunks
// of j, so that consecutive workgroups read the same tiles.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<vec4<f32>>;
// [rank, n_j]
@group(0) @binding(3)
var<storage, read> phi: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(5)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(6)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(7)
var<storage, read> a_values: array<f32>;
// [n_signals, n_j]
@group(0) @binding(8)
var<storage, read> psi: array<vec4<f32>>;
// [n_j, n_pixels / 4]
@group(0) @binding(9)
var<storage, read_write> out: array<vec4<f32>>;

override cell_size: u32;
// number of cells along the fov width and height
override n_cells_x: u32;
override n_cells_y: u32;
override max_cell_cols: u32;
// columns of phi, psi and rows of out, a multiple of 8
override n_j: u32;
override with_signals: bool;

// 8 columns of phi for each column of U that supports the cell
var<workgroup> phi_tile: array<array<vec4<f32>, 2>, max_cell_cols>;

// columns j0:j0 + 8 of 4 pixels
struct Columns {
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
    c4: vec4<f32>,
    c5: vec4<f32>,
    c6: vec4<f32>,
    c7: vec4<f32>,
}

// x plus the outer product of 4 pixels t and 8 columns (p0, p1)
fn columns_fma(x: Columns, t: vec4<f32>, p0: vec4<f32>, p1: vec4<f32>) -> Columns {
    return Columns(
        fma(t, vec4<f32>(p0.x), x.c0),
        fma(t, vec4<f32>(p0.y), x.c1),
        fma(t, vec4<f32>(p0.z), x.c2),
        fma(t, vec4<f32>(p0.w), x.c3),
        fma(t, vec4<f32>(p1.x), x.c4),
        fma(t, vec4<f32>(p1.y), x.c5),
        fma(t, vec4<f32>(p1.z), x.c6),
        fma(t, vec4<f32>(p1.w), x.c7),
    );
}

@compute @workgroup_size(cell_size * cell_size / 4u)
fn cell_spmm(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_invocations = cell_size * cell_size / 4u;
    let n_j4 = n_j / 4u;
    let j4 = 2u * wid.x;
    let cell = wid.y;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;

    for (var k = lid; k < n_cols; k += n_invocations) {
        let row = cell_cols[col_start + k] * n_j4 + j4;
        phi_tile[k][0] = phi[row];
        phi_tile[k][1] = phi[row + 1u];
    }
    workgroupBarrier();

    var sum = Columns();
    for (var k = 0u; k < n_cols; k++) {
        sum = columns_fma(sum, tiles[(col_start + k) * n_invocations + lid], phi_tile[k][0], phi_tile[k][1]);
    }

    // first of the 4 pixels of this invocation
    let width = n_cells_x * cell_size;
    let row = (cell / n_cells_x) * cell_size + (4u * lid) / cell_size;
    let col = (cell % n_cells_x) * cell_size + (4u * lid) % cell_size;
    let p0 = row * width + col;

    if (with_signals) {
        for (var i = 0u; i < 4u; i++) {
            // the a term of pixel p0 + i, as the outer product of a one-hot vector and the rows of psi
            let one_hot = vec4<f32>(vec4<u32>(i) == vec4<u32>(0u, 1u, 2u, 3u));
            for (var e = pixel_ptr[p0 + i]; e < pixel_ptr[p0 + i + 1u]; e++) {
                let row = pixel_signals[e] * n_j4 + j4;
                sum = columns_fma(sum, -a_values[pixel_entries[e]] * one_hot, psi[row], psi[row + 1u]);
            }
        }
    }

    let n_pixels4 = width * n_cells_y * cell_size / 4u;
    let o = 4u * j4 * n_pixels4 + p0 / 4u;
    out[o] = sum.c0;
    out[o + n_pixels4] = sum.c1;
    out[o + 2u * n_pixels4] = sum.c2;
    out[o + 3u * n_pixels4] = sum.c3;
    out[o + 4u * n_pixels4] = sum.c4;
    out[o + 5u * n_pixels4] = sum.c5;
    out[o + 6u * n_pixels4] = sum.c6;
    out[o + 7u * n_pixels4] = sum.c7;
}
