// partial[c, j] = sum over the pixels p of the cell of U[p, col(c)] * scale[p] * x[j, p], for each tile column c
//
// The transpose of the spatial matrix U (stored as one dense tile per cell, see _compression.py) times pixel data x,
// [n_j, n_pixels] (a column per row), the pixels scaled by scale. The tile columns of a column of U lie in several
// cells, their partial sums are added by col_reduce.
//
// One workgroup per cell and chunk of 32 columns j: invocation lid % 64 computes the tile columns lid % 64 + 64 i of the
// cell, lid / 64 selects 8 of the columns j, whose values of x are the same for 64 consecutive invocations. Each
// invocation reads its own tile column, which the other invocations of the workgroup read too. Workgroup x indexes
// the chunks of j, so that consecutive workgroups read the same tiles. Staging x or the tiles through workgroup memory
// was slower on the Radeon 780M.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> tiles: array<vec4<f32>>;
// [n_j, n_pixels / 4]
@group(0) @binding(2)
var<storage, read> x: array<vec4<f32>>;
// [n_pixels / 4]
@group(0) @binding(3)
var<storage, read> scale: array<vec4<f32>>;
// [n_tile_cols, n_j]
@group(0) @binding(4)
var<storage, read_write> partial: array<vec4<f32>>;

override cell_size: u32;
// number of cells along the fov width and height
override n_cells_x: u32;
override n_cells_y: u32;
// rows of x and columns of partial, a multiple of 8
override n_j: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn cell_spmm_t(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_cell_pixels4 = cell_size * cell_size / 4u;
    let n_j4 = n_j / 4u;
    let cell = wid.y;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    let width = n_cells_x * cell_size;
    let n_pixels4 = width * n_cells_y * cell_size / 4u;
    // vec4 of the cell's first pixel, rows of the cell are cell_size / 4 vec4s apart by width / 4
    let cell_p4 = ((cell / n_cells_x) * cell_size * width + (cell % n_cells_x) * cell_size) / 4u;
    let row4 = cell_size / 4u;
    let j0 = 32u * wid.x;
    // columns j..j + 8 of this invocation
    let j = j0 + 8u * (lid / 64u);
    if (j >= n_j) {
        return;
    }
    for (var c = lid % 64u; c < n_cols; c += 64u) {
        var sum0 = vec4<f32>(0.0);
        var sum1 = vec4<f32>(0.0);
        for (var q = 0u; q < n_cell_pixels4; q++) {
            let p4 = cell_p4 + (q / row4) * (width / 4u) + q % row4;
            let t = tiles[(col_start + c) * n_cell_pixels4 + q] * scale[p4];
            let x0 = j * n_pixels4 + p4;
            sum0 += vec4<f32>(
                dot(t, x[x0]),
                dot(t, x[x0 + n_pixels4]),
                dot(t, x[x0 + 2u * n_pixels4]),
                dot(t, x[x0 + 3u * n_pixels4]),
            );
            sum1 += vec4<f32>(
                dot(t, x[x0 + 4u * n_pixels4]),
                dot(t, x[x0 + 5u * n_pixels4]),
                dot(t, x[x0 + 6u * n_pixels4]),
                dot(t, x[x0 + 7u * n_pixels4]),
            );
        }
        partial[(col_start + c) * n_j4 + j / 4u] = sum0;
        partial[(col_start + c) * n_j4 + j / 4u + 1u] = sum1;
    }
}
