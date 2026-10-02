// one frame of the denoised movie: (U @ V[:, t]) * noise_variance_image + mean_image
//
// U is stored as one dense tile per cell (see _compression.py). All pixels of a cell are supported
// on the same columns of U, so each value V[k, t] that a cell needs is loaded once into workgroup
// memory and shared by every pixel of the cell.
//
// One workgroup per cell, each invocation computes 4 horizontally adjacent pixels of the cell.
// Adjacent invocations read adjacent 16 byte vec4s of a tile column, so the reads of U are coalesced.

// start of each cell's columns in cell_cols, [n_cells + 1]
@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
// column of U for each tile column
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
// each tile column is cell_size * cell_size contiguous values, pixels of a cell in row-major order
@group(0) @binding(2)
var<storage, read> tiles: array<vec4<f32>>;
// V, [rank, n_frames_padded]
@group(0) @binding(3)
var<storage, read> temporal_compressed: array<f32>;

// t, the frame index to compute
@group(0) @binding(4)
var<uniform> t: u32;

// final result is written into a texture so we can visualize it
@group(0) @binding(5) var out_tex: texture_storage_2d<r32float, write>;

@group(0) @binding(6)
var<storage, read> noise_variance_image: array<f32>;
@group(0) @binding(7)
var<storage, read> mean_image: array<f32>;

override cell_size: u32;
// number of cells along the fov width
override n_cells_x: u32;
override n_frames_padded: u32;
// max number of columns of U that support a cell
override max_cell_cols: u32;

// V[k, t] for each column k that supports this cell
var<workgroup> v_t: array<f32, max_cell_cols>;

@compute @workgroup_size(cell_size * cell_size / 4u)
fn uv_frame(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_invocations = cell_size * cell_size / 4u;

    let cell = wid.y * n_cells_x + wid.x;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;

    for (var k = lid; k < n_cols; k += n_invocations) {
        v_t[k] = temporal_compressed[cell_cols[col_start + k] * n_frames_padded + t];
    }
    workgroupBarrier();

    var sum = vec4<f32>(0.0);
    for (var k = 0u; k < n_cols; k++) {
        sum = fma(tiles[(col_start + k) * n_invocations + lid], vec4<f32>(v_t[k]), sum);
    }

    // first of the 4 pixels computed by this invocation
    let row = wid.y * cell_size + (4u * lid) / cell_size;
    let col = wid.x * cell_size + (4u * lid) % cell_size;
    let width = n_cells_x * cell_size;

    for (var i = 0u; i < 4u; i++) {
        let p = row * width + col + i;
        let val = fma(sum[i], noise_variance_image[p], mean_image[p]);
        textureStore(out_tex, vec2u(col + i, row), vec4f(val, 0.0, 0.0, 0.0));
    }
}
