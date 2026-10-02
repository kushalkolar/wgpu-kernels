// frame t of masknmf's PMD, AC and residual arrays without rescaling: U V[:, t], a c[t] and
// U V[:, t] - U x - a c[t] - b, x = q0 q1[:, t] the ring term of the frame on the columns of U (ring_frame.wgsl)
//
// U as in uv_frame.wgsl: one dense tile per cell, the values V[k, t] and x[k] of the columns k that support a cell are
// loaded once into workgroup memory and shared by every pixel of the cell. One workgroup per cell, each invocation
// computes 4 horizontally adjacent pixels of the cell. a c[t] is summed over the entries of a at each pixel, in the
// pixel-major view of a (see compute_signal_structures).

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
@group(0) @binding(4)
var<uniform> t: u32;
// [rank]
@group(0) @binding(5)
var<storage, read> x: array<f32>;
@group(0) @binding(6)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(7)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(8)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(9)
var<storage, read> a_values: array<f32>;
// c, [n_signals, n_frames_padded]
@group(0) @binding(10)
var<storage, read> temporal_demixed: array<f32>;
@group(0) @binding(11)
var<storage, read> b: array<f32>;

@group(0) @binding(12) var pmd_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(13) var ac_texture: texture_storage_2d<r32float, write>;
@group(0) @binding(14) var residual_texture: texture_storage_2d<r32float, write>;

override cell_size: u32;
// number of cells along the fov width
override n_cells_x: u32;
override n_frames_padded: u32;
// max number of columns of U that support a cell
override max_cell_cols: u32;

// V[k, t] and x[k] for each column k that supports this cell
var<workgroup> v_t: array<f32, max_cell_cols>;
var<workgroup> x_t: array<f32, max_cell_cols>;

@compute @workgroup_size(cell_size * cell_size / 4u)
fn demixing_frames(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_invocations = cell_size * cell_size / 4u;

    let cell = wid.y * n_cells_x + wid.x;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;

    for (var k = lid; k < n_cols; k += n_invocations) {
        let col = cell_cols[col_start + k];
        v_t[k] = temporal_compressed[col * n_frames_padded + t];
        x_t[k] = x[col];
    }
    workgroupBarrier();

    var uv = vec4<f32>(0.0);
    var ux = vec4<f32>(0.0);
    for (var k = 0u; k < n_cols; k++) {
        let u = tiles[(col_start + k) * n_invocations + lid];
        uv = fma(u, vec4<f32>(v_t[k]), uv);
        ux = fma(u, vec4<f32>(x_t[k]), ux);
    }

    // first of the 4 pixels computed by this invocation
    let row = wid.y * cell_size + (4u * lid) / cell_size;
    let col = wid.x * cell_size + (4u * lid) % cell_size;
    let width = n_cells_x * cell_size;

    for (var i = 0u; i < 4u; i++) {
        let p = row * width + col + i;
        var ac = 0.0;
        for (var e = pixel_ptr[p]; e < pixel_ptr[p + 1u]; e++) {
            ac = fma(a_values[pixel_entries[e]], temporal_demixed[pixel_signals[e] * n_frames_padded + t], ac);
        }
        let pixel = vec2u(col + i, row);
        textureStore(pmd_texture, pixel, vec4f(uv[i], 0.0, 0.0, 0.0));
        textureStore(ac_texture, pixel, vec4f(ac, 0.0, 0.0, 0.0));
        textureStore(residual_texture, pixel, vec4f(uv[i] - ux[i] - ac - b[p], 0.0, 0.0, 0.0));
    }
}
