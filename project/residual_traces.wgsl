// The residual movie of a band of the fov, one row of cells, for local correlation images: for the pixels p of the
// band, q their row-major index in the band, traces[q, t] = u_p^T V'[:, t] - sum_s a_ps c_s(t) for the frames
// t < 4 n4, with V' = V - q0 q1 (subtract_ring.wgsl) or V without a ring term, 0 beyond the movie's frames as V' and c
// are.
//
// One workgroup per cell of the band and chunk of 4 chunk4 frames, one invocation per two pixels of the cell, half a
// cell apart. The chunk of V' for the cell's columns of U is staged in workgroup memory, each of its values read once
// for both pixels, then each invocation sums the pixels' rows of the cell's tile times it and subtracts a c^T of the
// signals with an entry at each pixel.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<f32>;
// V' [rank, n4]
@group(0) @binding(3)
var<storage, read> v: array<vec4<f32>>;
// the pixel-major view of a, see compute_signal_structures
@group(0) @binding(4)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(5)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(6)
var<storage, read> pixel_signals: array<u32>;
@group(0) @binding(7)
var<storage, read> a_values: array<f32>;
// c [n_signals, n4]
@group(0) @binding(8)
var<storage, read> c: array<vec4<f32>>;
// [n_cells_x cell_size^2, n4]
@group(0) @binding(9)
var<storage, read_write> traces: array<vec4<f32>>;
// x: the row of cells of the band, a uniform with a row per band
@group(0) @binding(10)
var<uniform> band: vec4<u32>;

override cell_size: u32;
// number of cells along the fov width
override n_cells_x: u32;
override max_cell_cols: u32;
// vec4s per row of V', c and traces
override n4: u32;

const chunk4: u32 = 8u;

// the chunk of V' for the cell's columns
var<workgroup> v_chunk: array<vec4<f32>, max_cell_cols * chunk4>;

// workgroup x indexes the chunks of frames, workgroup y the cells of the band
@compute @workgroup_size(cell_size * cell_size / 2u)
fn residual_traces(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_cell_pixels = cell_size * cell_size;
    let n_invocations = n_cell_pixels / 2u;
    let cell = band.x * n_cells_x + wid.y;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    let first4 = wid.x * chunk4;

    for (var i = lid; i < n_cols * chunk4; i += n_invocations) {
        let col = cell_cols[col_start + i / chunk4];
        v_chunk[i] = v[col * n4 + min(first4 + i % chunk4, n4 - 1u)];
    }
    workgroupBarrier();

    // this invocation's pixels lid and lid + n_invocations of the cell, rows half a cell apart
    let t = col_start * n_cell_pixels + lid;
    var acc0: array<vec4<f32>, chunk4>;
    var acc1: array<vec4<f32>, chunk4>;
    for (var k = 0u; k < n_cols; k++) {
        let u0 = vec4<f32>(tiles[t + k * n_cell_pixels]);
        let u1 = vec4<f32>(tiles[t + n_invocations + k * n_cell_pixels]);
        for (var j = 0u; j < chunk4; j++) {
            let x = v_chunk[k * chunk4 + j];
            acc0[j] = fma(u0, x, acc0[j]);
            acc1[j] = fma(u1, x, acc1[j]);
        }
    }

    finish(acc0, lid, wid.y, first4);
    finish(acc1, lid + n_invocations, wid.y, first4);
}

// subtracts a c^T at the pixel of the cell and writes its traces
fn finish(sums: array<vec4<f32>, chunk4>, pixel: u32, cell_x: u32, first4: u32) {
    var acc = sums;
    let width = n_cells_x * cell_size;
    let row = pixel / cell_size;
    let col = cell_x * cell_size + pixel % cell_size;
    let p = (band.x * cell_size + row) * width + col;
    for (var e = pixel_ptr[p]; e < pixel_ptr[p + 1u]; e++) {
        let a = vec4<f32>(-a_values[pixel_entries[e]]);
        let s = pixel_signals[e] * n4 + first4;
        for (var j = 0u; j < chunk4; j++) {
            acc[j] = fma(a, c[s + min(j, n4 - 1u - first4)], acc[j]);
        }
    }
    let q = (row * width + col) * n4 + first4;
    for (var j = 0u; j < chunk4; j++) {
        if (first4 + j < n4) {
            traces[q + j] = acc[j];
        }
    }
}
