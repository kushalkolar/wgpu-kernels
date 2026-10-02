// For each pixel p of cell c, with u_p the row of U for p (the cell's tile, see compute_cell_tiles in _compression.py),
// M_c the cell's matrix from cell_gram.wgsl and s a vector over the columns of U:
//
//     x_p = u_p^T M_c u_p    and    y_p = u_p^T s
//
// The epilogue gives the outputs:
//
//     0: masknmf's standard correlation image mean and normalizer, M_c the Gram matrix of the rows of V and s their
//        sums: out0 = sqrt(-2 y_p m_p + m_p^2 n_sum + x_p + noise_term), 0 where masknmf's is nan (a negative
//        radicand), out1 = m_p = y_p / n_sum
//     1: masknmf's robust noise std per pixel (_sketch_robust_variance_term), M_c the Gram matrix over the sampled
//        frames and s the sums over the last frame batch: out0 = sqrt(max(x_p / n_sum - (y_p / n_sum)^2, 0))
//     2: out0 = sqrt(x_p), the norm of the pixel's row of U V' for M_c the Gram matrix of the rows of V'
//
// out0 and out1 are images, pixels in row-major order. One workgroup per cell, M_c and the cell's entries of s staged
// through workgroup memory, each invocation computes 4 horizontally adjacent pixels (the tiles are vec4s of 4 pixels),
// 4 columns of M_c at a time.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read> gram_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> gram: array<vec4<f32>>;
// [rank]
@group(0) @binding(5)
var<storage, read> sums: array<f32>;
@group(0) @binding(6)
var<storage, read_write> out0: array<vec4<f32>>;
@group(0) @binding(7)
var<storage, read_write> out1: array<vec4<f32>>;
// x: the noise term T s^2 of epilogue 0
@group(0) @binding(8)
var<uniform> noise_term: vec4<f32>;

override cell_size: u32;
// number of cells along the fov width and height
override n_cells_x: u32;
override n_cells_y: u32;
override max_cell_cols: u32;
override epilogue: u32;
// number of frames of the sums: the frames (epilogue 0) or the sampled frames (epilogue 1)
override n_sum: f32;

// vec4s of a row of M_c for the largest cell
override nb_max: u32 = (max_cell_cols + 3u) / 4u;

var<workgroup> m_tile: array<vec4<f32>, 4u * nb_max * nb_max>;
var<workgroup> s_tile: array<f32, max_cell_cols>;

@compute @workgroup_size(cell_size * cell_size / 4u)
fn cell_quadratic_form(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_invocations = cell_size * cell_size / 4u;
    let cell = wid.x;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    // rows of M_c are nb vec4s, as in cell_gram.wgsl
    let nb = (n_cols + 3u) / 4u;

    let base = gram_ptr[cell] / 4u;
    for (var i = lid; i < 4u * nb * nb; i += n_invocations) {
        m_tile[i] = gram[base + i];
    }
    for (var k = lid; k < n_cols; k += n_invocations) {
        s_tile[k] = sums[cell_cols[col_start + k]];
    }
    workgroupBarrier();

    // tile column k of this invocation's 4 pixels is tiles[t + k * n_invocations]
    let t = col_start * n_invocations + lid;
    var y = vec4<f32>();
    for (var k = 0u; k < n_cols; k++) {
        y = fma(tiles[t + k * n_invocations], vec4<f32>(s_tile[k]), y);
    }
    var x = vec4<f32>();
    for (var lb = 0u; lb < nb; lb++) {
        // z_i = sum_k u_k M_c[k, 4 lb + i]
        var z0 = vec4<f32>();
        var z1 = vec4<f32>();
        var z2 = vec4<f32>();
        var z3 = vec4<f32>();
        for (var k = 0u; k < n_cols; k++) {
            let u = tiles[t + k * n_invocations];
            let m = m_tile[k * nb + lb];
            z0 = fma(u, vec4<f32>(m.x), z0);
            z1 = fma(u, vec4<f32>(m.y), z1);
            z2 = fma(u, vec4<f32>(m.z), z2);
            z3 = fma(u, vec4<f32>(m.w), z3);
        }
        // x += sum_i z_i u_(4 lb + i) over the columns 4 lb + i < n_cols
        let l = 4u * lb;
        x = fma(z0, tiles[t + l * n_invocations], x);
        if (l + 1u < n_cols) {
            x = fma(z1, tiles[t + (l + 1u) * n_invocations], x);
        }
        if (l + 2u < n_cols) {
            x = fma(z2, tiles[t + (l + 2u) * n_invocations], x);
        }
        if (l + 3u < n_cols) {
            x = fma(z3, tiles[t + (l + 3u) * n_invocations], x);
        }
    }

    // first of the 4 pixels of this invocation
    let width = n_cells_x * cell_size;
    let row = (cell / n_cells_x) * cell_size + (4u * lid) / cell_size;
    let col = (cell % n_cells_x) * cell_size + (4u * lid) % cell_size;
    let p4 = (row * width + col) / 4u;

    if (epilogue == 0u) {
        let mean = y / n_sum;
        var norm = -2.0 * y * mean;
        norm += mean * mean * n_sum;
        norm += x;
        norm += noise_term.x;
        out0[p4] = select(vec4<f32>(), sqrt(norm), norm > vec4<f32>());
        out1[p4] = mean;
    } else if (epilogue == 1u) {
        let mean = y / n_sum;
        out0[p4] = sqrt(max(x / n_sum - mean * mean, vec4<f32>()));
    } else {
        out0[p4] = sqrt(x);
    }
}
