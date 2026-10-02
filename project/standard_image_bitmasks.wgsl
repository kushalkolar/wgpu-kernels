// masknmf's standard correlation images thresholded as in its merge test (_compute_indices_to_merge), a bit per pixel:
//
//     bit(i, p) = image_i(p) > threshold,    image_i(p) = (u_p^T W_i - m_p s_i) / n_p
//
// with u_p the row of U for p, W = V c~ [rank, sizes.y], m and n the mean and normalizer of the standard correlation
// images and s_i the sum of c~_i (StandardCorrelationImages.getitem_tensor). Values that masknmf's nan_to_num sets to 0
// (n_p = 0, or not finite) give 0. Row i of bits holds the bits of signal i for the pixels in cell-major order: n_words
// words per cell, bit b of word w is pixel 32 w + b of the cell in row-major order.
//
// One workgroup per cell and chunk of 64 signals, a small matrix product U_c W_c over the cell's columns of U:
// invocation lid computes 16 signals (lid / 64) of 4 horizontally adjacent pixels at a time (the tiles are vec4s of 4
// pixels, quads lid % 64 + 64 q), the rows of W for the cell's columns and the chunk's signals are staged through
// workgroup memory. The bits of the invocations are combined with atomicOr in workgroup memory.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(2)
var<storage, read> tiles: array<vec4<f32>>;
// [rank, sizes.y]
@group(0) @binding(3)
var<storage, read> w: array<vec4<f32>>;
// images of the pixels in row-major order
@group(0) @binding(4)
var<storage, read> mean: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read> normalizer: array<vec4<f32>>;
// stats of standardize.wgsl, z is the sum of c~_i
@group(0) @binding(6)
var<storage, read> stats: array<vec4<f32>>;
// [n_signals, row_words]
@group(0) @binding(7)
var<storage, read_write> bits: array<u32>;
// x: n_signals, y: the columns of W, a multiple of 4; uniforms, since they change with each support update
@group(0) @binding(8)
var<uniform> sizes: vec4<u32>;

override cell_size: u32;
// number of cells along the fov width and height
override n_cells_x: u32;
override n_cells_y: u32;
override max_cell_cols: u32;
override threshold: f32;
// words per row of bits, at least n_cells * n_words
override row_words: u32;

const wg_size: u32 = 256u;
const chunk_signals: u32 = 64u;

// words of bits per cell and signal
override n_words: u32 = (cell_size * cell_size + 31u) / 32u;

// the chunk's 64 signals for each of the cell's columns of U, 16 vec4s per column
var<workgroup> w_tile: array<vec4<f32>, 16u * max_cell_cols>;
var<workgroup> word_tile: array<atomic<u32>, n_words * chunk_signals>;

// 16 signals of 4 pixels, signal j in sj
struct Signals {
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
    s3: vec4<f32>,
    s4: vec4<f32>,
    s5: vec4<f32>,
    s6: vec4<f32>,
    s7: vec4<f32>,
    s8: vec4<f32>,
    s9: vec4<f32>,
    s10: vec4<f32>,
    s11: vec4<f32>,
    s12: vec4<f32>,
    s13: vec4<f32>,
    s14: vec4<f32>,
    s15: vec4<f32>,
}

// x plus the outer product of 4 pixels u and 16 signals (w0, w1, w2, w3)
fn signals_fma(x: Signals, u: vec4<f32>, w0: vec4<f32>, w1: vec4<f32>, w2: vec4<f32>, w3: vec4<f32>) -> Signals {
    return Signals(
        fma(u, vec4<f32>(w0.x), x.s0),
        fma(u, vec4<f32>(w0.y), x.s1),
        fma(u, vec4<f32>(w0.z), x.s2),
        fma(u, vec4<f32>(w0.w), x.s3),
        fma(u, vec4<f32>(w1.x), x.s4),
        fma(u, vec4<f32>(w1.y), x.s5),
        fma(u, vec4<f32>(w1.z), x.s6),
        fma(u, vec4<f32>(w1.w), x.s7),
        fma(u, vec4<f32>(w2.x), x.s8),
        fma(u, vec4<f32>(w2.y), x.s9),
        fma(u, vec4<f32>(w2.z), x.s10),
        fma(u, vec4<f32>(w2.w), x.s11),
        fma(u, vec4<f32>(w3.x), x.s12),
        fma(u, vec4<f32>(w3.y), x.s13),
        fma(u, vec4<f32>(w3.z), x.s14),
        fma(u, vec4<f32>(w3.w), x.s15),
    );
}

// the 4 bits of pixel quad q for signal j of the chunk, in their word of the cell
fn store_bits(j: u32, x: vec4<f32>, m: vec4<f32>, n: vec4<f32>, q: u32) {
    let image = (x - m * stats[min(j, sizes.x - 1u)].z) / n;
    let above = (n > vec4<f32>()) & (abs(image) <= vec4<f32>(3.4028235e38)) & (image > vec4<f32>(threshold));
    let nibble = dot(vec4<u32>(above), vec4<u32>(1u, 2u, 4u, 8u));
    if (nibble != 0u) {
        atomicOr(&word_tile[n_words * (j % chunk_signals) + q / 8u], nibble << (4u * (q % 8u)));
    }
}

@compute @workgroup_size(wg_size)
fn standard_image_bitmasks(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_quads = cell_size * cell_size / 4u;
    let cell = wid.x;
    let first_signal = wid.y * chunk_signals;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    let w_stride4 = sizes.y / 4u;

    // rows of W for the cell's columns and the chunk's signals, signals beyond the columns of W read as the last one
    for (var i = lid; i < 16u * n_cols; i += wg_size) {
        let j4 = min(first_signal / 4u + i % 16u, w_stride4 - 1u);
        w_tile[i] = w[cell_cols[col_start + i / 16u] * w_stride4 + j4];
    }
    for (var i = lid; i < n_words * chunk_signals; i += wg_size) {
        atomicStore(&word_tile[i], 0u);
    }
    workgroupBarrier();

    // signals 16 group:16 group + 16 of the chunk, 4 vec4s of each row of w_tile
    let group = lid / 64u;
    let j = first_signal + 16u * group;
    let width = n_cells_x * cell_size;
    for (var q = lid % 64u; q < n_quads; q += 64u) {
        let t = col_start * n_quads + q;
        var x = Signals();
        for (var k = 0u; k < n_cols; k++) {
            let base = 16u * k + 4u * group;
            x = signals_fma(
                x, tiles[t + k * n_quads], w_tile[base], w_tile[base + 1u], w_tile[base + 2u], w_tile[base + 3u]
            );
        }

        // mean and normalizer of the 4 pixels
        let row = (cell / n_cells_x) * cell_size + (4u * q) / cell_size;
        let col = (cell % n_cells_x) * cell_size + (4u * q) % cell_size;
        let p4 = (row * width + col) / 4u;
        let m = mean[p4];
        let n = normalizer[p4];
        store_bits(j, x.s0, m, n, q);
        store_bits(j + 1u, x.s1, m, n, q);
        store_bits(j + 2u, x.s2, m, n, q);
        store_bits(j + 3u, x.s3, m, n, q);
        store_bits(j + 4u, x.s4, m, n, q);
        store_bits(j + 5u, x.s5, m, n, q);
        store_bits(j + 6u, x.s6, m, n, q);
        store_bits(j + 7u, x.s7, m, n, q);
        store_bits(j + 8u, x.s8, m, n, q);
        store_bits(j + 9u, x.s9, m, n, q);
        store_bits(j + 10u, x.s10, m, n, q);
        store_bits(j + 11u, x.s11, m, n, q);
        store_bits(j + 12u, x.s12, m, n, q);
        store_bits(j + 13u, x.s13, m, n, q);
        store_bits(j + 14u, x.s14, m, n, q);
        store_bits(j + 15u, x.s15, m, n, q);
    }
    workgroupBarrier();

    // the words of the chunk's signals for this cell
    for (var i = lid; i < n_words * chunk_signals; i += wg_size) {
        let signal = first_signal + i / n_words;
        if (signal < sizes.x) {
            bits[signal * row_words + cell * n_words + i % n_words] = atomicLoad(&word_tile[i]);
        }
    }
}
