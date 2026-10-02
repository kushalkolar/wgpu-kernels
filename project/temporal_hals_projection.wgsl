// Quantities of masknmf's temporal_update_hals that only depend on a, one workgroup per signal i:
//
//   w_i = a_i^T U, one value per column of U in each block that overlaps the support of i, written to the
//         weights of the work items of temporal_hals_partials
//   ring_w_i = w_i q0, [ring_rank_padded]
//   atb_i = a_i . b
//   a_sq_i = a_i . a_i
//   ata[e] = a_i . a_j for each edge e = (i, j) of the overlap graph
//
// Each value of w is computed by one invocation as a sum over the pixels of a_i that lie in the block,
// so no atomics are needed and the result is deterministic.

// a, signal-major
@group(0) @binding(0)
var<storage, read> a_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> a_pixels: array<u32>;
@group(0) @binding(2)
var<storage, read> a_values: array<f32>;
// a, pixel-major
@group(0) @binding(3)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(5)
var<storage, read> pixel_signals: array<u32>;

// U as cell tiles
@group(0) @binding(6)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(7)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(8)
var<storage, read> tiles: array<f32>;
@group(0) @binding(9)
var<storage, read> block_col_ptr: array<u32>;
// (row, col, row_end, col_end) of each block
@group(0) @binding(10)
var<storage, read> block_rects: array<u32>;

@group(0) @binding(11)
var<storage, read> pair_ptr: array<u32>;
@group(0) @binding(12)
var<storage, read> pair_blocks: array<u32>;
// start of each pair's weights in w, the weight of column r of the pair's block is at w_offsets + 16 r
@group(0) @binding(13)
var<storage, read> w_offsets: array<u32>;
@group(0) @binding(14)
var<storage, read_write> w: array<f32>;

// factorized ring term q0, [rank, ring_rank_padded]
@group(0) @binding(15)
var<storage, read> ring_left: array<f32>;
// [n_signals, ring_rank_padded]
@group(0) @binding(16)
var<storage, read_write> ring_w: array<f32>;

// static baseline
@group(0) @binding(17)
var<storage, read> b: array<f32>;
@group(0) @binding(18)
var<storage, read_write> atb: array<f32>;
@group(0) @binding(19)
var<storage, read_write> a_sq: array<f32>;

@group(0) @binding(20)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(21)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(22)
var<storage, read_write> ata: array<f32>;
// 0 if there is no ring term, a uniform so that a change of the ring rank does not recreate the pipeline
@group(0) @binding(23)
var<uniform> ring_rank_padded: u32;

override cell_size: u32;
override fov_width: u32;
override max_signal_pairs: u32;
override max_block_cols: u32;

const wg_size: u32 = 64u;

// w_i and the column of U of each value
var<workgroup> w_values: array<f32, max_signal_pairs * max_block_cols>;
var<workgroup> w_cols: array<u32, max_signal_pairs * max_block_cols>;
// start of each pair's values in w_values
var<workgroup> slot_offsets: array<u32, max_signal_pairs + 1u>;

// position of column k in the (sorted) columns of a cell
fn find_cell_col(cell: u32, k: u32) -> u32 {
    var lo = cell_col_ptr[cell];
    var hi = cell_col_ptr[cell + 1u];
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (cell_cols[mid] < k) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

@compute @workgroup_size(wg_size)
fn temporal_hals_projection(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = wid.y * nwg.x + wid.x;
    if (i >= arrayLength(&atb)) {
        return;
    }

    let pair_start = pair_ptr[i];
    let n_pairs = pair_ptr[i + 1u] - pair_start;
    let entry_start = a_ptr[i];
    let entry_end = a_ptr[i + 1u];
    let n_cells_x = fov_width / cell_size;
    let n_cell_pixels = cell_size * cell_size;

    if (lid == 0u) {
        var offset = 0u;
        for (var s = 0u; s < n_pairs; s++) {
            slot_offsets[s] = offset;
            let block = pair_blocks[pair_start + s];
            offset += block_col_ptr[block + 1u] - block_col_ptr[block];
        }
        slot_offsets[n_pairs] = offset;
    }
    workgroupBarrier();

    let n_w = slot_offsets[n_pairs];
    for (var idx = lid; idx < n_w; idx += wg_size) {
        var slot = 0u;
        for (; slot + 1u < n_pairs; slot++) {
            if (idx < slot_offsets[slot + 1u]) {
                break;
            }
        }
        let block = pair_blocks[pair_start + slot];
        let k = block_col_ptr[block] + idx - slot_offsets[slot];
        let row0 = block_rects[4u * block];
        let col0 = block_rects[4u * block + 1u];
        let row1 = block_rects[4u * block + 2u];
        let col1 = block_rects[4u * block + 3u];

        var sum = 0.0;
        for (var e = entry_start; e < entry_end; e++) {
            let p = a_pixels[e];
            let row = p / fov_width;
            let col = p % fov_width;
            if (row < row0 || row >= row1 || col < col0 || col >= col1) {
                continue;
            }
            let cell = (row / cell_size) * n_cells_x + col / cell_size;
            let cell_pixel = (row % cell_size) * cell_size + col % cell_size;
            let tc = find_cell_col(cell, k);
            sum = fma(a_values[e], tiles[tc * n_cell_pixels + cell_pixel], sum);
        }

        w_values[idx] = sum;
        w_cols[idx] = k;
        w[w_offsets[pair_start + slot] + 16u * (idx - slot_offsets[slot])] = sum;
    }
    workgroupBarrier();

    // ring_w_i = w_i q0
    for (var rho = lid; rho < ring_rank_padded; rho += wg_size) {
        var sum = 0.0;
        for (var idx = 0u; idx < n_w; idx++) {
            sum = fma(w_values[idx], ring_left[w_cols[idx] * ring_rank_padded + rho], sum);
        }
        ring_w[i * ring_rank_padded + rho] = sum;
    }

    // a_i . b and a_i . a_i
    var ab = 0.0;
    var aa = 0.0;
    for (var e = entry_start + lid; e < entry_end; e += wg_size) {
        let value = a_values[e];
        ab = fma(value, b[a_pixels[e]], ab);
        aa = fma(value, value, aa);
    }
    let total_ab = workgroup_sum(ab, lid);
    let total_aa = workgroup_sum(aa, lid);
    if (lid == 0u) {
        atb[i] = total_ab;
        a_sq[i] = total_aa;
    }

    // a_i . a_j for each neighbor j, over the pixels of a_i
    for (var edge = neighbor_ptr[i] + lid; edge < neighbor_ptr[i + 1u]; edge += wg_size) {
        let j = neighbors[edge];
        var sum = 0.0;
        for (var e = entry_start; e < entry_end; e++) {
            let p = a_pixels[e];
            for (var q = pixel_ptr[p]; q < pixel_ptr[p + 1u]; q++) {
                if (pixel_signals[q] == j) {
                    sum = fma(a_values[e], a_values[pixel_entries[q]], sum);
                    break;
                }
            }
        }
        ata[edge] = sum;
    }
}
