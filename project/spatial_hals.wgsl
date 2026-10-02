// Spatial HALS update of the signals in one group (masknmf spatial_update_hals):
//
//   a[p, i] = max(0, a[p, i] + (U diff_i)[p] - sum_j a[p, j] ctc[j, i] - b[p] sum(c_i) / ||c_i||^2)
//
// for each pixel p in the support of signal i, where ctc[j, i] = c_j . c_i / ||c_i||^2.
// The signals of one group are updated in parallel, one workgroup per signal, groups are dispatched in order.
// Like masknmf, every signal of a group is updated from the values of a before the group. If the signals of
// the group overlap in a, which can happen after a support update, the sum reads a_before, a copy of a made
// before the group. Otherwise it reads a: the entries of the other signals at a pixel are not written during
// the group. Each a[p, i] is written only by the invocation that reads it, so there are no races.

struct Group {
    start: u32,
    count: u32,
    // 1 if the signals of the group overlap in a
    overlaps: u32,
}

@group(0) @binding(0)
var<uniform> group: Group;
@group(0) @binding(1)
var<storage, read> group_signals: array<u32>;

// a, signal-major: signal i has entries a_ptr[i]:a_ptr[i + 1]
@group(0) @binding(2)
var<storage, read> a_ptr: array<u32>;
@group(0) @binding(3)
var<storage, read> a_pixels: array<u32>;
@group(0) @binding(4)
var<storage, read_write> a_values: array<f32>;
// a, pixel-major: entries of pixel p are pixel_entries[pixel_ptr[p]:pixel_ptr[p + 1]]
@group(0) @binding(5)
var<storage, read> pixel_ptr: array<u32>;
@group(0) @binding(6)
var<storage, read> pixel_entries: array<u32>;
@group(0) @binding(7)
var<storage, read> pixel_signals: array<u32>;

// U as cell tiles
@group(0) @binding(8)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(9)
var<storage, read> cell_cols: array<u32>;
@group(0) @binding(10)
var<storage, read> tiles: array<f32>;
@group(0) @binding(11)
var<storage, read> col_block: array<u32>;
@group(0) @binding(12)
var<storage, read> block_col_ptr: array<u32>;

// (signal, block) pairs of signal i are pair_ptr[i]:pair_ptr[i + 1]
@group(0) @binding(13)
var<storage, read> pair_ptr: array<u32>;
@group(0) @binding(14)
var<storage, read> pair_blocks: array<u32>;
@group(0) @binding(15)
var<storage, read> pair_value_offsets: array<u32>;
@group(0) @binding(16)
var<storage, read> diff: array<f32>;

@group(0) @binding(17)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(18)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(19)
var<storage, read> gram: array<f32>;
@group(0) @binding(20)
var<storage, read> c_sum: array<f32>;
@group(0) @binding(21)
var<storage, read> c_sq: array<f32>;
// static baseline
@group(0) @binding(22)
var<storage, read> b: array<f32>;
// values of a before this group if the signals of the group overlap in a, same layout as a_values
@group(0) @binding(23)
var<storage, read> a_before: array<f32>;

override cell_size: u32;
override fov_width: u32;
override max_signal_pairs: u32;
override max_neighbors: u32;

const wg_size: u32 = 64u;

var<workgroup> signal_blocks: array<u32, max_signal_pairs>;
var<workgroup> signal_block_offsets: array<u32, max_signal_pairs>;
var<workgroup> neighbor_ids: array<u32, max_neighbors>;
// ctc[j, i] for each neighbor j
var<workgroup> neighbor_ctc: array<f32, max_neighbors>;

@compute @workgroup_size(wg_size)
fn spatial_hals(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    if (wid.x >= group.count) {
        return;
    }
    let i = group_signals[group.start + wid.x];

    let pair_start = pair_ptr[i];
    let n_pairs = pair_ptr[i + 1u] - pair_start;
    for (var s = lid; s < n_pairs; s += wg_size) {
        signal_blocks[s] = pair_blocks[pair_start + s];
        signal_block_offsets[s] = pair_value_offsets[pair_start + s];
    }

    let c_sq_i = c_sq[i];
    let nb_start = neighbor_ptr[i];
    let n_neighbors = neighbor_ptr[i + 1u] - nb_start;
    for (var e = lid; e < n_neighbors; e += wg_size) {
        neighbor_ids[e] = neighbors[nb_start + e];
        neighbor_ctc[e] = gram[nb_start + e] / c_sq_i;
    }
    workgroupBarrier();

    let b_scale = c_sum[i] / c_sq_i;
    let n_cells_x = fov_width / cell_size;
    let n_cell_pixels = cell_size * cell_size;

    for (var e = a_ptr[i] + lid; e < a_ptr[i + 1u]; e += wg_size) {
        let p = a_pixels[e];
        let row = p / fov_width;
        let col = p % fov_width;
        let cell = (row / cell_size) * n_cells_x + col / cell_size;
        let cell_pixel = (row % cell_size) * cell_size + col % cell_size;

        // (U diff_i)[p], the columns of U at p are the columns of p's cell
        var term1 = 0.0;
        for (var tc = cell_col_ptr[cell]; tc < cell_col_ptr[cell + 1u]; tc++) {
            let k = cell_cols[tc];
            let block = col_block[k];
            var slot = 0u;
            for (; slot < n_pairs; slot++) {
                if (signal_blocks[slot] == block) {
                    break;
                }
            }
            term1 = fma(
                tiles[tc * n_cell_pixels + cell_pixel],
                diff[signal_block_offsets[slot] + k - block_col_ptr[block]],
                term1,
            );
        }

        // sum_j a[p, j] ctc[j, i], ctc[i, i] = 1
        var term2 = 0.0;
        for (var q = pixel_ptr[p]; q < pixel_ptr[p + 1u]; q++) {
            let j = pixel_signals[q];
            var weight = 1.0;
            if (j != i) {
                for (var n = 0u; n < n_neighbors; n++) {
                    if (neighbor_ids[n] == j) {
                        weight = neighbor_ctc[n];
                        break;
                    }
                }
            }
            var a_j: f32;
            if (group.overlaps == 1u) {
                a_j = a_before[pixel_entries[q]];
            } else {
                a_j = a_values[pixel_entries[q]];
            }
            term2 = fma(a_j, weight, term2);
        }

        a_values[e] = max(0.0, a_values[e] + term1 - term2 - b[p] * b_scale);
    }
}
