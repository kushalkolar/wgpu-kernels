// partial[row, t] = w_row . V[columns of the superblock, t] for each (signal, superblock) pair, over all frames.
// Summed over the superblocks of each signal by temporal_hals this is (a_i^T U V)[t], the a^T U V term of
// masknmf's temporal_update_hals. A superblock is a square of blocks of U, so there are fewer partial rows to
// write and read than (signal, block) pairs.
//
// One workgroup per (work item, wg_size vec4s of frames), a work item is a superblock and up to 16 of its
// signals. Each invocation computes one vec4 of frames of the item's 16 partial rows in registers, reading the
// rows of V of the superblock's columns directly: adjacent invocations read adjacent vec4s of a row, and all
// invocations read the same weights. The weights of a work item are dense, [column of the superblock,
// 16 signals], zero where a signal does not overlap the column's block.

struct Item {
    superblock: u32,
    // the partial rows of a work item are consecutive
    first_row: u32,
    n_rows: u32,
    // start of the work item's weights in w, in vec4s
    w_start: u32,
}

@group(0) @binding(0)
var<storage, read> items: array<Item>;
// V, [rank, n_frames_padded]
@group(0) @binding(1)
var<storage, read> temporal_compressed: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read> block_col_ptr: array<u32>;
// blocks of superblock s are superblock_blocks[superblock_block_ptr[s]:superblock_block_ptr[s + 1]], in the
// order of the superblock's columns
@group(0) @binding(3)
var<storage, read> superblock_block_ptr: array<u32>;
@group(0) @binding(4)
var<storage, read> superblock_blocks: array<u32>;
// weights of the work items, 4 vec4s (16 signals) per column of the superblock
@group(0) @binding(5)
var<storage, read> w: array<vec4<f32>>;
// [n_partial_rows, n_frames_padded]
@group(0) @binding(6)
var<storage, read_write> partial: array<vec4<f32>>;

override n_frames_padded: u32;

override wg_size: u32 = 128u;

@compute @workgroup_size(wg_size)
fn temporal_hals_partials(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n4 = n_frames_padded / 4u;
    let t4 = wid.y * wg_size + lid;
    if (t4 >= n4) {
        return;
    }
    let item = items[wid.x];

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var acc4 = vec4<f32>(0.0);
    var acc5 = vec4<f32>(0.0);
    var acc6 = vec4<f32>(0.0);
    var acc7 = vec4<f32>(0.0);
    var acc8 = vec4<f32>(0.0);
    var acc9 = vec4<f32>(0.0);
    var acc10 = vec4<f32>(0.0);
    var acc11 = vec4<f32>(0.0);
    var acc12 = vec4<f32>(0.0);
    var acc13 = vec4<f32>(0.0);
    var acc14 = vec4<f32>(0.0);
    var acc15 = vec4<f32>(0.0);

    var wi = item.w_start;
    for (var sb = superblock_block_ptr[item.superblock]; sb < superblock_block_ptr[item.superblock + 1u]; sb++) {
        let block = superblock_blocks[sb];
        for (var k = block_col_ptr[block]; k < block_col_ptr[block + 1u]; k++) {
            let v = temporal_compressed[k * n4 + t4];
            let w0 = w[wi];
            let w1 = w[wi + 1u];
            let w2 = w[wi + 2u];
            let w3 = w[wi + 3u];
            acc0 = fma(vec4<f32>(w0.x), v, acc0);
            acc1 = fma(vec4<f32>(w0.y), v, acc1);
            acc2 = fma(vec4<f32>(w0.z), v, acc2);
            acc3 = fma(vec4<f32>(w0.w), v, acc3);
            acc4 = fma(vec4<f32>(w1.x), v, acc4);
            acc5 = fma(vec4<f32>(w1.y), v, acc5);
            acc6 = fma(vec4<f32>(w1.z), v, acc6);
            acc7 = fma(vec4<f32>(w1.w), v, acc7);
            acc8 = fma(vec4<f32>(w2.x), v, acc8);
            acc9 = fma(vec4<f32>(w2.y), v, acc9);
            acc10 = fma(vec4<f32>(w2.z), v, acc10);
            acc11 = fma(vec4<f32>(w2.w), v, acc11);
            acc12 = fma(vec4<f32>(w3.x), v, acc12);
            acc13 = fma(vec4<f32>(w3.y), v, acc13);
            acc14 = fma(vec4<f32>(w3.z), v, acc14);
            acc15 = fma(vec4<f32>(w3.w), v, acc15);
            wi += 4u;
        }
    }

    let accs = array<vec4<f32>, 16>(
        acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7,
        acc8, acc9, acc10, acc11, acc12, acc13, acc14, acc15,
    );
    for (var s = 0u; s < 16u; s++) {
        if (s < item.n_rows) {
            partial[(item.first_row + s) * n4 + t4] = accs[s];
        }
    }
}
