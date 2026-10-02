// Per cell of U (see compute_cell_tiles in _compression.py), a small matrix from the rows of p and q for the cell's
// n_cols columns of U, summed over the columns k < n_k of the rows:
//
//     out_c = alpha * P_c Q_c^T, or out_c += alpha * P_c Q_c^T with accumulate
//
// With symmetric, q is p: only the 4 x 4 blocks on and above the diagonal are computed and each is also written
// transposed. out_c is n_pad x n_pad row-major at gram_ptr[c], n_pad = n_cols rounded up to a multiple of 4, its rows
// and columns beyond n_cols are written as 0.
//
// One 4 x 4 block of out_c per invocation and wg_size blocks per workgroup, as MAGMA's vbatched SYRK computes one
// output tile per thread block and skips those below the diagonal: workgroup (c, s) computes blocks
// s wg_size:(s + 1) wg_size of cell c and ends at once if the cell has fewer. When a workgroup has fewer blocks than
// invocations, groups of invocations split the columns of each chunk and their partial blocks are summed at the end.
// Chunks of kc columns are staged through workgroup memory k-major, so that one vec4 read gives 4 rows at one column,
// and the next chunk is loaded into registers while the current one is multiplied. The sums are blocked: over the
// columns of a chunk, over 16 chunks, and in total.

@group(0) @binding(0)
var<storage, read> cell_col_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cell_cols: array<u32>;
// start of each cell's matrix in out, in floats, a multiple of 4
@group(0) @binding(2)
var<storage, read> gram_ptr: array<u32>;
// [rank, k_stride4]
@group(0) @binding(3)
var<storage, read> p: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec4<f32>>;

override max_cell_cols: u32;
// columns summed, a multiple of 4
override n_k: u32;
// row stride of p and q in vec4s
override k_stride4: u32;
override symmetric: bool;
override accumulate: bool;
override alpha: f32 = 1.0;

const wg_size: u32 = 256u;
// columns per chunk
const kc: u32 = 32u;
const kc4: u32 = kc / 4u;

// vec4s of 4 rows of the largest cell
override nb_max: u32 = (max_cell_cols + 3u) / 4u;

// the chunk of P_c at 0, of Q_c at kc * nb_max, entry k * nb_max + b holds rows 4b:4b + 4 at column k; at the end the
// partial blocks of the invocations
var<workgroup> lds: array<vec4<f32>, max(2u * kc * nb_max, 4u * wg_size)>;

// The vec4s of one chunk that an invocation loads: slot s is row s / kc4 and columns 4 (s % kc4):4 (s % kc4) + 4 of
// the chunk, slots lid, lid + wg_size and lid + 2 wg_size cover the rows of the largest cell (max_cell_cols <= 96,
// checked on the host).
struct Chunk {
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
}

// a 4 x 4 block, row i in ri
struct Block {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    r3: vec4<f32>,
}

fn block_fma(x: Block, a: vec4<f32>, b: vec4<f32>) -> Block {
    return Block(
        fma(vec4<f32>(a.x), b, x.r0),
        fma(vec4<f32>(a.y), b, x.r1),
        fma(vec4<f32>(a.z), b, x.r2),
        fma(vec4<f32>(a.w), b, x.r3),
    );
}

fn block_add(x: Block, y: Block) -> Block {
    return Block(x.r0 + y.r0, x.r1 + y.r1, x.r2 + y.r2, x.r3 + y.r3);
}

// loads are unconditional, with the row and column clamped; the values beyond them are zeroed when stored
fn load_slot(from_q: bool, s: u32, col_start: u32, n_cols: u32, k4: u32) -> vec4<f32> {
    let row = min(s / kc4, max(n_cols, 1u) - 1u);
    let index = cell_cols[col_start + row] * k_stride4 + min(k4 + s % kc4, n_k / 4u - 1u);
    if (from_q) {
        return q[index];
    }
    return p[index];
}

fn load_chunk(from_q: bool, lid: u32, col_start: u32, n_cols: u32, k4: u32) -> Chunk {
    return Chunk(
        load_slot(from_q, lid, col_start, n_cols, k4),
        load_slot(from_q, lid + wg_size, col_start, n_cols, k4),
        load_slot(from_q, lid + 2u * wg_size, col_start, n_cols, k4),
    );
}

// the 4 columns of a row as 4 entries of the k-major chunk at offset, rows n_cols:4 nb as 0
fn store_slot(offset: u32, s: u32, x: vec4<f32>, n_cols: u32, nb: u32, k_valid4: u32) {
    let row = s / kc4;
    let kv = s % kc4;
    if (row < 4u * nb) {
        let value = select(vec4<f32>(), x, row < n_cols && kv < k_valid4);
        let base = offset + 4u * kv * nb_max + row / 4u;
        lds[base][row % 4u] = value.x;
        lds[base + nb_max][row % 4u] = value.y;
        lds[base + 2u * nb_max][row % 4u] = value.z;
        lds[base + 3u * nb_max][row % 4u] = value.w;
    }
}

fn store_chunk(offset: u32, c: Chunk, lid: u32, n_cols: u32, nb: u32, k_valid4: u32) {
    store_slot(offset, lid, c.s0, n_cols, nb, k_valid4);
    store_slot(offset, lid + wg_size, c.s1, n_cols, nb, k_valid4);
    store_slot(offset, lid + 2u * wg_size, c.s2, n_cols, nb, k_valid4);
}

fn write_row(index: u32, x: vec4<f32>) {
    if (accumulate) {
        out[index] = fma(vec4<f32>(alpha), x, out[index]);
    } else {
        out[index] = alpha * x;
    }
}

@compute @workgroup_size(wg_size)
fn cell_gram(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let cell = wid.x;
    let col_start = cell_col_ptr[cell];
    let n_cols = cell_col_ptr[cell + 1u] - col_start;
    // vec4s of rows of P_c, the rows of out_c are nb vec4s
    let nb = (n_cols + 3u) / 4u;
    let n_blocks = select(nb * nb, nb * (nb + 1u) / 2u, symmetric);
    let first_block = wid.y * wg_size;
    if (first_block >= n_blocks) {
        return;
    }
    let set_blocks = min(n_blocks - first_block, wg_size);
    let n_groups = clamp(wg_size / set_blocks, 1u, kc);
    let group = lid / set_blocks;

    // block (bi, bj) of this invocation, row-major over the blocks on and above the diagonal when symmetric
    var bi = 0u;
    var bj = first_block + lid % set_blocks;
    if (symmetric) {
        while (bj >= nb - bi) {
            bj -= nb - bi;
            bi++;
        }
        bj += bi;
    } else {
        bi = bj / nb;
        bj = bj % nb;
    }
    let q_offset = select(kc * nb_max, 0u, symmetric);

    let n_k4 = n_k / 4u;
    let n_chunks = (n_k4 + kc4 - 1u) / kc4;
    var staged_p = load_chunk(false, lid, col_start, n_cols, 0u);
    var staged_q = Chunk();
    if (!symmetric) {
        staged_q = load_chunk(true, lid, col_start, n_cols, 0u);
    }

    var total = Block();
    var mid = Block();
    for (var chunk = 0u; chunk < n_chunks; chunk++) {
        let k4 = chunk * kc4;
        let k_valid4 = min(kc4, n_k4 - k4);
        store_chunk(0u, staged_p, lid, n_cols, nb, k_valid4);
        if (!symmetric) {
            store_chunk(q_offset, staged_q, lid, n_cols, nb, k_valid4);
        }
        workgroupBarrier();

        // the next chunk, the last one again after it
        let next4 = min(k4 + kc4, (n_chunks - 1u) * kc4);
        staged_p = load_chunk(false, lid, col_start, n_cols, next4);
        if (!symmetric) {
            staged_q = load_chunk(true, lid, col_start, n_cols, next4);
        }

        var sum = Block();
        if (group < n_groups) {
            for (var k = group; k < kc; k += n_groups) {
                sum = block_fma(sum, lds[k * nb_max + bi], lds[q_offset + k * nb_max + bj]);
            }
        }
        mid = block_add(mid, sum);
        if (chunk % 16u == 15u) {
            total = block_add(total, mid);
            mid = Block();
        }
        workgroupBarrier();
    }
    total = block_add(total, mid);

    // sum the partial blocks of the groups
    lds[4u * lid] = total.r0;
    lds[4u * lid + 1u] = total.r1;
    lds[4u * lid + 2u] = total.r2;
    lds[4u * lid + 3u] = total.r3;
    workgroupBarrier();
    if (lid >= set_blocks) {
        return;
    }
    var x = total;
    for (var g = 1u; g < n_groups; g++) {
        let o = 4u * (g * set_blocks + lid);
        x = block_add(x, Block(lds[o], lds[o + 1u], lds[o + 2u], lds[o + 3u]));
    }

    let base = gram_ptr[cell] / 4u;
    write_row(base + 4u * bi * nb + bj, x.r0);
    write_row(base + (4u * bi + 1u) * nb + bj, x.r1);
    write_row(base + (4u * bi + 2u) * nb + bj, x.r2);
    write_row(base + (4u * bi + 3u) * nb + bj, x.r3);
    if (symmetric && bi != bj) {
        write_row(base + 4u * bj * nb + bi, vec4<f32>(x.r0.x, x.r1.x, x.r2.x, x.r3.x));
        write_row(base + (4u * bj + 1u) * nb + bi, vec4<f32>(x.r0.y, x.r1.y, x.r2.y, x.r3.y));
        write_row(base + (4u * bj + 2u) * nb + bi, vec4<f32>(x.r0.z, x.r1.z, x.r2.z, x.r3.z));
        write_row(base + (4u * bj + 3u) * nb + bi, vec4<f32>(x.r0.w, x.r1.w, x.r2.w, x.r3.w));
    }
}
