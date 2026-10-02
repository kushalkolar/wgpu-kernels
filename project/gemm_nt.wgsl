// out = A @ B^T, A is [M, K] and B is [N, K], both row-major so the reduction dimension is contiguous
//
// Hierarchical tiling ("Efficient GEMM in CUDA", CUTLASS docs): each workgroup computes a BM x BN tile
// of out, staging BK wide chunks of A and B through workgroup memory. Each invocation computes a TM x TN
// register tile as a sum of outer products, 8 x 8 so each pair of vec4 reads from workgroup memory feeds
// 16 FMAs. The next chunk is loaded into registers while the current chunk is used from workgroup memory
// (register double buffering, WGSL has no async copies). The loads are unconditional, with the rows and k
// clamped into the matrices, and the values out of range are zeroed when a chunk is stored to workgroup
// memory, so that the loads of a chunk are in flight together: behind bounds checks each load waited for
// its data before the next one (RADV).
//
// Split-K: workgroup z computes the partial product over its range of K and writes it to out[z].
// gemm_nt_reduce sums the partial products in a fixed order, so the result is deterministic.
// Stream-K style in-kernel fixup is not used since WebGPU has no forward progress guarantee between
// workgroups.

@group(0) @binding(0)
var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> b: array<vec4<f32>>;
// [n_splits, M, ldc] partial products, or the result when n_splits is 1
@group(0) @binding(2)
var<storage, read_write> out: array<vec4<f32>>;
// [M, ldc] result of gemm_nt_reduce
@group(0) @binding(3)
var<storage, read_write> result: array<vec4<f32>>;

// uniforms instead of override constants, so that one pipeline serves every shape
struct Params {
    M: u32,
    // multiple of 4
    N: u32,
    // multiple of 4
    K: u32,
    // row strides in floats, multiples of 4
    lda: u32,
    ldb: u32,
    ldc: u32,
    // multiple of BK
    k_per_split: u32,
    n_splits: u32,
    // 1 if workgroup x indexes the tiles along M, so that consecutive workgroups share a tile of B, otherwise
    // along N so that they share a tile of A
    m_fastest: u32,
}

@group(0) @binding(4)
var<uniform> params: Params;

const BM: u32 = 128u;
const BN: u32 = 128u;
const BK: u32 = 32u;
// 256 invocations as a 16 x 16 grid of 8 x 8 register tiles
const TM: u32 = 8u;
const TN: u32 = 8u;
const wg_size: u32 = 256u;

// chunk_fma writes out the steps of a chunk
const_assert BK == 32u;

// transposed chunks, BK rows of BM / 4 + 1 vec4s, one vec4 of padding per row against bank conflicts. Two
// dimensional so that naga clamps the row and the column index separately, see chunk_fma
const a_row: u32 = BM / 4u + 1u;
const b_row: u32 = BN / 4u + 1u;
var<workgroup> a_tile: array<array<vec4<f32>, a_row>, BK>;
var<workgroup> b_tile: array<array<vec4<f32>, b_row>, BK>;

// each invocation loads the vec4 at k offset 4 * (lid % 8) of 4 consecutive rows, 4 * (lid / 8):
// adjacent invocations read adjacent vec4s of a row, and the 4 x 4 block is stored transposed as vec4s
struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    r3: vec4<f32>,
}

// 8 x 8 register tile, a_rj is row r (rows 4:8 are the second group of 4 rows) and col group j
struct RegisterTile {
    a00: vec4<f32>,
    a01: vec4<f32>,
    a10: vec4<f32>,
    a11: vec4<f32>,
    a20: vec4<f32>,
    a21: vec4<f32>,
    a30: vec4<f32>,
    a31: vec4<f32>,
    a40: vec4<f32>,
    a41: vec4<f32>,
    a50: vec4<f32>,
    a51: vec4<f32>,
    a60: vec4<f32>,
    a61: vec4<f32>,
    a70: vec4<f32>,
    a71: vec4<f32>,
}

// rows beyond the matrix read its last row and k beyond the split the last vec4 of the split, zeroed when stored
fn load_row_a(m: u32, kk: u32, k_end: u32) -> vec4<f32> {
    return a[(min(m, params.M - 1u) * params.lda + min(kk, k_end - 4u)) / 4u];
}

fn load_row_b(n: u32, kk: u32, k_end: u32) -> vec4<f32> {
    return b[(min(n, params.N - 1u) * params.ldb + min(kk, k_end - 4u)) / 4u];
}

fn load_a(m0: u32, k: u32, k_end: u32, lid: u32) -> Rows {
    let m = m0 + 4u * (lid / 8u);
    let kk = k + 4u * (lid % 8u);
    return Rows(
        load_row_a(m, kk, k_end),
        load_row_a(m + 1u, kk, k_end),
        load_row_a(m + 2u, kk, k_end),
        load_row_a(m + 3u, kk, k_end),
    );
}

fn load_b(n0: u32, k: u32, k_end: u32, lid: u32) -> Rows {
    let n = n0 + 4u * (lid / 8u);
    let kk = k + 4u * (lid % 8u);
    return Rows(
        load_row_b(n, kk, k_end),
        load_row_b(n + 1u, kk, k_end),
        load_row_b(n + 2u, kk, k_end),
        load_row_b(n + 3u, kk, k_end),
    );
}

// whether the 4 rows that invocation lid loads from rows0 on are within n_rows
fn valid_rows(rows0: u32, n_rows: u32, lid: u32) -> vec4<bool> {
    return vec4<u32>(rows0 + 4u * (lid / 8u)) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n_rows);
}

// the rows, zero where they are not valid
fn masked(x: Rows, valid: vec4<bool>) -> Rows {
    let zero = vec4<f32>(0.0);
    return Rows(
        select(zero, x.r0, valid.x),
        select(zero, x.r1, valid.y),
        select(zero, x.r2, valid.z),
        select(zero, x.r3, valid.w),
    );
}

fn store_chunk(ra: Rows, rb: Rows, lid: u32) {
    let row4 = lid / 8u;
    let k4 = 4u * (lid % 8u);
    a_tile[k4][row4] = vec4<f32>(ra.r0.x, ra.r1.x, ra.r2.x, ra.r3.x);
    a_tile[k4 + 1u][row4] = vec4<f32>(ra.r0.y, ra.r1.y, ra.r2.y, ra.r3.y);
    a_tile[k4 + 2u][row4] = vec4<f32>(ra.r0.z, ra.r1.z, ra.r2.z, ra.r3.z);
    a_tile[k4 + 3u][row4] = vec4<f32>(ra.r0.w, ra.r1.w, ra.r2.w, ra.r3.w);
    b_tile[k4][row4] = vec4<f32>(rb.r0.x, rb.r1.x, rb.r2.x, rb.r3.x);
    b_tile[k4 + 1u][row4] = vec4<f32>(rb.r0.y, rb.r1.y, rb.r2.y, rb.r3.y);
    b_tile[k4 + 2u][row4] = vec4<f32>(rb.r0.z, rb.r1.z, rb.r2.z, rb.r3.z);
    b_tile[k4 + 3u][row4] = vec4<f32>(rb.r0.w, rb.r1.w, rb.r2.w, rb.r3.w);
}

// the tile plus the outer product of a0, a1 (rows 0:4 and 4:8) and b0, b1 (col groups 0 and 1)
fn tile_fma(x: RegisterTile, a0: vec4<f32>, a1: vec4<f32>, b0: vec4<f32>, b1: vec4<f32>) -> RegisterTile {
    return RegisterTile(
        fma(vec4<f32>(a0.x), b0, x.a00),
        fma(vec4<f32>(a0.x), b1, x.a01),
        fma(vec4<f32>(a0.y), b0, x.a10),
        fma(vec4<f32>(a0.y), b1, x.a11),
        fma(vec4<f32>(a0.z), b0, x.a20),
        fma(vec4<f32>(a0.z), b1, x.a21),
        fma(vec4<f32>(a0.w), b0, x.a30),
        fma(vec4<f32>(a0.w), b1, x.a31),
        fma(vec4<f32>(a1.x), b0, x.a40),
        fma(vec4<f32>(a1.x), b1, x.a41),
        fma(vec4<f32>(a1.y), b0, x.a50),
        fma(vec4<f32>(a1.y), b1, x.a51),
        fma(vec4<f32>(a1.z), b0, x.a60),
        fma(vec4<f32>(a1.z), b1, x.a61),
        fma(vec4<f32>(a1.w), b0, x.a70),
        fma(vec4<f32>(a1.w), b1, x.a71),
    );
}

// the tile plus the products of step kk of the chunk in workgroup memory
fn step_fma(x: RegisterTile, kk: u32, tx: u32, ty: u32) -> RegisterTile {
    return tile_fma(x, a_tile[kk][ty], a_tile[kk][ty + 16u], b_tile[kk][tx], b_tile[kk][tx + 16u]);
}

// steps k0:k0 + 8
fn steps_fma(x: RegisterTile, k0: u32, tx: u32, ty: u32) -> RegisterTile {
    var y = step_fma(x, k0, tx, ty);
    y = step_fma(y, k0 + 1u, tx, ty);
    y = step_fma(y, k0 + 2u, tx, ty);
    y = step_fma(y, k0 + 3u, tx, ty);
    y = step_fma(y, k0 + 4u, tx, ty);
    y = step_fma(y, k0 + 5u, tx, ty);
    y = step_fma(y, k0 + 6u, tx, ty);
    return step_fma(y, k0 + 7u, tx, ty);
}

// the tile plus the products of the chunk in workgroup memory. The steps are written out so that kk is a
// constant in each: the column indices, and naga's clamps of them, are then the same in every step, so each read
// from workgroup memory can be a fixed offset from one of 4 addresses, with no loop control between the FMAs. As
// a loop, which the compiler did not unroll, each step also ran 12 VALU instructions for the addresses and 9
// scalar instructions and 3 branches of loop control (RADV)
fn chunk_fma(x: RegisterTile, tx: u32, ty: u32) -> RegisterTile {
    var y = steps_fma(x, 0u, tx, ty);
    y = steps_fma(y, 8u, tx, ty);
    y = steps_fma(y, 16u, tx, ty);
    return steps_fma(y, 24u, tx, ty);
}

// writes cols n:n + 4 from row0 and n + 64:n + 68 from row1 of row m, N is a multiple of 4 so each vec4 is either
// within N or beyond it
fn write_row(m: u32, n: u32, split: u32, row0: vec4<f32>, row1: vec4<f32>) {
    if (m >= params.M) {
        return;
    }
    let base = (split * params.M + m) * params.ldc;
    if (n < params.N) {
        out[(base + n) / 4u] = row0;
    }
    if (n + 64u < params.N) {
        out[(base + n + 64u) / 4u] = row1;
    }
}

@compute @workgroup_size(wg_size)
fn gemm_nt(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let m0 = select(wid.y, wid.x, params.m_fastest == 1u) * BM;
    let n0 = select(wid.x, wid.y, params.m_fastest == 1u) * BN;
    let k_start = wid.z * params.k_per_split;
    let k_end = min(params.K, k_start + params.k_per_split);

    // register tile: rows 4 * ty:4 * ty + 4 and 64 + 4 * ty:64 + 4 * ty + 4, cols likewise with tx, so that
    // adjacent invocations read adjacent vec4s of workgroup memory without bank conflicts
    let tx = lid % 16u;
    let ty = lid / 16u;
    var acc = RegisterTile();

    let a_valid = valid_rows(m0, params.M, lid);
    let b_valid = valid_rows(n0, params.N, lid);

    var ra = load_a(m0, k_start, k_end, lid);
    var rb = load_b(n0, k_start, k_end, lid);

    for (var k = k_start; k < k_end; k += BK) {
        // zero the rows beyond M and N and the vec4s beyond the end of the split
        let k_valid = vec4<bool>(k + 4u * (lid % 8u) < k_end);
        store_chunk(masked(ra, a_valid & k_valid), masked(rb, b_valid & k_valid), lid);
        workgroupBarrier();

        // prefetch the next chunk while this one is used, after the last chunk this reads its last vec4s again
        ra = load_a(m0, k + BK, k_end, lid);
        rb = load_b(n0, k + BK, k_end, lid);

        acc = chunk_fma(acc, tx, ty);
        workgroupBarrier();
    }

    write_row(m0 + 4u * ty + 0u, n0 + 4u * tx, wid.z, acc.a00, acc.a01);
    write_row(m0 + 4u * ty + 1u, n0 + 4u * tx, wid.z, acc.a10, acc.a11);
    write_row(m0 + 4u * ty + 2u, n0 + 4u * tx, wid.z, acc.a20, acc.a21);
    write_row(m0 + 4u * ty + 3u, n0 + 4u * tx, wid.z, acc.a30, acc.a31);
    write_row(m0 + 64u + 4u * ty + 0u, n0 + 4u * tx, wid.z, acc.a40, acc.a41);
    write_row(m0 + 64u + 4u * ty + 1u, n0 + 4u * tx, wid.z, acc.a50, acc.a51);
    write_row(m0 + 64u + 4u * ty + 2u, n0 + 4u * tx, wid.z, acc.a60, acc.a61);
    write_row(m0 + 64u + 4u * ty + 3u, n0 + 4u * tx, wid.z, acc.a70, acc.a71);
}

// one vec4 of the result per invocation
@compute @workgroup_size(wg_size)
fn gemm_nt_reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let index = gid.y * nwg.x * wg_size + gid.x;
    let n4 = params.N / 4u;
    if (index >= params.M * n4) {
        return;
    }
    let m = index / n4;
    let n = index % n4;
    let ldc4 = params.ldc / 4u;

    var sum = vec4<f32>(0.0);
    for (var s = 0u; s < params.n_splits; s++) {
        sum += out[(s * params.M + m) * ldc4 + n];
    }
    result[m * ldc4 + n] = sum;
}
