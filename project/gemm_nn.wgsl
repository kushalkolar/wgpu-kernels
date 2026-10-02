// out = A @ B, A is [M, K] and B is [K, N], both row-major
//
// For subgroups of 64 (RDNA, RADV runs compute shaders in wave64): an FMA of wave64 runs at about half rate when all
// three operands are in vector registers, and at 80-90% of the peak when one is a scalar register (measured on the
// Radeon 780M). Each subgroup computes 8 rows of out, so the values of A of a step are the same for the whole
// subgroup: they are read with scalar loads and are the scalar operand of the FMAs. Each invocation computes 8 columns,
// the vec4s lane and lane + 64 of the workgroup's columns, from chunks of B staged through workgroup memory. The next
// chunk of B is loaded into registers while the current chunk is used (register double buffering). The values of A of
// a chunk, 8 rows x BK, are loaded as one block before the chunk is used: loaded per step, each wait for the scalar
// loads, which complete out of order, also waited for the loads from workgroup memory (RADV).
//
// HALS uses it for the ring term instead of gemm_nt when the subgroup size is 64, see get_subgroup_size in _hals.py

@group(0) @binding(0)
var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2)
var<storage, read_write> out: array<vec4<f32>>;

struct Params {
    M: u32,
    // multiple of 4
    N: u32,
    // multiple of 4
    K: u32,
}

@group(0) @binding(3)
var<uniform> params: Params;

// subgroup size the kernel is written for
const S: u32 = 64u;
const wg_size: u32 = 256u;
// rows of out per subgroup and per workgroup
const TM: u32 = 8u;
const BM: u32 = TM * wg_size / S;
// vec4s of out per row of a workgroup, 2 per invocation of a subgroup
const BN4: u32 = 2u * S;
const BK: u32 = 8u;

// chunk of B, BK rows of BN4 vec4s, adjacent invocations read adjacent vec4s so without bank conflicts
var<workgroup> b_tile: array<array<vec4<f32>, BN4>, BK>;

// the vec4s of a chunk of B that an invocation stages, BK * BN4 / wg_size, named rather than an array so that they
// stay in registers
struct Staged {
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
    s3: vec4<f32>,
}

// rows m0:m0 + 8 and k0:k0 + 8 of A, uniform in the subgroup: row r is in ar0 (k0:k0 + 4) and ar1 (k0 + 4:k0 + 8)
struct ABlock {
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

// 8 x 8 register tile, a_rj is row r and col group j (the vec4s lane and lane + S), as in gemm_nt.wgsl
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

// vec4 pos of the chunk from row k0: row pos / BN4, vec4 pos % BN4 from vec4 n4 of the row. Rows beyond K read row
// K - 1 and vec4s beyond N the last one of the row, the rows beyond K are zeroed when stored
fn load_b(k0: u32, n4: u32, pos: u32, K: u32, N4: u32) -> vec4<f32> {
    return b[min(k0 + pos / BN4, K - 1u) * N4 + min(n4 + pos % BN4, N4 - 1u)];
}

fn load_chunk(k0: u32, n4: u32, lid: u32, K: u32, N4: u32) -> Staged {
    return Staged(
        load_b(k0, n4, lid, K, N4),
        load_b(k0, n4, lid + wg_size, K, N4),
        load_b(k0, n4, lid + 2u * wg_size, K, N4),
        load_b(k0, n4, lid + 3u * wg_size, K, N4),
    );
}

fn store_b(x: vec4<f32>, k0: u32, pos: u32, K: u32) {
    b_tile[pos / BN4][pos % BN4] = select(vec4<f32>(0.0), x, k0 + pos / BN4 < K);
}

fn store_chunk(staged: Staged, k0: u32, lid: u32, K: u32) {
    store_b(staged.s0, k0, lid, K);
    store_b(staged.s1, k0, lid + wg_size, K);
    store_b(staged.s2, k0, lid + 2u * wg_size, K);
    store_b(staged.s3, k0, lid + 3u * wg_size, K);
}

// rows beyond M read row M - 1, and k beyond K the last vec4 of the row, which meets rows of b_tile that are zero
fn load_a(m: u32, k4: u32, M: u32, K4: u32) -> vec4<f32> {
    return a[min(m, M - 1u) * K4 + min(k4, K4 - 1u)];
}

fn load_block(m0: u32, k0: u32, M: u32, K4: u32) -> ABlock {
    let k4 = k0 / 4u;
    return ABlock(
        load_a(m0, k4, M, K4),
        load_a(m0, k4 + 1u, M, K4),
        load_a(m0 + 1u, k4, M, K4),
        load_a(m0 + 1u, k4 + 1u, M, K4),
        load_a(m0 + 2u, k4, M, K4),
        load_a(m0 + 2u, k4 + 1u, M, K4),
        load_a(m0 + 3u, k4, M, K4),
        load_a(m0 + 3u, k4 + 1u, M, K4),
        load_a(m0 + 4u, k4, M, K4),
        load_a(m0 + 4u, k4 + 1u, M, K4),
        load_a(m0 + 5u, k4, M, K4),
        load_a(m0 + 5u, k4 + 1u, M, K4),
        load_a(m0 + 6u, k4, M, K4),
        load_a(m0 + 6u, k4 + 1u, M, K4),
        load_a(m0 + 7u, k4, M, K4),
        load_a(m0 + 7u, k4 + 1u, M, K4),
    );
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

// the tile plus the products of step kk of the chunk: component kk % 4 of the vec4s kk / 4 of the rows of the block
fn step_fma(x: RegisterTile, block: ABlock, kk: u32, lane: u32) -> RegisterTile {
    let h = kk >= 4u;
    let c = kk % 4u;
    let a0 = vec4<f32>(
        select(block.a00, block.a01, h)[c],
        select(block.a10, block.a11, h)[c],
        select(block.a20, block.a21, h)[c],
        select(block.a30, block.a31, h)[c],
    );
    let a1 = vec4<f32>(
        select(block.a40, block.a41, h)[c],
        select(block.a50, block.a51, h)[c],
        select(block.a60, block.a61, h)[c],
        select(block.a70, block.a71, h)[c],
    );
    return tile_fma(x, a0, a1, b_tile[kk][lane], b_tile[kk][lane + S]);
}

// the tile plus the products of the chunk, the steps written out so that kk is a constant in each, see gemm_nt.wgsl
fn chunk_fma(x: RegisterTile, block: ABlock, lane: u32) -> RegisterTile {
    var y = step_fma(x, block, 0u, lane);
    y = step_fma(y, block, 1u, lane);
    y = step_fma(y, block, 2u, lane);
    y = step_fma(y, block, 3u, lane);
    y = step_fma(y, block, 4u, lane);
    y = step_fma(y, block, 5u, lane);
    y = step_fma(y, block, 6u, lane);
    return step_fma(y, block, 7u, lane);
}

// writes the vec4s lane from row0 and lane + S from row1 of row m, from vec4 n4 of the row
fn write_row(m: u32, n4: u32, lane: u32, M: u32, N4: u32, row0: vec4<f32>, row1: vec4<f32>) {
    if (m >= M) {
        return;
    }
    if (n4 + lane < N4) {
        out[m * N4 + n4 + lane] = row0;
    }
    if (n4 + lane + S < N4) {
        out[m * N4 + n4 + lane + S] = row1;
    }
}

@compute @workgroup_size(wg_size)
fn gemm_nn(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_index) lid: u32,
    @builtin(subgroup_id) sid: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    // the uniforms are read once here: read where they are used, they were read again, and waited for, in each chunk
    let M = params.M;
    let N4 = params.N / 4u;
    let K = params.K;
    // workgroup x indexes the tiles along M, so that consecutive workgroups use the same chunks of B
    let m0 = wid.x * BM + sid * TM;
    let n4 = wid.y * BN4;

    var acc = RegisterTile();
    var staged = load_chunk(0u, n4, lid, K, N4);

    for (var k0 = 0u; k0 < K; k0 += BK) {
        let block = load_block(m0, k0, M, K / 4u);
        store_chunk(staged, k0, lid, K);
        workgroupBarrier();

        // prefetch the next chunk while this one is used, after the last chunk this reads its last row again
        staged = load_chunk(k0 + BK, n4, lid, K, N4);

        acc = chunk_fma(acc, block, lane);
        workgroupBarrier();
    }

    write_row(m0, n4, lane, M, N4, acc.a00, acc.a01);
    write_row(m0 + 1u, n4, lane, M, N4, acc.a10, acc.a11);
    write_row(m0 + 2u, n4, lane, M, N4, acc.a20, acc.a21);
    write_row(m0 + 3u, n4, lane, M, N4, acc.a30, acc.a31);
    write_row(m0 + 4u, n4, lane, M, N4, acc.a40, acc.a41);
    write_row(m0 + 5u, n4, lane, M, N4, acc.a50, acc.a51);
    write_row(m0 + 6u, n4, lane, M, N4, acc.a60, acc.a61);
    write_row(m0 + 7u, n4, lane, M, N4, acc.a70, acc.a71);
}
