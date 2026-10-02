// diff[k, i] = (V[k] . c_i - q0[k] . (q1 c_i)) / ||c_i||^2
// for each column k of U in each block that overlaps the support of signal i, the "diff" term of
// masknmf's spatial_update_hals, computed only where it is used (a sampled dense-dense product, SDDMM).
//
// One workgroup per work item: a superblock of 2 x 2 blocks and up to 32 of the signals that overlap its blocks.
// The rows of V of the blocks (their columns of U) and the rows of c of the signals are staged through workgroup
// memory in chunks of frames with coalesced vec4 loads, so each row of V is read once per work item instead of
// once per 16 signals of its block, and each row of c once per superblock instead of once per block. The next
// chunk is loaded into registers while the current chunk is used from workgroup memory (register double
// buffering), so that the loads are in flight during the compute.
//
// Each invocation accumulates a 4 x 4 register tile of (row of V, signal) dot products, the 8 vec4 reads from
// workgroup memory of each step feed 64 FMAs. The frames of each chunk are split among as many invocations per
// tile (lanes) as fit in the workgroup, so that all invocations compute, and the lanes' dot products are summed
// at the end. The frames are summed in three levels, over each chunk, over 16 chunks and over the blocks of 16
// chunks (blocked summation), which keeps the float32 rounding error of sums over tens of thousands of frames
// small. Tiles of rows of a block that a signal does not overlap are computed but not written.

// V, [rank, n_frames_padded]
@group(0) @binding(0)
var<storage, read> temporal_compressed: array<vec4<f32>>;
// c, [n_signals, n_frames_padded]
@group(0) @binding(1)
var<storage, read> temporal_demixed: array<vec4<f32>>;

// the rows of V of a work item are the columns of up to 4 blocks, block after block
struct Item {
    // first column of U of each block
    col_starts: vec4<u32>,
    // first row of each block in the rows of the work item, n_rows for unused blocks
    row_starts: vec4<u32>,
    n_rows: u32,
    n_signals: u32,
}

@group(0) @binding(2)
var<storage, read> items: array<Item>;
// [n_items, max_signals], signals of each work item
@group(0) @binding(3)
var<storage, read> item_signals: array<u32>;
// [n_items, max_signals, 4], start of the values in diff of each (signal, block) pair of each work item, one value
// per column of the block, no_pair where the signal does not overlap the block
@group(0) @binding(4)
var<storage, read> item_pair_offsets: array<u32>;
// factorized ring term q0, [rank, ring_rank_padded]
@group(0) @binding(5)
var<storage, read> ring_left: array<vec4<f32>>;
// c q1^T, [n_signals, ring_rank_padded]
@group(0) @binding(6)
var<storage, read> ring_c: array<vec4<f32>>;
@group(0) @binding(7)
var<storage, read> c_sq: array<f32>;
@group(0) @binding(8)
var<storage, read_write> diff: array<f32>;
// 0 if there is no ring term, a uniform so that a change of the ring rank does not recreate the pipeline
@group(0) @binding(9)
var<uniform> ring_rank_padded: u32;

override n_frames_padded: u32;
// max rows of V of a work item, 4 * max_block_cols
override max_rows: u32;
// frames per chunk, in vec4s, a power of two
override chunk4: u32 = 16u;

const wg_size: u32 = 256u;
const max_signals: u32 = 32u;
const no_pair: u32 = 0xffffffffu;

// vec4s of a chunk that each invocation stages from V and from c, at most 8 together, the fields of Staged. The
// rows of V and of c are staged by separate vec4s so that each load reads one buffer. Overrides must be declared
// before the workgroup arrays whose size depends on overrides, naga 27 panics otherwise
override n_stage_v: u32 = (max_rows * chunk4 + wg_size - 1u) / wg_size;
override n_stage_c: u32 = (max_signals * chunk4 + wg_size - 1u) / wg_size;
// rows of v_tile and c_tile, all the rows that the staged vec4s cover
override v_rows: u32 = n_stage_v * wg_size / chunk4;
override c_rows: u32 = n_stage_c * wg_size / chunk4;
// row stride of v_tile and c_tile in vec4s, one vec4 of padding per row against bank conflicts
override row_stride4: u32 = chunk4 + 1u;

// rows in quad-major order, see quad_major. After the last chunk v_tile holds the totals of up to half of the
// invocations, 4 vec4s each, while the lanes are summed
var<workgroup> v_tile: array<vec4<f32>, max(v_rows * row_stride4, 2u * wg_size)>;
var<workgroup> c_tile: array<vec4<f32>, c_rows * row_stride4>;

// the vec4s of a chunk that an invocation stages, named rather than an array so that they stay in registers
struct Staged {
    s0: vec4<f32>,
    s1: vec4<f32>,
    s2: vec4<f32>,
    s3: vec4<f32>,
    s4: vec4<f32>,
    s5: vec4<f32>,
    s6: vec4<f32>,
    s7: vec4<f32>,
}

// start of the row in V or c of each vec4 that an invocation stages, in vec4s, named like Staged
struct StagedRows {
    r0: u32,
    r1: u32,
    r2: u32,
    r3: u32,
    r4: u32,
    r5: u32,
    r6: u32,
    r7: u32,
}

// 4 x 4 register tile of (row of V, signal) dot products, a_rs of row r and signal s, one sum per component of
// the vec4s
struct RegisterTile {
    a00: vec4<f32>,
    a01: vec4<f32>,
    a02: vec4<f32>,
    a03: vec4<f32>,
    a10: vec4<f32>,
    a11: vec4<f32>,
    a12: vec4<f32>,
    a13: vec4<f32>,
    a20: vec4<f32>,
    a21: vec4<f32>,
    a22: vec4<f32>,
    a23: vec4<f32>,
    a30: vec4<f32>,
    a31: vec4<f32>,
    a32: vec4<f32>,
    a33: vec4<f32>,
}

// the tile plus the products of one step of rows v0, v1, v2, v3 and c0, c1, c2, c3
fn tile_fma(
    x: RegisterTile,
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
    c3: vec4<f32>,
) -> RegisterTile {
    return RegisterTile(
        fma(v0, c0, x.a00),
        fma(v0, c1, x.a01),
        fma(v0, c2, x.a02),
        fma(v0, c3, x.a03),
        fma(v1, c0, x.a10),
        fma(v1, c1, x.a11),
        fma(v1, c2, x.a12),
        fma(v1, c3, x.a13),
        fma(v2, c0, x.a20),
        fma(v2, c1, x.a21),
        fma(v2, c2, x.a22),
        fma(v2, c3, x.a23),
        fma(v3, c0, x.a30),
        fma(v3, c1, x.a31),
        fma(v3, c2, x.a32),
        fma(v3, c3, x.a33),
    );
}

// the dot products of one row of the tile, sums of the components
fn row_dots(a0: vec4<f32>, a1: vec4<f32>, a2: vec4<f32>, a3: vec4<f32>) -> vec4<f32> {
    let one = vec4<f32>(1.0);
    return vec4<f32>(dot(a0, one), dot(a1, one), dot(a2, one), dot(a3, one));
}

// position of a row in v_tile or c_tile: row 4 q + i is at i * n_rows / 4 + q. The invocations of a wave read
// rows i of consecutive quads at the same time, which are consecutive and with the odd row stride fall into
// different banks
fn quad_major(row: u32, n_rows: u32) -> u32 {
    return (row % 4u) * (n_rows / 4u) + row / 4u;
}

// block of row r of the work item's rows of V, rows beyond n_rows get the last block
fn row_block(item: Item, r: u32) -> u32 {
    return u32(r >= item.row_starts.y) + u32(r >= item.row_starts.z) + u32(r >= item.row_starts.w);
}

// column of U of row r of the work item's rows of V, 0 for rows beyond n_rows
fn row_column(item: Item, r: u32) -> u32 {
    let b = row_block(item, r);
    return select(0u, item.col_starts[b] + r - item.row_starts[b], r < item.n_rows);
}

// start of the row of vec4 k that invocation lid stages: vec4 lid + k * wg_size of the rows of V for k < n_stage_v,
// of the rows of c for the next n_stage_c, each row chunk4 vec4s. Rows beyond the work item's rows or signals are
// read from row 0, they only feed results that are not written
fn staged_row(k: u32, lid: u32, item: Item, signals: u32) -> u32 {
    let n4 = n_frames_padded / 4u;
    if (k < n_stage_v) {
        return row_column(item, (lid + k * wg_size) / chunk4) * n4;
    }
    if (k < n_stage_v + n_stage_c) {
        let s = (lid + (k - n_stage_v) * wg_size) / chunk4;
        if (s < item.n_signals) {
            return item_signals[signals + s] * n4;
        }
    }
    return 0u;
}

fn staged_rows(lid: u32, item: Item, signals: u32) -> StagedRows {
    return StagedRows(
        staged_row(0u, lid, item, signals),
        staged_row(1u, lid, item, signals),
        staged_row(2u, lid, item, signals),
        staged_row(3u, lid, item, signals),
        staged_row(4u, lid, item, signals),
        staged_row(5u, lid, item, signals),
        staged_row(6u, lid, item, signals),
        staged_row(7u, lid, item, signals),
    );
}

// vec4 k that an invocation stages, frame vec4 t of its row
fn load_staged(k: u32, row: u32, t: u32) -> vec4<f32> {
    if (k < n_stage_v) {
        return temporal_compressed[row + t];
    }
    if (k < n_stage_v + n_stage_c) {
        return temporal_demixed[row + t];
    }
    return vec4<f32>(0.0);
}

// the vec4s of the chunk that starts at frame vec4 t0 that invocation lid stages, all at frame vec4
// t0 + lid % chunk4 of their rows. Beyond the last frame they read the last frame, store_chunk zeroes them
fn load_chunk(rows: StagedRows, t0: u32, lid: u32) -> Staged {
    let t = min(t0 + lid % chunk4, n_frames_padded / 4u - 1u);
    return Staged(
        load_staged(0u, rows.r0, t),
        load_staged(1u, rows.r1, t),
        load_staged(2u, rows.r2, t),
        load_staged(3u, rows.r3, t),
        load_staged(4u, rows.r4, t),
        load_staged(5u, rows.r5, t),
        load_staged(6u, rows.r6, t),
        load_staged(7u, rows.r7, t),
    );
}

// writes vec4 k that invocation lid staged, see staged_row, to v_tile or c_tile
fn store_staged(k: u32, lid: u32, x: vec4<f32>) {
    if (k < n_stage_v) {
        let j = lid + k * wg_size;
        v_tile[quad_major(j / chunk4, v_rows) * row_stride4 + j % chunk4] = x;
    } else if (k < n_stage_v + n_stage_c) {
        let j = lid + (k - n_stage_v) * wg_size;
        c_tile[quad_major(j / chunk4, c_rows) * row_stride4 + j % chunk4] = x;
    }
}

// the chunk that starts at frame vec4 t0, zero beyond the last frame
fn store_chunk(staged: Staged, t0: u32, lid: u32) {
    let in_range = t0 + lid % chunk4 < n_frames_padded / 4u;
    store_staged(0u, lid, select(vec4<f32>(0.0), staged.s0, in_range));
    store_staged(1u, lid, select(vec4<f32>(0.0), staged.s1, in_range));
    store_staged(2u, lid, select(vec4<f32>(0.0), staged.s2, in_range));
    store_staged(3u, lid, select(vec4<f32>(0.0), staged.s3, in_range));
    store_staged(4u, lid, select(vec4<f32>(0.0), staged.s4, in_range));
    store_staged(5u, lid, select(vec4<f32>(0.0), staged.s5, in_range));
    store_staged(6u, lid, select(vec4<f32>(0.0), staged.s6, in_range));
    store_staged(7u, lid, select(vec4<f32>(0.0), staged.s7, in_range));
}

// writes the dot product of row r of the work item's rows of V and signal s of the work item, if the signal
// overlaps the row's block
fn write_diff(item: Item, signals: u32, r: u32, s: u32, value: f32) {
    if (r >= item.n_rows || s >= item.n_signals) {
        return;
    }
    let b = row_block(item, r);
    let offset = item_pair_offsets[(signals + s) * 4u + b];
    if (offset != no_pair) {
        diff[offset + r - item.row_starts[b]] = value / c_sq[item_signals[signals + s]];
    }
}

@compute @workgroup_size(wg_size)
fn spatial_hals_diff(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_index) lid: u32,
) {
    let item_index = wid.y * nwg.x + wid.x;
    if (item_index >= arrayLength(&items)) {
        return;
    }
    let item = items[item_index];
    let signals = item_index * max_signals;
    let n4 = n_frames_padded / 4u;

    // this invocation's register tile, rows 4 rq:4 rq + 4 of the work item's rows of V and signals 4 sq:4 sq + 4,
    // and its lane: the frames of each chunk and the ring rank are split among the n_lanes invocations of a tile
    let n_row_quads = (item.n_rows + 3u) / 4u;
    let n_tiles = n_row_quads * ((item.n_signals + 3u) / 4u);
    let n_lanes = min(wg_size / n_tiles, chunk4);
    let tile = lid % n_tiles;
    let lane = lid / n_tiles;
    let is_computing = lane < n_lanes;
    let rq = tile % n_row_quads;
    let sq = tile / n_row_quads;

    // row 4 rq + i of v_tile is at v_start + i * v_step, see quad_major, likewise for c_tile
    let v_start = rq * row_stride4;
    let v_step = v_rows / 4u * row_stride4;
    let c_start = sq * row_stride4;
    let c_step = c_rows / 4u * row_stride4;

    // the lane's sums over up to 16 chunks and its totals over the blocks of 16 chunks, sums_r[s] and totals_r[s]
    // of row 4 rq + r and signal 4 sq + s
    var sums0 = vec4<f32>(0.0);
    var sums1 = vec4<f32>(0.0);
    var sums2 = vec4<f32>(0.0);
    var sums3 = vec4<f32>(0.0);
    var totals0 = vec4<f32>(0.0);
    var totals1 = vec4<f32>(0.0);
    var totals2 = vec4<f32>(0.0);
    var totals3 = vec4<f32>(0.0);

    let rows = staged_rows(lid, item, signals);
    var staged = load_chunk(rows, 0u, lid);

    for (var t0 = 0u; t0 < n4; t0 += chunk4) {
        store_chunk(staged, t0, lid);
        workgroupBarrier();

        if (t0 + chunk4 < n4) {
            staged = load_chunk(rows, t0 + chunk4, lid);
        }

        if (is_computing) {
            var acc = RegisterTile();
            for (var t = lane; t < chunk4; t += n_lanes) {
                acc = tile_fma(
                    acc,
                    v_tile[v_start + t],
                    v_tile[v_start + v_step + t],
                    v_tile[v_start + 2u * v_step + t],
                    v_tile[v_start + 3u * v_step + t],
                    c_tile[c_start + t],
                    c_tile[c_start + c_step + t],
                    c_tile[c_start + 2u * c_step + t],
                    c_tile[c_start + 3u * c_step + t],
                );
            }
            sums0 += row_dots(acc.a00, acc.a01, acc.a02, acc.a03);
            sums1 += row_dots(acc.a10, acc.a11, acc.a12, acc.a13);
            sums2 += row_dots(acc.a20, acc.a21, acc.a22, acc.a23);
            sums3 += row_dots(acc.a30, acc.a31, acc.a32, acc.a33);
            if ((t0 / chunk4) % 16u == 15u) {
                totals0 += sums0;
                totals1 += sums1;
                totals2 += sums2;
                totals3 += sums3;
                sums0 = vec4<f32>(0.0);
                sums1 = vec4<f32>(0.0);
                sums2 = vec4<f32>(0.0);
                sums3 = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();
    }
    totals0 += sums0;
    totals1 += sums1;
    totals2 += sums2;
    totals3 += sums3;

    // subtract q0[k] . (q1 c_i): the same product with the rows of q0 for the rows of V and the rows of q1 c for
    // the rows of c, over the ring rank instead of the frames. Rows beyond the work item's rows read row 0 of q0,
    // unused signal slots hold signal 0
    let r0 = 4u * rq;
    let s0 = 4u * sq;
    if (is_computing) {
        let k0 = row_column(item, r0);
        let k1 = row_column(item, r0 + 1u);
        let k2 = row_column(item, r0 + 2u);
        let k3 = row_column(item, r0 + 3u);
        let i0 = item_signals[signals + s0];
        let i1 = item_signals[signals + s0 + 1u];
        let i2 = item_signals[signals + s0 + 2u];
        let i3 = item_signals[signals + s0 + 3u];
        let rrp4 = ring_rank_padded / 4u;
        var acc = RegisterTile();
        for (var rho = lane; rho < rrp4; rho += n_lanes) {
            acc = tile_fma(
                acc,
                -ring_left[k0 * rrp4 + rho],
                -ring_left[k1 * rrp4 + rho],
                -ring_left[k2 * rrp4 + rho],
                -ring_left[k3 * rrp4 + rho],
                ring_c[i0 * rrp4 + rho],
                ring_c[i1 * rrp4 + rho],
                ring_c[i2 * rrp4 + rho],
                ring_c[i3 * rrp4 + rho],
            );
        }
        totals0 += row_dots(acc.a00, acc.a01, acc.a02, acc.a03);
        totals1 += row_dots(acc.a10, acc.a11, acc.a12, acc.a13);
        totals2 += row_dots(acc.a20, acc.a21, acc.a22, acc.a23);
        totals3 += row_dots(acc.a30, acc.a31, acc.a32, acc.a33);
    }

    // sum the lanes of each tile through v_tile: of the n lanes left, lanes half:n add theirs to lanes 0:n - half
    for (var n = n_lanes; n > 1u; n = (n + 1u) / 2u) {
        let half = (n + 1u) / 2u;
        if (lane >= half && lane < n) {
            let slot = ((lane - half) * n_tiles + tile) * 4u;
            v_tile[slot] = totals0;
            v_tile[slot + 1u] = totals1;
            v_tile[slot + 2u] = totals2;
            v_tile[slot + 3u] = totals3;
        }
        workgroupBarrier();
        if (lane < n - half) {
            let slot = (lane * n_tiles + tile) * 4u;
            totals0 += v_tile[slot];
            totals1 += v_tile[slot + 1u];
            totals2 += v_tile[slot + 2u];
            totals3 += v_tile[slot + 3u];
        }
        workgroupBarrier();
    }

    if (lane != 0u) {
        return;
    }
    write_diff(item, signals, r0, s0, totals0.x);
    write_diff(item, signals, r0, s0 + 1u, totals0.y);
    write_diff(item, signals, r0, s0 + 2u, totals0.z);
    write_diff(item, signals, r0, s0 + 3u, totals0.w);
    write_diff(item, signals, r0 + 1u, s0, totals1.x);
    write_diff(item, signals, r0 + 1u, s0 + 1u, totals1.y);
    write_diff(item, signals, r0 + 1u, s0 + 2u, totals1.z);
    write_diff(item, signals, r0 + 1u, s0 + 3u, totals1.w);
    write_diff(item, signals, r0 + 2u, s0, totals2.x);
    write_diff(item, signals, r0 + 2u, s0 + 1u, totals2.y);
    write_diff(item, signals, r0 + 2u, s0 + 2u, totals2.z);
    write_diff(item, signals, r0 + 2u, s0 + 3u, totals2.w);
    write_diff(item, signals, r0 + 3u, s0, totals3.x);
    write_diff(item, signals, r0 + 3u, s0 + 1u, totals3.y);
    write_diff(item, signals, r0 + 3u, s0 + 2u, totals3.z);
    write_diff(item, signals, r0 + 3u, s0 + 3u, totals3.w);
}
