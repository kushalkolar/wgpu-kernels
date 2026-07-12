// gemm_ABT_partial.wgsl
// Split-K variant of gemm_ABT.wgsl. Divides the reduction axis n into
// k_chunks contiguous chunks and dispatches one workgroup per
// (row_strip, chunk) pair, so the total workgroup count is
// (m/rows_per_wg) × k_chunks — enough to fill the GPU even when m is
// small. Each workgroup accumulates a partial (rows_per_wg × r) tile
// for its chunk of n; a second pass (reduce_ABT) sums along k_chunks.
//
// C = A · B^T, computed as
//   partials[chunk, i, k] = sum_{j in chunk's n-range} A[i, j] · B[k, j]
//
// A ∈ R^(m × n) row-major, B ∈ R^(r × n) row-major (small r).
// partials ∈ R^(k_chunks × m × r), laid out as
//   partials[chunk * m * r + i * r + k].

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

override r: u32;
override wg_size: u32;
override rows_per_wg: u32;
override j_tile: u32;
override m: u32;
override n: u32;
override k_chunks: u32;
override chunk_size: u32;  // ceil(n / k_chunks) — last chunk may extend past n

var<workgroup> A_tile: array<f32, rows_per_wg * j_tile>;
var<workgroup> B_tile: array<f32, r * j_tile>;
var<workgroup> C_local: array<f32, rows_per_wg * r>;

@compute @workgroup_size(wg_size)
fn gemm_ABT_partial(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let i0 = wgid.x * rows_per_wg;
    let chunk = wgid.y;
    let j_start = chunk * chunk_size;
    let j_end = min(j_start + chunk_size, n);

    var oi: u32 = lid.x;
    while (oi < rows_per_wg * r) {
        C_local[oi] = 0.0;
        oi = oi + wg_size;
    }
    workgroupBarrier();

    var j0: u32 = j_start;
    while (j0 < j_end) {
        // Load B_tile: (r × j_tile) with zero-padding past j_end.
        var bi: u32 = lid.x;
        while (bi < r * j_tile) {
            let k = bi / j_tile;
            let jl = bi % j_tile;
            let j = j0 + jl;
            var v: f32 = 0.0;
            if (j < j_end) {
                v = B[k * n + j];
            }
            B_tile[k * j_tile + jl] = v;
            bi = bi + wg_size;
        }

        // Load A_tile: (rows_per_wg × j_tile) with zero-padding past m/j_end.
        var ai: u32 = lid.x;
        while (ai < rows_per_wg * j_tile) {
            let ir = ai / j_tile;
            let jl = ai % j_tile;
            let i = i0 + ir;
            let j = j0 + jl;
            var v: f32 = 0.0;
            if (i < m && j < j_end) {
                v = A[i * n + j];
            }
            A_tile[ir * j_tile + jl] = v;
            ai = ai + wg_size;
        }
        workgroupBarrier();

        // Accumulate. Padded loads make the inner j loop branch-free.
        var pi: u32 = lid.x;
        while (pi < rows_per_wg * r) {
            let ir = pi / r;
            let k = pi % r;
            var s: f32 = 0.0;
            for (var jl: u32 = 0u; jl < j_tile; jl = jl + 1u) {
                s = fma(A_tile[ir * j_tile + jl], B_tile[k * j_tile + jl], s);
            }
            C_local[pi] = C_local[pi] + s;
            pi = pi + wg_size;
        }
        workgroupBarrier();

        j0 = j0 + j_tile;
    }

    // Write this chunk's contribution to partials.
    var wi: u32 = lid.x;
    while (wi < rows_per_wg * r) {
        let ir = wi / r;
        let k = wi % r;
        let i = i0 + ir;
        if (i < m) {
            partials[chunk * m * r + i * r + k] = C_local[wi];
        }
        wi = wi + wg_size;
    }
}
