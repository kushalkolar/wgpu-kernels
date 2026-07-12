// gemm_ABT.wgsl
// Computes C = A · B^T where:
//   A ∈ R^(m × n) row-major
//   B ∈ R^(r × n) row-major   (small r, typical rank ≤ 32)
//   C ∈ R^(m × r) row-major
//
// Serves the HALS precomputes R = X H^T (A=X, B=H) and S = H H^T
// (A=H, B=H, m=r).
//
// Strategy: tiled reduction. One workgroup produces a (rows_per_wg × r)
// output strip of C. The reduction axis n is walked in chunks of j_tile.
// Each chunk: cooperatively load an (rows_per_wg × j_tile) slab of A and an
// (r × j_tile) slab of B into workgroup memory, then split the
// (rows_per_wg × r) output cells across threads and accumulate.
//
// Loads pad with zero past m or n so the inner accumulation loop always
// runs a fixed j_tile iterations and has no tail-divergence.
//
// Access patterns:
//   A load  — adjacent threads read adjacent A[i, j] columns → coalesced.
//   B load  — adjacent threads read adjacent B[k, j] columns → coalesced.
//   C write — adjacent threads (same ir, adjacent k) write adjacent
//             C[i, k]  → coalesced.

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

override r: u32;             // rank (small)
override wg_size: u32;
override rows_per_wg: u32;   // output rows per workgroup
override j_tile: u32;        // reduction chunk length
override m: u32;             // number of A rows / C rows
override n: u32;             // reduction axis

var<workgroup> A_tile: array<f32, rows_per_wg * j_tile>;
var<workgroup> B_tile: array<f32, r * j_tile>;
var<workgroup> C_local: array<f32, rows_per_wg * r>;

@compute @workgroup_size(wg_size)
fn gemm_ABT(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let i0 = wgid.x * rows_per_wg;

    var oi: u32 = lid.x;
    while (oi < rows_per_wg * r) {
        C_local[oi] = 0.0;
        oi = oi + wg_size;
    }
    workgroupBarrier();

    var j0: u32 = 0u;
    while (j0 < n) {
        // Load B_tile: (r × j_tile) block from B, padding past n with 0.
        var bi: u32 = lid.x;
        while (bi < r * j_tile) {
            let k = bi / j_tile;
            let jl = bi % j_tile;
            let j = j0 + jl;
            var v: f32 = 0.0;
            if (j < n) {
                v = B[k * n + j];
            }
            B_tile[k * j_tile + jl] = v;
            bi = bi + wg_size;
        }

        // Load A_tile: (rows_per_wg × j_tile) block from A, padding past m/n.
        var ai: u32 = lid.x;
        while (ai < rows_per_wg * j_tile) {
            let ir = ai / j_tile;
            let jl = ai % j_tile;
            let i = i0 + ir;
            let j = j0 + jl;
            var v: f32 = 0.0;
            if (i < m && j < n) {
                v = A[i * n + j];
            }
            A_tile[ir * j_tile + jl] = v;
            ai = ai + wg_size;
        }
        workgroupBarrier();

        // Accumulate. Threads split the (rows_per_wg × r) output cells.
        // The j_tile inner loop uses padded zeros for out-of-range jl so no
        // masking needed here.
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

    var wi: u32 = lid.x;
    while (wi < rows_per_wg * r) {
        let ir = wi / r;
        let k = wi % r;
        let i = i0 + ir;
        if (i < m) {
            C[i * r + k] = C_local[wi];
        }
        wi = wi + wg_size;
    }
}
