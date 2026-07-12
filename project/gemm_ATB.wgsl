// gemm_ATB.wgsl
// Computes C = A^T B where:
//   A ∈ R^(m × r)  row-major   (small r, typical rank ≤ 32)
//   B ∈ R^(m × n)  row-major
//   C ∈ R^(r × n)  row-major
//
// Serves the HALS precomputes P = W^T X (with A=W, B=X, r=rank, n=data cols)
// and Q = W^T W (with A=W, B=W, n=r).
//
// Strategy: outer-product accumulation. One workgroup owns wg_size output
// columns; each thread owns one column and holds r accumulators in
// workgroup shared memory (function-scope arrays cannot be sized by override
// constants in WGSL). Iterate through m in chunks of m_tile rows: each chunk,
// cooperatively load A[m_chunk, :] into shared, each thread reads its own
// B[i, j] and multiply-adds into all r accumulators.
//
// Access patterns:
//   A load  — contiguous block of A row-major, coalesced by threads.
//   B read  — adjacent threads read adjacent B columns for the same row.
//   C write — adjacent threads write adjacent C columns for the same row.
//   A_tile  — broadcast reads within a warp (all threads read same address);
//             cheap on modern GPUs, no bank conflicts.

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

override r: u32;         // rank (small, typical ≤ 32)
override wg_size: u32;   // threads per workgroup; each thread owns one output column
override m_tile: u32;    // rows of A/B loaded per iteration
override m: u32;         // reduction axis length
override n: u32;         // output column count

var<workgroup> A_tile: array<f32, m_tile * r>;
var<workgroup> C_local: array<f32, r * wg_size>;

@compute @workgroup_size(wg_size)
fn gemm_ATB(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let j = wgid.x * wg_size + lid.x;
    let j_ok = j < n;

    for (var k: u32 = 0u; k < r; k = k + 1u) {
        C_local[k * wg_size + lid.x] = 0.0;
    }

    var m_start: u32 = 0u;
    while (m_start < m) {
        let chunk = min(m_tile, m - m_start);

        // Cooperatively load A_tile. A is row-major so
        // A[m_start:m_start+chunk, :] lives contiguously starting at A[m_start*r].
        let total = chunk * r;
        var li: u32 = lid.x;
        while (li < total) {
            A_tile[li] = A[m_start * r + li];
            li = li + wg_size;
        }
        workgroupBarrier();

        if (j_ok) {
            for (var i: u32 = 0u; i < chunk; i = i + 1u) {
                let b_val = B[(m_start + i) * n + j];
                for (var k: u32 = 0u; k < r; k = k + 1u) {
                    C_local[k * wg_size + lid.x] = fma(
                        A_tile[i * r + k],
                        b_val,
                        C_local[k * wg_size + lid.x],
                    );
                }
            }
        }
        workgroupBarrier();

        m_start = m_start + m_tile;
    }

    if (j_ok) {
        for (var k: u32 = 0u; k < r; k = k + 1u) {
            C[k * n + j] = C_local[k * wg_size + lid.x];
        }
    }
}
