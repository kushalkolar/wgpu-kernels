// hals_sweep_H.wgsl
// Fused HALS inner sweep for the H update. Runs n_inner_sweeps full
// row-sweeps of H in a single dispatch, keeping the H column strip and Q
// entirely in workgroup shared memory across all sweeps.
//
// Each workgroup owns wg_size columns of H (thread lid.x owns column
// j = c0 + lid.x). Since the row update
//
//   H(ℓ, j) ← max(eps, [P(ℓ, j) - Σ_{k≠ℓ} Q(ℓ, k) H(k, j)] / Q(ℓ, ℓ))
//
// only touches column j, workgroups are fully independent.
//
// Global memory traffic per outer iteration:
//   - H read:  r · n (once at start)
//   - H write: r · n (once at end)
//   - P read:  n_inner_sweeps · r · n (per inner update, coalesced,
//              hot in L2 across sweeps)
//   - Q read:  r · r (once, small)
// vs O(n_inner_sweeps · r · n) reads and writes on H without fusion.
//
// Workgroup memory budget (r × wg_size · 4 + r² · 4 bytes) must fit under
// the WGPU limit (typically 16 KB). Driver picks wg_size accordingly.

@group(0) @binding(0) var<storage, read> P: array<f32>;       // r × n
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // r × r
@group(0) @binding(2) var<storage, read_write> H: array<f32>; // r × n

override r: u32;
override wg_size: u32;
override n: u32;
override n_inner_sweeps: u32;
override eps: f32;

var<workgroup> H_local: array<f32, r * wg_size>;
var<workgroup> Q_local: array<f32, r * r>;

@compute @workgroup_size(wg_size)
fn hals_sweep(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let j = wgid.x * wg_size + lid.x;
    let j_ok = j < n;

    var qi: u32 = lid.x;
    while (qi < r * r) {
        Q_local[qi] = Q[qi];
        qi = qi + wg_size;
    }

    if (j_ok) {
        for (var k: u32 = 0u; k < r; k = k + 1u) {
            H_local[k * wg_size + lid.x] = H[k * n + j];
        }
    }
    workgroupBarrier();

    for (var sweep: u32 = 0u; sweep < n_inner_sweeps; sweep = sweep + 1u) {
        for (var l: u32 = 0u; l < r; l = l + 1u) {
            let q_ll = Q_local[l * r + l];
            if (j_ok && q_ll > 0.0) {
                // Full row dot product then subtract the k=ℓ term
                // (equivalent to skipping k=ℓ in the sum but keeps the
                // inner loop branch-free).
                var s: f32 = 0.0;
                for (var k: u32 = 0u; k < r; k = k + 1u) {
                    s = fma(Q_local[l * r + k], H_local[k * wg_size + lid.x], s);
                }
                let s_excl = s - q_ll * H_local[l * wg_size + lid.x];
                let num = P[l * n + j] - s_excl;
                H_local[l * wg_size + lid.x] = max(eps, num / q_ll);
            }
            workgroupBarrier();
        }
    }

    if (j_ok) {
        for (var k: u32 = 0u; k < r; k = k + 1u) {
            H[k * n + j] = H_local[k * wg_size + lid.x];
        }
    }
}
