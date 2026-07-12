// hals_sweep_W.wgsl
// Fused HALS inner sweep for the W column update. Symmetric to
// hals_sweep_H, but W and R are stored m × r row-major (natural W layout)
// so each thread's per-column-of-W reads have stride r across the
// workgroup — uncoalesced from global memory. We fix this with a
// cooperative load that walks the strip in linear order (coalesced) and
// transposes into workgroup memory. Shared-memory stride is padded to
// (wg_size + 1) to avoid bank conflicts on the transposed writes.
//
// Update:
//   W(i, ℓ) ← max(eps, [R(i, ℓ) - Σ_{k≠ℓ} S(k, ℓ) W(i, k)] / S(ℓ, ℓ))
//
// S is symmetric (S = H H^T), so S(k, ℓ) = S(ℓ, k) and we use S(ℓ, k)
// via the same row-of-S indexing as the H sweep.

@group(0) @binding(0) var<storage, read> R: array<f32>;       // m × r
@group(0) @binding(1) var<storage, read> S: array<f32>;       // r × r
@group(0) @binding(2) var<storage, read_write> W: array<f32>; // m × r

override r: u32;
override wg_size: u32;
override m: u32;
override n_inner_sweeps: u32;
override eps: f32;

var<workgroup> W_strip: array<f32, r * (wg_size + 1u)>;
var<workgroup> R_strip: array<f32, r * (wg_size + 1u)>;
var<workgroup> S_local: array<f32, r * r>;

@compute @workgroup_size(wg_size)
fn hals_sweep(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let i0 = wgid.x * wg_size;
    let i = i0 + lid.x;
    let i_ok = i < m;
    let stride = wg_size + 1u;

    // Load S (small r × r).
    var qi: u32 = lid.x;
    while (qi < r * r) {
        S_local[qi] = S[qi];
        qi = qi + wg_size;
    }

    // Cooperative transposed load. Adjacent threads step through global
    // memory linearly (coalesced); destination indexing does the transpose.
    let strip_len = wg_size * r;
    var li: u32 = lid.x;
    while (li < strip_len) {
        let i_local = li / r;
        let k = li % r;
        let gi = i0 + i_local;
        var w_val: f32 = 0.0;
        var r_val: f32 = 0.0;
        if (gi < m) {
            w_val = W[gi * r + k];
            r_val = R[gi * r + k];
        }
        W_strip[k * stride + i_local] = w_val;
        R_strip[k * stride + i_local] = r_val;
        li = li + wg_size;
    }
    workgroupBarrier();

    for (var sweep: u32 = 0u; sweep < n_inner_sweeps; sweep = sweep + 1u) {
        for (var l: u32 = 0u; l < r; l = l + 1u) {
            let s_ll = S_local[l * r + l];
            if (i_ok && s_ll > 0.0) {
                var s: f32 = 0.0;
                for (var k: u32 = 0u; k < r; k = k + 1u) {
                    s = fma(S_local[l * r + k], W_strip[k * stride + lid.x], s);
                }
                let s_excl = s - s_ll * W_strip[l * stride + lid.x];
                let num = R_strip[l * stride + lid.x] - s_excl;
                W_strip[l * stride + lid.x] = max(eps, num / s_ll);
            }
            workgroupBarrier();
        }
    }

    var wi: u32 = lid.x;
    while (wi < strip_len) {
        let i_local = wi / r;
        let k = wi % r;
        let gi = i0 + i_local;
        if (gi < m) {
            W[gi * r + k] = W_strip[k * stride + i_local];
        }
        wi = wi + wg_size;
    }
}
