// reduce_ABT.wgsl
// Sums partials[k_chunks × m × r] along the k_chunks axis to produce the
// final C[m × r]. Pass 2 of the split-K gemm_ABT.
//
// One thread per output cell; each reads k_chunks values and sums.

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;

override wg_size: u32;
override m: u32;
override r: u32;
override k_chunks: u32;

@compute @workgroup_size(wg_size)
fn reduce(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let total = m * r;
    if (idx >= total) {
        return;
    }

    var s: f32 = 0.0;
    for (var k: u32 = 0u; k < k_chunks; k = k + 1u) {
        s = s + partials[k * total + idx];
    }
    C[idx] = s;
}
