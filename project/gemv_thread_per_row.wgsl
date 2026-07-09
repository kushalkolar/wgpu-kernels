// Dense GEMV, thread-per-row form for tall matrices (M >> N, small N).
// One lane computes one full row's dot product against v; one workgroup
// covers wg_size rows. No cross-thread reduction and no barriers except the
// initial cooperative load of v into shared memory.
//
// Coalescing requires A in column-major layout: A_col[j * M + row] = A[row, j].
// At iteration j, adjacent lanes (owning adjacent rows) read adjacent
// A_col entries.
//
// A_col is A stored column-major (length M*N), y is [M], v is [N]. All f32.

@group(0) @binding(0)
var<storage, read> A_col: array<f32>;

@group(0) @binding(1)
var<storage, read> v: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

override wg_size: u32;
override N: u32;

var<workgroup> v_shared: array<f32, N>;

@compute @workgroup_size(wg_size)
fn gemv(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let M = arrayLength(&y);
    let row = (wgid.y * nwg.x + wgid.x) * wg_size + lid.x;

    // Cooperatively load v into shared memory. All lanes participate even if
    // their row is out of range, so we hit the barrier uniformly.
    var i: u32 = lid.x;
    while (i < N) {
        v_shared[i] = v[i];
        i = i + wg_size;
    }
    workgroupBarrier();

    if (row >= M) {
        return;
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < N; j = j + 1u) {
        sum = fma(A_col[j * M + row], v_shared[j], sum);
    }
    y[row] = sum;
}
