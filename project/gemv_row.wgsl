// Dense GEMV, row-per-workgroup form.
// One workgroup computes y[row] = <A[row, :], v>.
// wg_size invocations stride along the row (coalesced reads), each accumulates
// an intermediate sum, then a tree reduction across the workgroup writes the
// final scalar.
//
// wg_size must be a power of two.
//
// A is [M, N] row-major, y is [M], v is [N]. All f32.

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> v: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

override wg_size: u32;
override N: u32;

var<workgroup> sdata: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn gemv(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let row = wgid.y * nwg.x + wgid.x;
    let M = arrayLength(&y);
    if (row >= M) {
        return;
    }

    let row_base = row * N;

    var sum_intermediate: f32 = 0.0;
    var j: u32 = lid.x;
    while (j < N) {
        sum_intermediate = fma(A[row_base + j], v[j], sum_intermediate);
        j = j + wg_size;
    }
    sdata[lid.x] = sum_intermediate;
    workgroupBarrier();

    for (var stride: u32 = wg_size / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            sdata[lid.x] = sdata[lid.x] + sdata[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        y[row] = sdata[0];
    }
}
