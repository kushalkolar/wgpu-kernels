// Dense GEMV, pass 1 of the split-K "fat" form.
// For matrices where M is small (few rows) and N is large, one-workgroup-per-row
// leaves the GPU underutilized. This kernel splits each row's reduction across
// K_chunks workgroups so the dispatch has (M * K_chunks) workgroups. Each
// workgroup handles a (row, column-chunk) pair and writes one partial into
// y_partials[row, chunk_id]. A second pass reduces along chunk_id.
//
// Avoiding f32 atomic add (not portable) is why we split into two passes.
//
// wg_size must be a power of two.
//
// Dispatch:  (K_chunks, M, 1)
// A is [M, N] row-major, v is [N], y_partials is [M, K_chunks]. All f32.

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> v: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y_partials: array<f32>;

override wg_size: u32;
override N: u32;
override chunk_size: u32;   // columns processed by one workgroup
override K_chunks: u32;     // total chunks per row

var<workgroup> sdata: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn gemv_partial(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let chunk = wgid.x;
    let row = wgid.y;
    let chunk_start = chunk * chunk_size;
    if (chunk_start >= N) {
        return;
    }
    let chunk_end = min(chunk_start + chunk_size, N);
    let row_base = row * N;

    var sum_intermediate: f32 = 0.0;
    var j: u32 = chunk_start + lid.x;
    while (j < chunk_end) {
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
        y_partials[row * K_chunks + chunk] = sdata[0];
    }
}
