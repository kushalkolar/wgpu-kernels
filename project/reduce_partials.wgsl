// Reduce y_partials[M, K_chunks] to y[M] along the K_chunks axis.
// Pass 2 of the split-K fat GEMV. Also usable standalone as a per-row sum.
//
// wg_size must be a power of two.
//
// Dispatch: (M, 1, 1). One workgroup per row.

@group(0) @binding(0)
var<storage, read> y_partials: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

override wg_size: u32;
override K_chunks: u32;

var<workgroup> sdata: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn reduce(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let row = wgid.x;
    let row_base = row * K_chunks;

    var sum_intermediate: f32 = 0.0;
    var j: u32 = lid.x;
    while (j < K_chunks) {
        sum_intermediate = sum_intermediate + y_partials[row_base + j];
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
