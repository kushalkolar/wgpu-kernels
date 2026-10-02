// workgroup sum and max of one value per invocation, tree reductions through workgroup memory
// (Harris "Optimizing Parallel Reduction in CUDA", sequential addressing). The order of the additions is
// fixed, so the sum is deterministic. Uses no subgroup operations, which are an optional feature.
// Must be called from uniform control flow.
//
// Requires: `wg_size` (the workgroup size, a power of 2) to be declared by the including shader.

var<workgroup> reduction_scratch: array<f32, wg_size>;

fn workgroup_sum(x: f32, lid: u32) -> f32 {
    // reduction_scratch may still be read from the previous call
    workgroupBarrier();
    reduction_scratch[lid] = x;
    workgroupBarrier();

    for (var stride = wg_size / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            reduction_scratch[lid] += reduction_scratch[lid + stride];
        }
        workgroupBarrier();
    }

    return reduction_scratch[0];
}

fn workgroup_max(x: f32, lid: u32) -> f32 {
    // reduction_scratch may still be read from the previous call
    workgroupBarrier();
    reduction_scratch[lid] = x;
    workgroupBarrier();

    for (var stride = wg_size / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            reduction_scratch[lid] = max(reduction_scratch[lid], reduction_scratch[lid + stride]);
        }
        workgroupBarrier();
    }

    return reduction_scratch[0];
}
