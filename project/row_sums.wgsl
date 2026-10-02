// out[i] = sum of x[i, first:last] for each row i of x, [n_rows, x_stride4] vec4s row-major. One workgroup per row and
// pairwise summation, for sums close in accuracy to masknmf's (torch.sum, cascade summation on the CPU): each chunk of
// 4 wg_size values is summed as one vec4 per invocation, pairwise within the vec4, and a tree through workgroup memory
// (workgroup_sum), the sums of the chunks again by a tree.

@group(0) @binding(0)
var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

override x_stride4: u32;
// columns of the sums, first <= last <= 4 x_stride4 and at most 4 wg_size^2 columns
override first: u32;
override last: u32;

const wg_size: u32 = 256u;

var<workgroup> chunk_sums: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn row_sums(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x * x_stride4;
    let first4 = first / 4u;
    let end4 = (last + 3u) / 4u;
    let n_chunks = (end4 - first4 + wg_size - 1u) / wg_size;
    for (var chunk = 0u; chunk < n_chunks; chunk++) {
        let t4 = first4 + chunk * wg_size + lid;
        let t = vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u);
        let inside = (t >= vec4<u32>(first)) & (t < vec4<u32>(last));
        let v = select(vec4<f32>(), x[row + min(t4, max(end4, 1u) - 1u)], inside);
        let s = workgroup_sum((v.x + v.y) + (v.z + v.w), lid);
        if (lid == 0u) {
            chunk_sums[chunk] = s;
        }
    }
    workgroupBarrier();
    let total = workgroup_sum(select(0.0, chunk_sums[lid], lid < n_chunks), lid);
    if (lid == 0u) {
        out[wid.x] = total;
    }
}
