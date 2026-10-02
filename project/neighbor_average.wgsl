// masknmf's local correlation image (local_mad_correlation_mat): the mean over the adjacent pixels of each pixel of the
// correlations of neighbor_correlations.wgsl, each stored at the later pixel of its pair in row-major order as
// (left, up, up left, up right) and summed over the groups of frames. One invocation per pixel.

// [n_groups, n_pixels]
@group(0) @binding(0)
var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(1)
var<storage, read_write> image: array<f32>;

override height: u32;
override width: u32;
override n_groups: u32;

const wg_size: u32 = 256u;

fn correlations(p: u32) -> vec4<f32> {
    let n_pixels = height * width;
    var sum = vec4<f32>();
    for (var g = 0u; g < n_groups; g++) {
        sum += partial[g * n_pixels + p];
    }
    return sum;
}

@compute @workgroup_size(wg_size)
fn neighbor_average(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let p = gid.y * nwg.x * wg_size + gid.x;
    if (p >= height * width) {
        return;
    }
    let row = p / width;
    let col = p % width;
    let own = correlations(p);
    var sum = (own.x + own.y) + (own.z + own.w);
    var count = 0u;
    count += select(0u, 1u, col > 0u);
    count += select(0u, 1u, row > 0u);
    count += select(0u, 1u, row > 0u && col > 0u);
    count += select(0u, 1u, row > 0u && col + 1u < width);
    if (col + 1u < width) {
        sum += correlations(p + 1u).x;
        count += 1u;
    }
    if (row + 1u < height) {
        sum += correlations(p + width).y;
        count += 1u;
        if (col + 1u < width) {
            sum += correlations(p + width + 1u).z;
            count += 1u;
        }
        if (col > 0u) {
            sum += correlations(p + width - 1u).w;
            count += 1u;
        }
    }
    image[p] = sum / f32(count);
}
