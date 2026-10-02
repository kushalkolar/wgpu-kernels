// masknmf's peaks of a local correlation image (find_local_peaks_2d): the pixels whose value is the max of the
// (2 radius + 1) x (2 radius + 1) window around them, the window cut at the fov, and above threshold, with the pixels of
// the border taken as 0. peaks[p] = 1 at the peaks, 0 elsewhere. One invocation per pixel.

@group(0) @binding(0)
var<storage, read> image: array<f32>;
@group(0) @binding(1)
var<storage, read_write> peaks: array<u32>;

override height: u32;
override width: u32;
override radius: u32;
override threshold: f32;

const wg_size: u32 = 256u;

// the image with the pixels of the border set to 0
fn value(row: u32, col: u32) -> f32 {
    if (row == 0u || col == 0u || row + 1u == height || col + 1u == width) {
        return 0.0;
    }
    return image[row * width + col];
}

@compute @workgroup_size(wg_size)
fn local_peaks(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let p = gid.y * nwg.x * wg_size + gid.x;
    if (p >= height * width) {
        return;
    }
    let row = p / width;
    let col = p % width;
    let x = value(row, col);
    var window_max = x;
    for (var r = max(row, radius) - radius; r <= min(row + radius, height - 1u); r++) {
        for (var c = max(col, radius) - radius; c <= min(col + radius, width - 1u); c++) {
            window_max = max(window_max, value(r, c));
        }
    }
    peaks[p] = select(0u, 1u, x == window_max && x > threshold);
}
