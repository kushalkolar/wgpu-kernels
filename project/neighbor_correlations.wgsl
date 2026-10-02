// masknmf's correlations of the thresholded, normalized traces of adjacent pixels in local correlation images
// (get_local_correlation_structure): for each pixel p of a band and its neighbors q to the left, above, above left and
// above right, so that each pair of adjacent pixels has one, the sum over a group of frames of y_p y_q with
// y = (r - mean) inv on the kept frames and 0 on the others, see trace_statistics.wgsl. The row above the band is the
// last row of the previous band (carry_traces, carry_stats). partial[group, p] = (left, up, up left, up right), 0 for
// neighbors outside the fov.
//
// One workgroup per cell of the band and group of frames, one invocation per pixel of the cell: the values y of the
// cell's pixels, of the row above and of the columns to the left and right, (cell_size + 1) x (cell_size + 2) pixels,
// are staged in workgroup memory for chunk4 vec4s of frames at a time.

// [n_cells_x cell_size^2, n4]
@group(0) @binding(0)
var<storage, read> traces: array<vec4<f32>>;
// [width, n4]
@group(0) @binding(1)
var<storage, read> carry_traces: array<vec4<f32>>;
// (median, threshold, mean, inv) per row of traces and carry_traces
@group(0) @binding(2)
var<storage, read> stats: array<vec4<f32>>;
@group(0) @binding(3)
var<storage, read> carry_stats: array<vec4<f32>>;
// [n_groups, n_pixels]
@group(0) @binding(4)
var<storage, read_write> partial: array<vec4<f32>>;
// x: the row of cells of the band, y: 1 if the band has a row above, a uniform with a row per band
@group(0) @binding(5)
var<uniform> band: vec4<u32>;

override cell_size: u32;
// number of cells along the fov width and height
override n_cells_x: u32;
override n_cells_y: u32;
// vec4s per row of traces
override n4: u32;
override n_frames: u32;
// vec4s of frames per group
override group4: u32;
// 0: unconstrained, 1: positive, 2: negative
override sign: u32;

const chunk4: u32 = 8u;

var<workgroup> values: array<vec4<f32>, (cell_size + 1u) * (cell_size + 2u) * chunk4>;

// y of the vec4 t4 of a trace with its statistics
fn normalized(x: vec4<f32>, s: vec4<f32>, t4: u32) -> vec4<f32> {
    let frames = vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(n_frames);
    var keep = frames & (abs(x - s.x) >= vec4<f32>(s.y));
    if (sign == 1u) {
        keep &= x > vec4<f32>(s.x);
    } else if (sign == 2u) {
        keep &= x < vec4<f32>(s.x);
    }
    return select(vec4<f32>(), (x - s.z) * s.w, keep);
}

fn hsum(v: vec4<f32>) -> f32 {
    return (v.x + v.y) + (v.z + v.w);
}

// workgroup x indexes the groups of frames, workgroup y the cells of the band
@compute @workgroup_size(cell_size * cell_size)
fn neighbor_correlations(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n_cell_pixels = cell_size * cell_size;
    let width = n_cells_x * cell_size;
    let region_width = cell_size + 2u;
    let n_values = (cell_size + 1u) * region_width * chunk4;
    let first_col = wid.y * cell_size;
    let group_end4 = min((wid.x + 1u) * group4, n4);

    // this invocation's pixel in the region, whose row 0 is the row above the band and column 0 left of the cell
    let row = lid / cell_size;
    let col = lid % cell_size;
    let center = (row + 1u) * region_width + col + 1u;

    var left = vec4<f32>();
    var up = vec4<f32>();
    var up_left = vec4<f32>();
    var up_right = vec4<f32>();
    for (var first4 = wid.x * group4; first4 < group_end4; first4 += chunk4) {
        for (var i = lid; i < n_values; i += n_cell_pixels) {
            let region_row = i / (region_width * chunk4);
            let fov_col = first_col + (i / chunk4) % region_width - 1u;
            let t4 = first4 + i % chunk4;
            var y = vec4<f32>();
            if (fov_col < width && t4 < group_end4) {
                if (region_row == 0u) {
                    if (band.y != 0u) {
                        y = normalized(carry_traces[fov_col * n4 + t4], carry_stats[fov_col], t4);
                    }
                } else {
                    let q = (region_row - 1u) * width + fov_col;
                    y = normalized(traces[q * n4 + t4], stats[q], t4);
                }
            }
            values[i] = y;
        }
        workgroupBarrier();

        for (var j = 0u; j < chunk4; j++) {
            let y = values[center * chunk4 + j];
            left = fma(y, values[(center - 1u) * chunk4 + j], left);
            up = fma(y, values[(center - region_width) * chunk4 + j], up);
            up_left = fma(y, values[(center - region_width - 1u) * chunk4 + j], up_left);
            up_right = fma(y, values[(center - region_width + 1u) * chunk4 + j], up_right);
        }
        workgroupBarrier();
    }

    let n_pixels = width * n_cells_y * cell_size;
    let p = (band.x * cell_size + row) * width + first_col + col;
    partial[wid.x * n_pixels + p] = vec4<f32>(hsum(left), hsum(up), hsum(up_left), hsum(up_right));
}
