// out [n_rows, out_stride]: column j < n_gathered is column columns[j] of x [n_rows, x_stride], the next n_y columns
// are the columns of y [n_rows, y_stride], the others 0. One invocation per entry of out, workgroups along x cover a
// row.

@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> columns: array<u32>;
@group(0) @binding(2)
var<storage, read> y: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;

struct Sizes {
    x_stride: u32,
    y_stride: u32,
    out_stride: u32,
    n_gathered: u32,
    n_y: u32,
}

// uniforms, since they change with the signals
@group(0) @binding(4)
var<uniform> sizes: Sizes;

override n_rows: u32;

@compute @workgroup_size(256)
fn assemble_columns(@builtin(global_invocation_id) gid: vec3u) {
    let j = gid.x;
    let i = gid.y;
    if (j >= sizes.out_stride || i >= n_rows) {
        return;
    }
    var value = 0.0;
    if (j < sizes.n_gathered) {
        value = x[i * sizes.x_stride + columns[j]];
    } else if (j < sizes.n_gathered + sizes.n_y) {
        value = y[i * sizes.y_stride + j - sizes.n_gathered];
    }
    out[i * sizes.out_stride + j] = value;
}
