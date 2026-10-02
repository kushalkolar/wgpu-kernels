// out[i, f] = x[i, columns[f]] for f < n_columns and 0 for n_columns <= f < n_out_cols, x is [n_rows, x_stride] and
// out [n_rows, n_out_cols], both row-major. One invocation per entry of out, workgroups along x cover a row.

@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> columns: array<u32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

override n_rows: u32;
override x_stride: u32;
override n_columns: u32;
override n_out_cols: u32;

@compute @workgroup_size(256)
fn gather_columns(@builtin(global_invocation_id) gid: vec3u) {
    let f = gid.x;
    let i = gid.y;
    if (f >= n_out_cols || i >= n_rows) {
        return;
    }
    var value = 0.0;
    if (f < n_columns) {
        value = x[i * x_stride + columns[f]];
    }
    out[i * n_out_cols + f] = value;
}
