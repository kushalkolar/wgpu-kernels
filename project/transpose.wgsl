// dst = src^T, src is [n_rows, n_cols] and dst is [n_cols, dst_stride], row-major, its columns n_rows:dst_stride not
// written
//
// Through a 16 x 16 tile of workgroup memory, with one float of padding per row so that the reads of a column of the
// tile fall into different banks. Adjacent invocations read and write adjacent floats.

@group(0) @binding(0)
var<storage, read> src: array<f32>;
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

override n_rows: u32;
override n_cols: u32;
// at least n_rows
override dst_stride: u32;

var<workgroup> tile: array<array<f32, 17>, 16>;

// workgroup x indexes the tiles along the columns of src, workgroup y along its rows
@compute @workgroup_size(16, 16)
fn transpose(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let row = wid.y * 16u + lid.y;
    let col = wid.x * 16u + lid.x;
    if (row < n_rows && col < n_cols) {
        tile[lid.y][lid.x] = src[row * n_cols + col];
    }
    workgroupBarrier();

    let dst_row = wid.x * 16u + lid.y;
    let dst_col = wid.y * 16u + lid.x;
    if (dst_row < n_cols && dst_col < n_rows) {
        dst[dst_row * dst_stride + dst_col] = tile[lid.x][lid.y];
    }
}
