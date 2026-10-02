// out[k, j] = sum of partial[c, j] over the tile columns c of column k of U, col_tile_ptr[k]:col_tile_ptr[k + 1] in
// col_tiles, for j < n_out, and 0 for n_out <= j < out_stride
//
// Adds the partial sums of cell_spmm_t. partial is [n_tile_cols, n_j], out is [rank, out_stride].

@group(0) @binding(0)
var<storage, read> col_tile_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> col_tiles: array<u32>;
@group(0) @binding(2)
var<storage, read> partial: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;

override rank: u32;
override n_j: u32;
override n_out: u32;
override out_stride: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn col_reduce(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let index = gid.y * nwg.x * wg_size + gid.x;
    if (index >= rank * out_stride) {
        return;
    }
    let k = index / out_stride;
    let j = index % out_stride;
    var sum = 0.0;
    if (j < n_out) {
        for (var e = col_tile_ptr[k]; e < col_tile_ptr[k + 1u]; e++) {
            sum += partial[col_tiles[e] * n_j + j];
        }
    }
    out[index] = sum;
}
