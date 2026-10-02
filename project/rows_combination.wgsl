// out[i, :] = base[i, :] + sum of values[e] * rows[cols[e], :] over the entries e of row i, rows_ptr[i]:rows_ptr[i + 1]
//
// A sparse matrix in CSR format times a dense matrix whose rows are n4 vec4s. Each invocation computes one vec4 of a
// row of out, adjacent invocations read adjacent vec4s of the rows. The entries of a row are the same for the whole
// workgroup.

@group(0) @binding(0)
var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> cols: array<u32>;
@group(0) @binding(2)
var<storage, read> values: array<f32>;
@group(0) @binding(3)
var<storage, read> rows: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> base: array<vec4<f32>>;
@group(0) @binding(5)
var<storage, read_write> out: array<vec4<f32>>;

// vec4s per row of rows, base and out
override n4: u32;

const wg_size: u32 = 256u;

// workgroup x indexes chunks of wg_size vec4s of a row, workgroup y the rows of out
@compute @workgroup_size(wg_size)
fn rows_combination(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = wid.y;
    let t4 = wid.x * wg_size + lid;
    if (t4 >= n4) {
        return;
    }
    var sum = base[i * n4 + t4];
    for (var e = row_ptr[i]; e < row_ptr[i + 1u]; e++) {
        sum = fma(vec4<f32>(values[e]), rows[cols[e] * n4 + t4], sum);
    }
    out[i * n4 + t4] = sum;
}
