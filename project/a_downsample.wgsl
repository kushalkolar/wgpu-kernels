// values[q] = scale * sum of a_values[entries[e]] over e in entry_ptr[q]:entry_ptr[q + 1]
//
// The spatial footprints a downsampled by bins of pixels, one value per (bin, signal) pair q: its entries are the
// entries of a of the signal in the bin. With scale -1 / factor^2 the values are those of masknmf's downsample_sparse,
// negated so that rows_combination subtracts the signals from the downsampled movie.

@group(0) @binding(0)
var<storage, read> entry_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> entries: array<u32>;
@group(0) @binding(2)
var<storage, read> a_values: array<f32>;
@group(0) @binding(3)
var<storage, read_write> values: array<f32>;

override n_pairs: u32;
override scale: f32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn a_downsample(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let q = gid.y * nwg.x * wg_size + gid.x;
    if (q >= n_pairs) {
        return;
    }
    var sum = 0.0;
    for (var e = entry_ptr[q]; e < entry_ptr[q + 1u]; e++) {
        sum += a_values[entries[e]];
    }
    values[q] = scale * sum;
}
