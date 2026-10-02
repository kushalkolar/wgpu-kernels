// out[i, j] = sum over the entries e of signal i of a_values[e] * x[j, a_pixels[e]], for j < n_j
//
// The transpose of the spatial footprints a times pixel data x [n_j, n_pixels] (a column per row), out is
// [n_signals, n_j]. One workgroup per signal and chunk of wg_size columns j, one invocation per j.

@group(0) @binding(0)
var<storage, read> a_ptr: array<u32>;
@group(0) @binding(1)
var<storage, read> a_pixels: array<u32>;
@group(0) @binding(2)
var<storage, read> a_values: array<f32>;
@group(0) @binding(3)
var<storage, read> x: array<f32>;
@group(0) @binding(4)
var<storage, read_write> out: array<f32>;
// x: n_signals, a uniform since it changes with the signals
@group(0) @binding(5)
var<uniform> n_signals: vec4<u32>;

override n_pixels: u32;
override n_j: u32;

const wg_size: u32 = 256u;

// workgroup x indexes the chunks of j, workgroup y the signals
@compute @workgroup_size(wg_size)
fn a_spmm_t(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = wid.y;
    let j = wid.x * wg_size + lid;
    if (i >= n_signals.x || j >= n_j) {
        return;
    }
    var sum = 0.0;
    for (var e = a_ptr[i]; e < a_ptr[i + 1u]; e++) {
        sum = fma(a_values[e], x[j * n_pixels + a_pixels[e]], sum);
    }
    out[i * n_j + j] = sum;
}
