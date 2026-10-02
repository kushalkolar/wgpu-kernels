// out[i] = x[i] + alpha y[i y_stride] for i < n, one invocation per entry, indexed as dispatch_grid's 2D grids.

@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> y: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
// x: n, y: y_stride; uniforms, since they change with the signals
@group(0) @binding(3)
var<uniform> sizes: vec4<u32>;

override alpha: f32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn axpy(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = gid.y * nwg.x * wg_size + gid.x;
    if (i >= sizes.x) {
        return;
    }
    out[i] = fma(alpha, y[i * sizes.y], x[i]);
}
