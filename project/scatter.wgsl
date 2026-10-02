// out[indices[i]] = x[i] for i < n, one invocation per entry, indexed as dispatch_grid's 2D grids.

@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
// x: n, a uniform since it changes with the signals
@group(0) @binding(3)
var<uniform> sizes: vec4<u32>;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn scatter(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = gid.y * nwg.x * wg_size + gid.x;
    if (i >= sizes.x) {
        return;
    }
    out[indices[i]] = x[i];
}
