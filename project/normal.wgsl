// out[i, t] standard normal for i < n_used and t < n_cols_used, 0 elsewhere, out is [n_rows, n_cols]
//
// Counter-based: each value is computed from (seed, i, t) with the PCG hash, two uniforms turned into a normal by the
// Box-Muller transform, so the result does not depend on the dispatch.

@group(0) @binding(0)
var<storage, read_write> out: array<f32>;

override n_rows: u32;
override n_cols: u32;
override n_used: u32;
override n_cols_used: u32;

@group(0) @binding(1)
var<uniform> seed: u32;

const wg_size: u32 = 256u;

// PCG hash, see Jarzynski and Olano, "Hash Functions for GPU Rendering", 2020
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// uniform in (0, 1]
fn unit_interval(h: u32) -> f32 {
    return (f32(h >> 8u) + 1.0) / 16777216.0;
}

@compute @workgroup_size(wg_size)
fn normal(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let index = gid.y * nwg.x * wg_size + gid.x;
    if (index >= n_rows * n_cols) {
        return;
    }
    let i = index / n_cols;
    let t = index % n_cols;
    var value = 0.0;
    if (i < n_used && t < n_cols_used) {
        let h = pcg(seed ^ pcg(index));
        let u1 = unit_interval(h);
        let u2 = unit_interval(pcg(h));
        value = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    }
    out[index] = value;
}
