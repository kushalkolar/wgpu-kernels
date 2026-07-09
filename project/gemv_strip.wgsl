// Dense GEMV, strip form.
// One workgroup handles R = rows_per_wg rows and caches a tile of v in
// workgroup shared memory so those R rows share each v-tile read from global
// memory. This trades a bigger workgroup for lower v-bandwidth cost and is a
// win when M is large enough that v gets re-fetched many times in gemv_row.
//
// Each lane holds R accumulators laid out in workgroup memory
// (accum[lane * rows_per_wg + r]). Reduction is done R times through sdata.
//
// wg_size must be a power of two.
//
// A is [M, N] row-major, y is [M], v is [N]. All f32.

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> v: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

override wg_size: u32;
override N: u32;
override rows_per_wg: u32;
override v_tile_size: u32;

var<workgroup> v_shared: array<f32, v_tile_size>;
var<workgroup> sdata: array<f32, wg_size>;
// Per-thread accumulators. Function-scope arrays cannot be sized by override
// constants in WGSL; workgroup ones can.
var<workgroup> accum: array<f32, wg_size * rows_per_wg>;

@compute @workgroup_size(wg_size)
fn gemv(
    @builtin(workgroup_id) wgid: vec3u,
    @builtin(num_workgroups) nwg: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let strip = wgid.y * nwg.x + wgid.x;
    let row0 = strip * rows_per_wg;
    let M = arrayLength(&y);
    if (row0 >= M) {
        return;
    }

    let accum_base = lid.x * rows_per_wg;
    for (var r: u32 = 0u; r < rows_per_wg; r = r + 1u) {
        accum[accum_base + r] = 0.0;
    }

    var t: u32 = 0u;
    while (t < N) {
        let tile_end = min(v_tile_size, N - t);

        // Cooperatively load v[t : t + tile_end] into v_shared.
        var i: u32 = lid.x;
        while (i < tile_end) {
            v_shared[i] = v[t + i];
            i = i + wg_size;
        }
        workgroupBarrier();

        // Each thread walks the tile with stride wg_size, accumulating for
        // every row in the strip against the same cached v_shared entries.
        for (var r: u32 = 0u; r < rows_per_wg; r = r + 1u) {
            let row = row0 + r;
            if (row < M) {
                let row_base = row * N + t;
                let ai = accum_base + r;
                var j: u32 = lid.x;
                while (j < tile_end) {
                    accum[ai] = fma(A[row_base + j], v_shared[j], accum[ai]);
                    j = j + wg_size;
                }
            }
        }
        workgroupBarrier();

        t = t + v_tile_size;
    }

    // Reduce the per-thread accumulators one row at a time through sdata.
    for (var r: u32 = 0u; r < rows_per_wg; r = r + 1u) {
        let row = row0 + r;
        sdata[lid.x] = accum[accum_base + r];
        workgroupBarrier();

        for (var stride: u32 = wg_size / 2u; stride > 0u; stride = stride / 2u) {
            if (lid.x < stride) {
                sdata[lid.x] = sdata[lid.x] + sdata[lid.x + stride];
            }
            workgroupBarrier();
        }

        if (lid.x == 0u && row < M) {
            y[row] = sdata[0];
        }
        workgroupBarrier();
    }
}
