// masknmf's ring model with unit weights and support (RingModel.forward): wx[j, p] is the sum of x[j] over the ring
// of pixels around p + (1, 1), zero outside the field of view. masknmf centers the ring kernel with a roll of
// -kh // 2 = -radius - 1 before its FFT convolution, which moves the ring by one pixel in both directions. Also the
// per-pixel weights of masknmf's lowrank_ring_update: weights[p] = sum_j wx[j, p] x[j, p] / sum_j wx[j, p]^2, 0 where
// the denominator is 0.
//
// x and wx are [n_j, height * width], a row-major image per row. One workgroup per 16 x 16 tile of pixels: each image
// of the tile and its halo is staged through workgroup memory, each invocation computes one pixel.

@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> wx: array<f32>;
@group(0) @binding(2)
var<storage, read_write> weights: array<f32>;
// (dy, dx) of the pixels of the ring around (0, 0)
@group(0) @binding(3)
var<storage, read> taps: array<vec2<i32>>;

override height: u32;
override width: u32;
override n_j: u32;
override n_taps: u32;
override radius: u32;

const T: u32 = 16u;
// the ring around p + (1, 1) covers p - (radius - 1) to p + radius + 1
override halo_before: u32 = radius - 1u;
override span: u32 = T + 2u * radius;

var<workgroup> tile: array<f32, span * span>;

@compute @workgroup_size(T, T)
fn ring(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(local_invocation_index) li: u32) {
    let y = wid.y * T + lid.y;
    let x_ = wid.x * T + lid.x;
    let inside = y < height && x_ < width;
    let n_pixels = height * width;
    // image coordinates of tile[0]
    let y0 = i32(wid.y * T) - i32(halo_before);
    let x0 = i32(wid.x * T) - i32(halo_before);

    var num = 0.0;
    var den = 0.0;
    for (var j = 0u; j < n_j; j++) {
        for (var i = li; i < span * span; i += T * T) {
            let yy = y0 + i32(i / span);
            let xx = x0 + i32(i % span);
            var value = 0.0;
            if (yy >= 0 && yy < i32(height) && xx >= 0 && xx < i32(width)) {
                value = x[j * n_pixels + u32(yy) * width + u32(xx)];
            }
            tile[i] = value;
        }
        workgroupBarrier();

        // tile index of p + (1, 1)
        let center = i32((lid.y + halo_before + 1u) * span + lid.x + halo_before + 1u);
        var sum = 0.0;
        for (var k = 0u; k < n_taps; k++) {
            sum += tile[u32(center + taps[k].x * i32(span) + taps[k].y)];
        }
        let own = tile[(lid.y + halo_before) * span + lid.x + halo_before];
        if (inside) {
            wx[j * n_pixels + y * width + x_] = sum;
        }
        num = fma(sum, own, num);
        den = fma(sum, sum, den);
        workgroupBarrier();
    }
    if (inside) {
        weights[y * width + x_] = select(0.0, num / den, den != 0.0);
    }
}
