// the y values of the positions [n_frames, 3] of a line = the trace of a signal, c[signal, :n_frames], and
// maxima[slot] = the max of the trace. One workgroup.

// c, [n_signals, n_frames_padded]
@group(0) @binding(0)
var<storage, read> temporal_demixed: array<f32>;
@group(0) @binding(1)
var<storage, read_write> positions: array<f32>;
@group(0) @binding(2)
var<storage, read_write> maxima: array<f32>;
// uniforms, since they change with every line
struct Trace {
    signal: u32,
    slot: u32,
}
@group(0) @binding(3)
var<uniform> trace: Trace;

override n_frames: u32;
override n_frames_padded: u32;

const wg_size: u32 = 256u;

@compute @workgroup_size(wg_size)
fn signal_trace(@builtin(local_invocation_index) lid: u32) {
    let row = trace.signal * n_frames_padded;
    var trace_max = -3.40282347e38;
    for (var i = lid; i < n_frames; i += wg_size) {
        let value = temporal_demixed[row + i];
        positions[3u * i + 1u] = value;
        trace_max = max(trace_max, value);
    }
    let total_max = workgroup_max(trace_max, lid);
    if (lid == 0u) {
        maxima[trace.slot] = total_max;
    }
}
