// Temporal HALS update of the signals in one group (masknmf temporal_update_hals):
//
//   c_i = c_i + (cumulator_i - sum_j ata[i, j] c_j) / ||a_i||^2, clipped at 0 if c_nonneg
//
// where the sum is over i and its neighbors in the overlap graph of a (ata[i, i] = ||a_i||^2), and
// cumulator_i = a_i^T U V - (a_i^T U q0) q1 - a_i^T b: the sum of the partial rows of i from
// temporal_hals_partials, minus the ring term from gemm_nt, minus atb_i.
// The signals of one group are updated in parallel, one workgroup per (signal, wg_size vec4s of frames),
// groups are dispatched in order. Like masknmf, every signal of a group is updated from the values of c
// before the group. If no two signals of the group overlap in a, c_i is read and written in place by the
// same invocation. Otherwise (group.deferred) the new rows are written to c_group and copied into c by
// temporal_hals_copy after the whole group.

struct Group {
    start: u32,
    count: u32,
    // 1 if the signals of the group overlap in a
    deferred: u32,
}

@group(0) @binding(0)
var<uniform> group: Group;
@group(0) @binding(1)
var<storage, read> group_signals: array<u32>;
// c, [n_signals, n_frames_padded]
@group(0) @binding(2)
var<storage, read_write> temporal_demixed: array<vec4<f32>>;
// (a_i^T U q0) q1, [n_signals, n_frames_padded]
@group(0) @binding(3)
var<storage, read> ring_term: array<vec4<f32>>;
@group(0) @binding(4)
var<storage, read> neighbor_ptr: array<u32>;
@group(0) @binding(5)
var<storage, read> neighbors: array<u32>;
@group(0) @binding(6)
var<storage, read> ata: array<f32>;
@group(0) @binding(7)
var<storage, read> a_sq: array<f32>;
// new rows of c of a deferred group, [max group size, n_frames_padded], row g is the g-th signal of the group
@group(0) @binding(8)
var<storage, read_write> c_group: array<vec4<f32>>;
// [n_partial_rows, n_frames_padded], the partial rows of signal i are
// signal_partials[signal_partial_ptr[i]:signal_partial_ptr[i + 1]]
@group(0) @binding(9)
var<storage, read> partial: array<vec4<f32>>;
@group(0) @binding(10)
var<storage, read> signal_partial_ptr: array<u32>;
@group(0) @binding(11)
var<storage, read> signal_partials: array<u32>;
@group(0) @binding(12)
var<storage, read> atb: array<f32>;
// 0 if there is no ring term, a uniform so that a change of the ring rank does not recreate the pipeline
@group(0) @binding(13)
var<uniform> ring_rank_padded: u32;

override n_frames: u32;
override n_frames_padded: u32;
override c_nonneg: bool;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size)
fn temporal_hals(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n4 = n_frames_padded / 4u;
    let t4 = wid.x * wg_size + lid;
    if (wid.y >= group.count || t4 >= n4) {
        return;
    }
    let i = group_signals[group.start + wid.y];

    let c_i = temporal_demixed[i * n4 + t4];
    let a_sq_i = a_sq[i];

    var ac = a_sq_i * c_i;
    for (var e = neighbor_ptr[i]; e < neighbor_ptr[i + 1u]; e++) {
        ac = fma(vec4<f32>(ata[e]), temporal_demixed[neighbors[e] * n4 + t4], ac);
    }

    var cumulator = vec4<f32>(0.0);
    for (var r = signal_partial_ptr[i]; r < signal_partial_ptr[i + 1u]; r++) {
        cumulator += partial[signal_partials[r] * n4 + t4];
    }
    if (ring_rank_padded > 0u) {
        cumulator -= ring_term[i * n4 + t4];
    }
    cumulator -= vec4<f32>(atb[i]);

    var value = c_i + (cumulator - ac) / a_sq_i;
    if (c_nonneg) {
        value = max(value, vec4<f32>(0.0));
    }

    let frames = vec4<u32>(4u * t4) + vec4<u32>(0u, 1u, 2u, 3u);
    value = select(vec4<f32>(0.0), value, frames < vec4<u32>(n_frames));

    if (group.deferred == 1u) {
        c_group[wid.y * n4 + t4] = value;
    } else {
        temporal_demixed[i * n4 + t4] = value;
    }
}

// copies the new rows of a deferred group from c_group into c
@compute @workgroup_size(wg_size)
fn temporal_hals_copy(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let n4 = n_frames_padded / 4u;
    let t4 = wid.x * wg_size + lid;
    if (wid.y >= group.count || t4 >= n4) {
        return;
    }
    let i = group_signals[group.start + wid.y];
    temporal_demixed[i * n4 + t4] = c_group[wid.y * n4 + t4];
}
