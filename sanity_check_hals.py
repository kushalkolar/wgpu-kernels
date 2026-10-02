"""
Compare the WGSL HALS updates against masknmf's regression_update, with the real compression results U, V
and synthetic spatial footprints (gaussian discs). The temporal traces are fitted to the footprints and the
movie by masknmf's temporal update in float64, so the updates start from a consistent state.

Each case is compared to masknmf in float32 and float64. The WGSL result must be at least as close to the
float64 result as masknmf's float32 result is, within a factor of 2.
"""

import numpy as np
import pandas as pd
import torch

import fastplotlib as fpl
from masknmf.demixing import regression_update
from masknmf.demixing.signal_demixer import _compute_hals_schedule

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, HALS

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)

dmr_path = "./demix_new.hdf5"

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
n_frames, height, width = compression["shape"]
rank = v.shape[0]

gpu_compression = CompressionBuffers(dmr_path)

# n_clustered signals are placed in an 80 x 80 pixel region: many signals per block and many color groups.
# superblock_size is the side of the superblocks of the temporal partial products, in blocks of U.
# mask_fraction > 0 removes that fraction of each signal's pixels from mask_ab, as after a support update,
# so the signals of a group can overlap in a. frame_batch_size splits the colors into groups.
cases = pd.DataFrame(
    [
        {
            "name": "sparse",
            "n_signals": 200,
            "n_clustered": 0,
            "zero_fraction": 0.0,
            "superblock_size": 1,
            "c_nonneg": True,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
        {
            "name": "dense",
            "n_signals": 600,
            "n_clustered": 300,
            "zero_fraction": 0.05,
            "superblock_size": 3,
            "c_nonneg": True,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
        {
            "name": "dense, c unconstrained",
            "n_signals": 600,
            "n_clustered": 300,
            "zero_fraction": 0.05,
            "superblock_size": 2,
            "c_nonneg": False,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
        {
            "name": "dense, mask_ab smaller than a",
            "n_signals": 600,
            "n_clustered": 300,
            "zero_fraction": 0.05,
            "superblock_size": 4,
            "c_nonneg": True,
            "mask_fraction": 0.3,
            "frame_batch_size": 64,
        },
    ]
)


def make_footprints(
    rng: np.random.Generator, n_signals: int, n_clustered: int, zero_fraction: float
) -> torch.Tensor:
    """gaussian discs, a zero_fraction of the entries of each disc are explicit zeros"""
    rows, cols = np.mgrid[:height, :width]
    entries = []
    for i in range(n_signals):
        if i < n_clustered:
            r0, c0 = rng.uniform(200, 280), rng.uniform(200, 280)
        else:
            r0, c0 = rng.uniform(12, height - 12), rng.uniform(12, width - 12)
        radius = rng.uniform(3, 7)
        d2 = (rows - r0) ** 2 + (cols - c0) ** 2
        mask = d2 <= radius**2
        pixels = (rows[mask] * width + cols[mask]).astype(np.int64)
        values = np.exp(-d2[mask] / (2 * (radius / 2) ** 2)).astype(np.float32)
        values[rng.random(values.size) < zero_fraction] = 0.0
        entries.append((pixels, np.full(pixels.size, i), values))

    pixels, signals, values = (np.concatenate(x) for x in zip(*entries))
    return torch.sparse_coo_tensor(
        np.stack([pixels, signals]), values, (height * width, n_signals)
    ).coalesce()


def masknmf_update(a, c, b, q, blocks, c_nonneg, dtype):
    """one spatial and temporal update with masknmf, returns the values of a and c"""
    # copies: to(), coalesce() and from_numpy() can share memory with the inputs, which spatial_update_hals
    # and temporal_update_hals update in place
    a_ref = a.clone().to(dtype).coalesce()
    c_ref = torch.from_numpy(c.copy()).to(dtype)
    b_ref = torch.from_numpy(b).to(dtype)[:, None]
    q_ref = tuple(torch.from_numpy(x).to(dtype) for x in q)
    u_ref, v_ref = u.to(dtype), v.to(dtype)

    a_ref = regression_update.spatial_update_hals(
        u_ref, v_ref, a_ref, c_ref, b_ref, q=q_ref, blocks=blocks, frame_batch_size=10**9
    )
    c_ref = regression_update.temporal_update_hals(
        u_ref, v_ref, a_ref, c_ref, b_ref, q=q_ref, c_nonneg=c_nonneg, blocks=blocks
    )
    return a_ref.coalesce().values().numpy(), c_ref.numpy()


results = []
for case in cases.itertuples():
    rng = np.random.default_rng(0)
    a = make_footprints(rng, case.n_signals, case.n_clustered, case.zero_fraction)
    b = (0.1 * rng.standard_normal(height * width)).astype(np.float32)
    ring_rank = 6
    q = (
        (0.01 * rng.standard_normal((rank, ring_rank))).astype(np.float32),
        rng.standard_normal((ring_rank, n_frames)).astype(np.float32),
    )

    # pandas stores the missing frame_batch_size as nan
    frame_batch_size = None if pd.isna(case.frame_batch_size) else int(case.frame_batch_size)

    # mask_ab without a fraction of each signal's pixels, at least one pixel of each signal is kept
    mask_ab = None
    if case.mask_fraction > 0:
        keep_entry = rng.random(a.indices().shape[1]) >= case.mask_fraction
        _, first_entry = np.unique(a.indices()[1].numpy(), return_index=True)
        keep_entry[first_entry] = True
        mask_ab = torch.sparse_coo_tensor(
            a.indices()[:, torch.from_numpy(keep_entry)],
            torch.ones(int(keep_entry.sum())),
            a.shape,
        ).coalesce()

    def masknmf_blocks():
        mask = a.bool() if mask_ab is None else mask_ab
        return _compute_hals_schedule(mask, "cpu", frame_batch_size=frame_batch_size or 10**9)

    blocks = masknmf_blocks()

    # consistent starting state: c fitted to a and the movie by temporal updates in float64
    c = torch.zeros(n_frames, case.n_signals, dtype=torch.float64)
    for _ in range(3):
        c = regression_update.temporal_update_hals(
            u.double(),
            v.double(),
            a.double().coalesce(),
            c,
            torch.from_numpy(b).double()[:, None],
            q=tuple(torch.from_numpy(x).double() for x in q),
            c_nonneg=case.c_nonneg,
            blocks=blocks,
        )
    c = c.numpy().astype(np.float32)

    # masknmf's demixing loop deletes signals whose temporal trace is zero ("zero c!"), drop them here too,
    # the updates divide by ||c_i||^2
    keep = np.flatnonzero(np.abs(c).sum(axis=0) > 0)
    n_dropped = case.n_signals - keep.size
    if n_dropped > 0:
        a = torch.index_select(a, 1, torch.from_numpy(keep)).coalesce()
        c = np.ascontiguousarray(c[:, keep])
        if mask_ab is not None:
            mask_ab = torch.index_select(mask_ab, 1, torch.from_numpy(keep)).coalesce()
        blocks = masknmf_blocks()

    gpu_signals = SignalBuffers(
        gpu_compression,
        a,
        c,
        b,
        factorized_ring_term=q,
        mask_ab=mask_ab,
        frame_batch_size=frame_batch_size,
    )
    hals = HALS(
        gpu_compression, gpu_signals, tuning={"partials_superblock_size": case.superblock_size}
    )

    # the groups must match masknmf's blocks, the order of the updates
    s = gpu_signals.structures
    groups = [
        s["group_signals"][s["group_ptr"][g] : s["group_ptr"][g + 1]].tolist()
        for g in range(s["group_ptr"].size - 1)
    ]
    assert groups == [blk.tolist() for blk in blocks], "groups do not match masknmf blocks"
    if mask_ab is not None:
        assert s["group_overlaps"].any(), "no group overlaps in a, the deferred update is not tested"

    a32, c32 = masknmf_update(a, c, b, q, blocks, case.c_nonneg, torch.float32)
    a64, c64 = masknmf_update(a, c, b, q, blocks, case.c_nonneg, torch.float64)

    hals.spatial_update()
    a_gpu = gpu_signals.get_a()
    hals.temporal_update(c_nonneg=case.c_nonneg)
    c_gpu = gpu_signals.get_c()
    assert torch.equal(a.indices(), a_gpu.indices())

    # max abs error relative to the max abs value of the float64 result
    scale_a, scale_c = np.abs(a64).max(), np.abs(c64).max()
    results.append(
        {
            "case": case.name,
            "dropped_signals": n_dropped,
            "groups": len(groups),
            "overlapping_groups": int(s["group_overlaps"].sum()),
            "max_superblock_signals": hals.diff_structures["max_superblock_signals"],
            "superblock_size": case.superblock_size,
            "partial_rows": hals.superblock_structures["n_partial_rows"],
            "a_wgsl": np.abs(a_gpu.values().numpy() - a64).max() / scale_a,
            "a_torch_float32": np.abs(a32 - a64).max() / scale_a,
            "c_wgsl": np.abs(c_gpu - c64).max() / scale_c,
            "c_torch_float32": np.abs(c32 - c64).max() / scale_c,
        }
    )

df = pd.DataFrame(results)
print("errors relative to masknmf in float64")
print(df.to_string(index=False, float_format="%.2e"))

assert np.all(df["a_wgsl"] <= 2 * df["a_torch_float32"]), "a: WGSL error larger than 2x masknmf float32 error"
assert np.all(df["c_wgsl"] <= 2 * df["c_torch_float32"]), "c: WGSL error larger than 2x masknmf float32 error"
print("passed")
