"""
Check that HALS.set_signals gives the same results as a new HALS. One HALS is reused for a sequence of
signals whose index structures, number of signals and ring rank change. For each, a temporal update without
the non-negativity constraint, a spatial update and a temporal update must give bitwise identical a and c
with the reused HALS and with a new HALS.

Also prints the time of set_signals and of creating a new HALS, the time of the first updates after each,
which includes creating the pipelines, and the number of pipelines created.
"""

import time

import numpy as np
import pandas as pd
import pygfx
import torch

import fastplotlib as fpl

from project import select_adapter, CompressionBuffers
from project import _spmv
from project._hals import SignalBuffers, HALS

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)

compression = CompressionBuffers("./demix_new.hdf5")
height, width = compression.fov_shape
device = pygfx.renderers.wgpu.get_shared().device

# count the pipelines created when the updates are recorded
n_pipelines_created = 0
encode = _spmv.ComputeShader.encode


def counting_encode(shader, *args, **kwargs):
    global n_pipelines_created
    n_pipelines_created += shader._pipeline is None
    return encode(shader, *args, **kwargs)


_spmv.ComputeShader.encode = counting_encode

# in this order the workgroup arrays and the temporary buffers grow, stay and grow again, the ring rank
# changes, and the ring term is removed and set again
cases = pd.DataFrame(
    [
        {
            "name": "sparse",
            "n_signals": 200,
            "n_clustered": 0,
            "ring_rank": 6,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
        {
            "name": "dense, mask_ab smaller than a",
            "n_signals": 600,
            "n_clustered": 300,
            "ring_rank": 10,
            "mask_fraction": 0.3,
            "frame_batch_size": 64,
        },
        {
            "name": "dense, no ring term",
            "n_signals": 600,
            "n_clustered": 300,
            "ring_rank": 0,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
        {
            "name": "sparse, 1500 signals",
            "n_signals": 1500,
            "n_clustered": 0,
            "ring_rank": 168,
            "mask_fraction": 0.0,
            "frame_batch_size": None,
        },
    ]
)


def make_signals(
    rng: np.random.Generator,
    n_signals: int,
    n_clustered: int,
    ring_rank: int,
    mask_fraction: float,
) -> dict:
    """
    gaussian discs for a, n_clustered of them in an 80 x 80 pixel region, random c, b and factorized ring
    term, and mask_ab without mask_fraction of each signal's pixels if mask_fraction > 0
    """
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
        entries.append((pixels, np.full(pixels.size, i), values))

    pixels, signals, values = (np.concatenate(x) for x in zip(*entries))
    a = torch.sparse_coo_tensor(
        np.stack([pixels, signals]), values, (height * width, n_signals)
    ).coalesce()

    factorized_ring_term = None
    if ring_rank > 0:
        factorized_ring_term = (
            (0.01 * rng.standard_normal((compression.rank, ring_rank))).astype(np.float32),
            rng.standard_normal((ring_rank, compression.n_frames)).astype(np.float32),
        )

    mask_ab = None
    if mask_fraction > 0:
        keep_entry = rng.random(a.indices().shape[1]) >= mask_fraction
        _, first_entry = np.unique(a.indices()[1].numpy(), return_index=True)
        keep_entry[first_entry] = True
        mask_ab = torch.sparse_coo_tensor(
            a.indices()[:, torch.from_numpy(keep_entry)],
            torch.ones(int(keep_entry.sum())),
            a.shape,
        ).coalesce()

    return {
        "a": a,
        "c": np.abs(rng.standard_normal((compression.n_frames, n_signals))).astype(np.float32),
        "b": (0.1 * rng.standard_normal(height * width)).astype(np.float32),
        "factorized_ring_term": factorized_ring_term,
        "mask_ab": mask_ab,
    }


def run_updates(hals: HALS):
    """the updates, until the GPU is done"""
    hals.temporal_update(c_nonneg=False)
    hals.spatial_update()
    hals.temporal_update(c_nonneg=True)
    device._poll_wait()


# the reused HALS, its pipelines are created by a first run of the updates
hals = HALS(
    compression, SignalBuffers(compression, **make_signals(np.random.default_rng(1), 200, 0, 6, 0.0))
)
run_updates(hals)

results = []
for case in cases.itertuples():
    inputs = make_signals(
        np.random.default_rng(0),
        case.n_signals,
        case.n_clustered,
        case.ring_rank,
        case.mask_fraction,
    )
    # pandas stores the missing frame_batch_size as nan
    frame_batch_size = None if pd.isna(case.frame_batch_size) else int(case.frame_batch_size)

    t0 = time.perf_counter()
    signals = SignalBuffers(compression, **inputs, frame_batch_size=frame_batch_size)
    t1 = time.perf_counter()
    hals.set_signals(signals)
    t2 = time.perf_counter()
    n_pipelines_created = 0
    run_updates(hals)
    t3 = time.perf_counter()
    reused_pipelines = n_pipelines_created

    new_signals = SignalBuffers(compression, **inputs, frame_batch_size=frame_batch_size)
    t4 = time.perf_counter()
    new_hals = HALS(compression, new_signals)
    t5 = time.perf_counter()
    n_pipelines_created = 0
    run_updates(new_hals)
    t6 = time.perf_counter()
    new_pipelines = n_pipelines_created

    a, c = signals.get_a().values().numpy(), signals.get_c()
    a_new, c_new = new_signals.get_a().values().numpy(), new_signals.get_c()
    assert np.isfinite(a).all() and np.isfinite(c).all(), f"{case.name}: non-finite values"

    # the same updates again with the reused HALS, without creating pipelines
    t7 = time.perf_counter()
    run_updates(hals)
    t8 = time.perf_counter()

    s = signals.structures
    results.append(
        {
            "case": case.name,
            "n_signals": case.n_signals,
            "ring_rank": case.ring_rank,
            "overlapping_groups": int(s["group_overlaps"].sum()),
            "max_superblock_signals": hals.diff_structures["max_superblock_signals"],
            "identical": np.array_equal(a, a_new) and np.array_equal(c, c_new),
            "signal_buffers_ms": 1e3 * (t1 - t0),
            "set_signals_ms": 1e3 * (t2 - t1),
            "new_hals_ms": 1e3 * (t5 - t4),
            "first_updates_reused_ms": 1e3 * (t3 - t2),
            "first_updates_new_ms": 1e3 * (t6 - t5),
            "updates_ms": 1e3 * (t8 - t7),
            "pipelines_reused": reused_pipelines,
            "pipelines_new": new_pipelines,
        }
    )
    del new_hals, new_signals

df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.1f"))

assert df["identical"].all(), "set_signals gives different results than a new HALS"
print("passed")
