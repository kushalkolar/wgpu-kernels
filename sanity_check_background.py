"""
Compare the WGSL fluctuating background update (the ring model) against masknmf's fluctuating_baseline_update, with
the real compression results U, V and synthetic spatial footprints (gaussian discs). The temporal traces are fitted to
the footprints and the movie by masknmf's temporal update in float64, so the residual U V - a c^T keeps the real
background. The random matrices of masknmf's randomized SVDs are drawn from a seeded generator and given to both.

Each case is compared to masknmf in float32 and to the same algorithm in float64 (masknmf's code casts to float32 in
places, the float64 version follows it step by step). The WGSL ring term must be at least as close to the float64
result as masknmf's float32 result is, within a factor of 2, and the background ranks must agree.
"""

import types

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing import regression_update
from masknmf.demixing.signal_demixer import DemixingState, _compute_hals_schedule
from masknmf.demixing.background_estimation import RingModel

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline, ring_taps

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
u_proj = compression["u_local_projector"].coalesce()
n_frames, height, width = compression["shape"]
rank = v.shape[0]

gpu_compression = CompressionBuffers(dmr_path)
fluctuating_baseline = FluctuatingBaseline(gpu_compression, u)

# background_rank None estimates the rank first, as masknmf does at the start of demixing and after each support
# update
cases = pd.DataFrame(
    [
        {"name": "sparse", "n_signals": 200, "n_clustered": 0, "background_rank": None},
        {"name": "dense", "n_signals": 600, "n_clustered": 300, "background_rank": None},
        {"name": "dense, given rank", "n_signals": 600, "n_clustered": 300, "background_rank": 40},
    ]
)


def make_footprints(rng: np.random.Generator, n_signals: int, n_clustered: int) -> torch.Tensor:
    """gaussian discs"""
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
    return torch.sparse_coo_tensor(np.stack([pixels, signals]), values, (height * width, n_signals)).coalesce()


def masknmf_ring_update(a, c, background_rank, rng):
    """
    masknmf's fluctuating_baseline_update in float32, its DemixingState methods on the attributes they read, with
    torch.randn replaced by samples from rng. Returns (q0, q1), the background rank and the random matrices drawn
    """
    state = types.SimpleNamespace(
        device="cpu",
        v=v.float(),
        u_sparse=u,
        a=a,
        c=torch.from_numpy(c),
        b=torch.zeros((height * width, 1)),
        d1=height,
        d2=width,
        shape=(height, width, n_frames),
        background_rank=background_rank,
        pmd_obj=types.SimpleNamespace(
            project_frames=lambda frames, standardize=False: torch.sparse.mm(u_proj.T, frames.float())
        ),
        W=RingModel(height, width, 10, "cpu", "C"),
    )
    state.lowrank_background_svd = types.MethodType(DemixingState.lowrank_background_svd, state)
    state.lowrank_ring_update = types.MethodType(DemixingState.lowrank_ring_update, state)

    omegas = []
    randn = torch.randn

    def rng_randn(*size, **kwargs):
        omegas.append(rng.standard_normal(size).astype(np.float32))
        return torch.from_numpy(omegas[-1])

    torch.randn = rng_randn
    try:
        DemixingState.fluctuating_baseline_update(state)
    finally:
        torch.randn = randn
    q0, q1 = state.factorized_ring_term
    return (q0.numpy(), q1.numpy()), state.background_rank, omegas


def torch_to_csr(x: torch.Tensor) -> sp.csr_matrix:
    x = x.coalesce()
    rows, cols = x.indices().numpy()
    return sp.csr_matrix((x.values().numpy().astype(np.float64), (rows, cols)), shape=tuple(x.shape))


def ring_update_float64(a, c, background_rank, omegas, factor=20, radius=10, sketch=300):
    """masknmf's fluctuating_baseline_update step by step in float64, with the given random matrices"""
    U, A, U_proj = torch_to_csr(u), torch_to_csr(a), torch_to_csr(u_proj)
    V, C = v.numpy().astype(np.float64), c.astype(np.float64)
    h, w = height // factor, width // factor

    def downsample(x):
        images = x.reshape(height, width, -1)[: h * factor, : w * factor]
        return images.reshape(h, factor, w, factor, -1).mean(axis=(1, 3)).reshape(h * w, -1)

    def downsample_sparse(m):
        coo = m.tocoo()
        rows, cols = coo.row // width, coo.row % width
        keep = (rows // factor < h) & (cols // factor < w)
        bins = (rows[keep] // factor) * w + cols[keep] // factor
        return sp.csr_matrix((coo.data[keep] / factor**2, (bins, coo.col[keep])), shape=(h * w, m.shape[1]))

    U_ds, A_ds = downsample_sparse(U), downsample_sparse(A)
    omegas = [o.astype(np.float64) for o in omegas]

    def lowrank_background_svd(k, omega):
        q, _ = np.linalg.qr(downsample(U @ (V @ omega) - A @ (C.T @ omega)))
        right = (U_ds.T @ q).T @ V - (A_ds.T @ q).T @ C.T
        _, _, v_bkgd = np.linalg.svd(right, full_matrices=False)
        left = U @ (V @ v_bkgd.T) - A @ (C.T @ v_bkgd.T)
        uu, s, v_left = np.linalg.svd(left, full_matrices=False)
        return uu[:, :k], s[:k], (v_left @ v_bkgd)[:k]

    if background_rank is None:
        _, s, _ = lowrank_background_svd(sketch, omegas.pop(0))
        background_rank = int(np.argmax(np.cumsum(s**2) / np.sum(s**2) >= 0.99)) + 1
    u_bkgd, s_bkgd, v_bkgd = lowrank_background_svd(background_rank, omegas.pop(0))

    # masknmf's ring model: the sum over the ring around p + (1, 1), its kernel is centered with a roll of
    # -kh // 2 = -radius - 1
    x = (U @ (U_proj.T @ u_bkgd)) * s_bkgd[None, :]
    images = x.reshape(height, width, -1)
    wx = np.zeros_like(images)
    for dy, dx in ring_taps(radius):
        oy, ox = 1 + dy, 1 + dx
        ys, ye, xs, xe = max(0, -oy), min(height, height - oy), max(0, -ox), min(width, width - ox)
        wx[ys:ye, xs:xe] += images[ys + oy : ye + oy, xs + ox : xe + ox]
    wx = wx.reshape(height * width, -1)
    with np.errstate(invalid="ignore"):
        weights = np.nan_to_num(np.sum(wx * x, axis=1) / np.sum(wx * wx, axis=1), nan=0.0)
    return (U_proj.T @ (wx * weights[:, None]), v_bkgd), background_rank


def ring_term_error(q, q_ref, probe):
    """max error of q0 q1 applied to probe [n_frames, m], relative to the max of the reference, invariant to the
    signs of the singular vectors"""
    x = q[0].astype(np.float64) @ (q[1].astype(np.float64) @ probe)
    ref = q_ref[0] @ (q_ref[1] @ probe)
    return np.abs(x - ref).max() / np.abs(ref).max()


results = []
probe = np.random.default_rng(1).standard_normal((n_frames, 16))
for case in cases.itertuples():
    rng = np.random.default_rng(0)
    a = make_footprints(rng, case.n_signals, case.n_clustered)
    background_rank = None if pd.isna(case.background_rank) else int(case.background_rank)

    # c fitted to a and the movie by temporal updates in float64, without a ring term
    blocks = _compute_hals_schedule(a.bool(), "cpu", frame_batch_size=10**9)
    c = torch.zeros(n_frames, case.n_signals, dtype=torch.float64)
    for _ in range(3):
        c = regression_update.temporal_update_hals(
            u.double(), v.double(), a.double().coalesce(), c, torch.zeros(height * width, 1, dtype=torch.float64),
            q=None, blocks=blocks,
        )
    c = c.numpy().astype(np.float32)

    q_masknmf, rank_masknmf, omegas = masknmf_ring_update(a, c, background_rank, np.random.default_rng(2))
    q_ref, rank_ref = ring_update_float64(a, c, background_rank, omegas)

    signals = SignalBuffers(gpu_compression, a, c, np.zeros(height * width, np.float32))
    fluctuating_baseline.set_signals(signals)
    rank_wgsl = fluctuating_baseline.update(background_rank, omegas=omegas)
    rrp = signals.ring_rank_padded
    q_wgsl = (
        read_buffer(signals.buffers["ring_left"], np.float32, (rank, rrp))[:, :rank_wgsl],
        read_buffer(signals.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
            :rank_wgsl, :n_frames
        ],
    )

    results.append(
        {
            "case": case.name,
            "n_signals": case.n_signals,
            "rank_masknmf": rank_masknmf,
            "rank_float64": rank_ref,
            "rank_wgsl": rank_wgsl,
            "ring_term_wgsl": ring_term_error(q_wgsl, q_ref, probe),
            "ring_term_torch_float32": ring_term_error(q_masknmf, q_ref, probe),
        }
    )
    print(results[-1])

df = pd.DataFrame(results)
print("errors of the ring term relative to masknmf's algorithm in float64")
print(df.to_string(index=False, float_format="%.2e"))

ranks_agree = (df["rank_wgsl"] == df["rank_masknmf"]) & (df["rank_wgsl"] == df["rank_float64"])
if not ranks_agree.all() or not (df["ring_term_wgsl"] <= 2 * df["ring_term_torch_float32"]).all():
    raise AssertionError("WGSL fluctuating background differs from masknmf")
print("passed")
