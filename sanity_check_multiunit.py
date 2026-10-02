"""
Compare the WGSL multiunit factorization (FluctuatingBaseline.get_multiunit_factorization) against masknmf's
extract_multiunit_factorization at the end of a pass, with the real compression results U, V, the signals of
sanity_check_merge.py and the ring term of rank 40 from the WGSL fluctuating background update. Both use the same
random matrix, masknmf's torch.randn replaced by it.

The ranks are compared to masknmf's float32 and to its algorithm in float64 (float64 QR and SVD), with the explained
variance around the 0.99 threshold. The singular values (the norms of the rows of term2) and the low-rank product
term1 term2 truncated to the smallest of the ranks, probed with random frames, must be at least as close to float64 as
masknmf's float32 values are, within a factor of 2.
"""

import types
from unittest import mock

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import DemixingState

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer
from project._background import FluctuatingBaseline

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"
n_signals = 400
test_rank = 200
num_oversamples = 5

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
u_projector = compression["u_local_projector"].coalesce()
n_frames, height, width = compression["shape"]
rank = v.shape[0]
n_pixels = height * width

gpu_compression = CompressionBuffers(dmr_path)

# the signals of sanity_check_merge.py and a ring term for them
rng = np.random.default_rng(1)
rows, cols = np.mgrid[:height, :width]
entries = []
centers = []
for i in range(n_signals):
    if i % 4 == 1:
        r0, c0 = centers[-1][0] + rng.uniform(-1.5, 1.5), centers[-1][1] + rng.uniform(-1.5, 1.5)
    else:
        r0, c0 = rng.uniform(12, height - 12), rng.uniform(12, width - 12)
    centers.append((r0, c0))
    radius = rng.uniform(3, 7)
    d2 = (rows - r0) ** 2 + (cols - c0) ** 2
    mask = d2 <= radius**2
    pixels = (rows[mask] * width + cols[mask]).astype(np.int64)
    entries.append((pixels, np.full(pixels.size, i), np.exp(-d2[mask] / (2 * (radius / 2) ** 2))))
pixels, signal_indices, values = (np.concatenate(x) for x in zip(*entries))
a = torch.sparse_coo_tensor(
    np.stack([pixels, signal_indices]), values.astype(np.float32), (n_pixels, n_signals)
).coalesce()
u_csr = sp.csr_matrix((u.values().numpy().astype(np.float64), u.indices().numpy()), shape=(n_pixels, rank))
a_csr = sp.csr_matrix((values, (pixels, signal_indices)), shape=(n_pixels, n_signals))
weights = np.asarray((a_csr.T @ u_csr).todense()) / np.asarray(a_csr.sum(axis=0)).T
c = (torch.from_numpy(weights.astype(np.float32)) @ v).numpy().T.copy()
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32))
background = FluctuatingBaseline(gpu_compression, u)
background.set_signals(signals)
background.update(40)
rrp = signals.ring_rank_padded
q0 = read_buffer(signals.buffers["ring_left"], np.float32, (rank, rrp))[:, :40].copy()
q1 = read_buffer(signals.buffers["ring_right"], np.float32, (rrp, gpu_compression.n_frames_padded))[
    :40, :n_frames
].copy()
omega = np.random.default_rng(2).standard_normal((n_frames, test_rank + num_oversamples)).astype(np.float32)

# WGSL
term1_wgsl, term2_wgsl = background.get_multiunit_factorization(test_rank, num_oversamples, omega=omega)

# masknmf with the same random matrix
state = types.SimpleNamespace(
    u_sparse=u,
    v=v,
    a=a,
    c=torch.from_numpy(c),
    b=torch.zeros(n_pixels, 1),
    factorized_ring_term=(torch.from_numpy(q0), torch.from_numpy(q1)),
    device="cpu",
    pmd_obj=types.SimpleNamespace(project_frames=lambda frames, standardize: torch.sparse.mm(u_projector.T, frames)),
)
state._multiunit_factorization_routine = types.MethodType(DemixingState._multiunit_factorization_routine, state)
with mock.patch("torch.randn", lambda *args, **kwargs: torch.from_numpy(omega)):
    term1_masknmf, term2_masknmf = DemixingState.extract_multiunit_factorization(state, test_rank, num_oversamples)
term1_masknmf, term2_masknmf = term1_masknmf.numpy(), term2_masknmf.numpy()

# masknmf's algorithm in float64
V = v.numpy().astype(np.float64)
Q0, Q1 = q0.astype(np.float64), q1.astype(np.float64)
C = c.astype(np.float64)
A = a_csr.astype(np.float64)
omega64 = omega.astype(np.float64)
y = u_csr @ (V @ omega64 - Q0 @ (Q1 @ omega64)) - A @ (C.T @ omega64)
y -= y.mean(axis=1, keepdims=True)
q, _ = np.linalg.qr(y)
projection = u_csr.T @ q
right = projection.T @ V - (projection.T @ Q0) @ Q1 - (A.T @ q).T @ C.T
left_svd, s64, right_svd = np.linalg.svd(right, full_matrices=False)
explained64 = np.cumsum(s64[:test_rank] ** 2) / np.sum(s64[:test_rank] ** 2)
r64 = int(np.argmax(explained64 >= 0.99)) + 1
u_projector_csr = sp.csr_matrix(
    (u_projector.values().numpy().astype(np.float64), u_projector.indices().numpy()), shape=(n_pixels, rank)
)
term1_64 = u_projector_csr.T @ (q @ left_svd[:, :r64])
term2_64 = s64[:r64, None] * right_svd[:r64]

ranks = {"wgsl": term1_wgsl.shape[1], "masknmf": term1_masknmf.shape[1], "float64": r64}
print(
    f"ranks: {ranks}, float64 explained variance at ranks {r64 - 1}, {r64}: {explained64[r64 - 2]:.6f}, "
    f"{explained64[r64 - 1]:.6f}"
)


def relative_error(x: np.ndarray, reference: np.ndarray) -> float:
    """max error relative to the max of the reference"""
    return float(np.abs(x - reference).max() / np.abs(reference).max())


# the singular values and the product term1 term2 of the first common components, probed with random frames
r = min(ranks.values())
probes = np.random.default_rng(3).standard_normal((n_frames, 8))
product64 = term1_64[:, :r] @ (term2_64[:r] @ probes)
results = []
for name, term1, term2 in (("wgsl", term1_wgsl, term2_wgsl), ("masknmf", term1_masknmf, term2_masknmf)):
    product = term1[:, :r].astype(np.float64) @ (term2[:r].astype(np.float64) @ probes)
    results.append(
        {
            "implementation": name,
            "singular values": relative_error(np.linalg.norm(term2[:r].astype(np.float64), axis=1), s64[:r]),
            "term1 term2": relative_error(product, product64),
        }
    )
df = pd.DataFrame(results)
print(f"errors relative to float64 over the first {r} components")
print(df.to_string(index=False, float_format="%.2e"))

wgsl, masknmf = df.iloc[0], df.iloc[1]
if not (
    wgsl["singular values"] <= 2 * masknmf["singular values"] and wgsl["term1 term2"] <= 2 * masknmf["term1 term2"]
):
    raise AssertionError("WGSL multiunit factorization differs from masknmf")
print("passed")
