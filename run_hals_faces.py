"""
NMF (HALS) decomposition of the Olivetti faces dataset on WGPU.

Factors X ∈ R^(4096 × 400) as W H with W ∈ R^(4096 × r), H ∈ R^(r × 400).
The columns of W are the classical NMF face-parts basis; the columns of H
are per-face coefficients.

Two Figures:
1. Grid of the r face parts (columns of W reshaped to 64×64).
2. Reconstruction comparison: a strip of original faces above their HALS
   reconstructions.

HALS runs on the GPU. Buffers are read back to CPU before plotting.
"""

import time

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import NMF
import fastplotlib as fpl

from project import HALS


R = 25
N_OUTER = 100
N_INNER_H = 4
N_INNER_W = 4
IMG_SHAPE = (64, 64)
GRID_PARTS = (5, 5)
N_RECON_SAMPLES = 8


adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
fpl.select_adapter(adapter)


print("Loading Olivetti faces...")
faces = fetch_olivetti_faces()
# faces.data is (n_samples, n_features) = (400, 4096). Factor with the
# Lee-Seung convention: X is (pixels, samples), columns of W are face parts.
X = faces.data.astype(np.float32).T
n_pixels, n_samples = X.shape
print(f"X shape: {X.shape},  rank r = {R}")

rng = np.random.default_rng(0)
W0 = (rng.uniform(0.0, 1.0, size=(n_pixels, R)) + 0.1).astype(np.float32)
H0 = (rng.uniform(0.0, 1.0, size=(R, n_samples)) + 0.1).astype(np.float32)

print(f"Running HALS: {N_OUTER} outer iters, "
      f"n_inner_H={N_INNER_H}, n_inner_W={N_INNER_W}")
hals = HALS(X, W0, H0, wg_size=64)
t0 = time.perf_counter()
hals.fit(N_OUTER, N_INNER_H, N_INNER_W)
elapsed_ms = (time.perf_counter() - t0) * 1000.0

W = hals.get_W()
H = hals.get_H()
err = hals.error()
print(f"  done in {elapsed_ms:.1f} ms,  final relative Frobenius error {err:.4f}")

face_parts = W.T.reshape(R, *IMG_SHAPE)
originals = X.T.reshape(n_samples, *IMG_SHAPE)
reconstructions = (W @ H).T.reshape(n_samples, *IMG_SHAPE)


fig_parts = fpl.Figure(shape=GRID_PARTS, size=(1000, 1000))
for i in range(R):
    row, col = divmod(i, GRID_PARTS[1])
    subplot = fig_parts[row, col]
    subplot.add_image(face_parts[i], cmap="viridis")
    subplot.axes.visible = False
    subplot.title = f"c{i}"
fig_parts.show()


sample_indices = np.linspace(0, n_samples - 1, N_RECON_SAMPLES, dtype=int)
fig_recon = fpl.Figure(shape=(2, N_RECON_SAMPLES), size=(1000, 1000))
for col, idx in enumerate(sample_indices):
    orig_sp = fig_recon[0, col]
    orig_sp.add_image(originals[idx], cmap="gray")
    orig_sp.axes.visible = False
    orig_sp.title = f"orig {idx}"

    rec_sp = fig_recon[1, col]
    rec_sp.add_image(reconstructions[idx], cmap="gray")
    rec_sp.axes.visible = False
    rec_sp.title = f"recon {idx}"
fig_recon.show()


# sklearn NMF (solver='cd' = HALS) from the same initial factorization.
# sklearn takes X as (n_samples, n_features) so we transpose and permute the
# init accordingly: WH_sklearn_init = H0.T @ W0.T = (W0 @ H0).T = X_init.T.
print("Running sklearn NMF for comparison...")
X_sk = X.T.astype(np.float64)
model = NMF(
    n_components=R,
    init="custom",
    solver="cd",
    beta_loss="frobenius",
    max_iter=500,
    tol=1e-6,
)
t_sk = time.perf_counter()
W_sk = model.fit_transform(
    X_sk,
    W=H0.T.astype(np.float64).copy(),
    H=W0.T.astype(np.float64).copy(),
)
elapsed_sk_ms = (time.perf_counter() - t_sk) * 1000.0
H_sk = model.components_
sk_err = float(np.linalg.norm(X_sk - W_sk @ H_sk) / np.linalg.norm(X_sk))
print(f"  sklearn done in {elapsed_sk_ms:.1f} ms,  final relative Frobenius error {sk_err:.4f}")

# For sklearn's factorization X.T ≈ W_sk H_sk, the rows of H_sk are the face
# parts (analogous to columns of our HALS W).
sk_face_parts = H_sk.reshape(R, *IMG_SHAPE).astype(np.float32)

fig_parts_sk = fpl.Figure(shape=GRID_PARTS, size=(1000, 1000))
for i in range(R):
    row, col = divmod(i, GRID_PARTS[1])
    subplot = fig_parts_sk[row, col]
    subplot.add_image(sk_face_parts[i], cmap="viridis")
    subplot.axes.visible = False
    subplot.title = f"sklearn c{i}"
fig_parts_sk.show()

fpl.loop.run()
