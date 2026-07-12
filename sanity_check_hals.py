"""
Sanity check for the WGPU HALS implementation.

Three checks per case:

1. NUMPY reference — iteration-for-iteration numpy HALS with the same
   fused-sweep semantics as the WGSL kernel. WGPU and numpy must produce
   essentially the same (W, H) up to float32 noise.
2. MONOTONICITY — the Frobenius error must not increase between outer
   iterations (a required property of HALS).
3. SKLEARN — sklearn's NMF with solver='cd' (its HALS path) run to
   convergence from the same init. Not exact — different scaling and
   stopping rules — so we only check the final relative Frobenius error
   is within a small factor.

Cases:
- Random small/medium synthetic X.
- Olivetti faces (4096 × 400), a canonical NMF benchmark.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_olivetti_faces

import fastplotlib as fpl

from project import HALS


adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
fpl.select_adapter(adapter)


F32_EPS = float(np.finfo(np.float32).eps)


def numpy_hals_iter(X, W, H, eps, n_inner_H, n_inner_W):
    """One HALS outer iteration in pure numpy, mirroring the WGSL fused-sweep
    semantics: precompute (P, Q) or (R, S) once per side, then n_inner
    row/column sweeps against those precomputes."""
    r = W.shape[1]

    P = (W.T @ X).astype(np.float32)
    Q = (W.T @ W).astype(np.float32)
    for _ in range(n_inner_H):
        for l in range(r):
            q_ll = Q[l, l]
            if q_ll <= 0.0:
                continue
            num = P[l, :] - Q[l, :] @ H + q_ll * H[l, :]
            H[l, :] = np.maximum(eps, num / q_ll).astype(np.float32)

    R = (X @ H.T).astype(np.float32)
    S = (H @ H.T).astype(np.float32)
    for _ in range(n_inner_W):
        for l in range(r):
            s_ll = S[l, l]
            if s_ll <= 0.0:
                continue
            num = R[:, l] - W @ S[:, l] + s_ll * W[:, l]
            W[:, l] = np.maximum(eps, num / s_ll).astype(np.float32)

    return W, H


def sklearn_err(X, W0, H0, r):
    """sklearn NMF with 'cd' (HALS) from the same init, run to convergence."""
    model = NMF(
        n_components=r,
        init="custom",
        solver="cd",
        beta_loss="frobenius",
        max_iter=500,
        tol=1e-6,
    )
    W = model.fit_transform(
        X.astype(np.float64),
        W=W0.astype(np.float64).copy(),
        H=H0.astype(np.float64).copy(),
    )
    H = model.components_
    return float(np.linalg.norm(X - W @ H) / np.linalg.norm(X))


# Load Olivetti faces once — 400 samples × 4096 features. We factor
# X = W H with X as pixels × samples (m=4096, n=400), the classical
# Lee-Seung convention where columns of W become face parts.
print("Loading Olivetti faces...")
faces = fetch_olivetti_faces()
X_faces = faces.data.astype(np.float32).T   # (4096, 400)
print(f"faces: {X_faces.shape}, min={X_faces.min():.3f}, max={X_faces.max():.3f}")


cases = pd.DataFrame(
    [
        {"name": "small",         "m":  128, "n":  64,  "r":  8, "n_outer":  1, "n_inner_H": 1, "n_inner_W": 1, "dataset": "random"},
        {"name": "small-Ainner",  "m":  128, "n":  64,  "r":  8, "n_outer":  1, "n_inner_H": 3, "n_inner_W": 3, "dataset": "random"},
        {"name": "medium",        "m":  512, "n": 256,  "r": 16, "n_outer":  5, "n_inner_H": 1, "n_inner_W": 1, "dataset": "random"},
        {"name": "medium-Ainner", "m":  512, "n": 256,  "r": 16, "n_outer":  5, "n_inner_H": 4, "n_inner_W": 4, "dataset": "random"},
        {"name": "rect",          "m": 1024, "n": 128,  "r": 16, "n_outer":  3, "n_inner_H": 2, "n_inner_W": 2, "dataset": "random"},
        {"name": "faces-r16",     "m": 4096, "n": 400,  "r": 16, "n_outer": 30, "n_inner_H": 4, "n_inner_W": 4, "dataset": "faces"},
        {"name": "faces-r25",     "m": 4096, "n": 400,  "r": 25, "n_outer": 30, "n_inner_H": 4, "n_inner_W": 4, "dataset": "faces"},
    ]
)

TOL_WH_1ITER = 1e-4    # tight W/H match required after 1 outer iter
TOL_ERR = 1e-3         # |err_gpu − err_np| absolute
TOL_SKLEARN = 0.25     # relative gap vs sklearn's converged error
rng = np.random.default_rng(seed=0)

w_rel_all, h_rel_all = [], []
err_gpu_all, err_np_all, err_sk_all = [], [], []
mono_all = []

for row in cases.itertuples(index=False):
    if row.dataset == "faces":
        X = X_faces
    else:
        X = rng.uniform(0.0, 1.0, size=(row.m, row.n)).astype(np.float32)
    W0 = (rng.uniform(0.0, 1.0, size=(row.m, row.r)) + 0.1).astype(np.float32)
    H0 = (rng.uniform(0.0, 1.0, size=(row.r, row.n)) + 0.1).astype(np.float32)

    print(f"\n[{row.name}] m={row.m}, n={row.n}, r={row.r}, "
          f"n_outer={row.n_outer}, n_inner=({row.n_inner_H}, {row.n_inner_W})")

    hals = HALS(X, W0.copy(), H0.copy(), wg_size=64)
    errs = []
    for _ in tqdm(range(row.n_outer), desc=row.name):
        hals.fit(1, row.n_inner_H, row.n_inner_W)
        errs.append(hals.error())

    W_np, H_np = W0.copy(), H0.copy()
    for _ in range(row.n_outer):
        W_np, H_np = numpy_hals_iter(X, W_np, H_np, F32_EPS, row.n_inner_H, row.n_inner_W)

    err_sklearn = sklearn_err(X, W0, H0, row.r)

    W_gpu = hals.get_W()
    H_gpu = hals.get_H()

    wr = float(np.linalg.norm(W_gpu - W_np) / max(np.linalg.norm(W_np), 1e-30))
    hr = float(np.linalg.norm(H_gpu - H_np) / max(np.linalg.norm(H_np), 1e-30))
    eg = float(errs[-1])
    en = float(np.linalg.norm(X - W_np @ H_np) / np.linalg.norm(X))
    mono_ok = all(errs[k + 1] <= errs[k] + 1e-5 for k in range(len(errs) - 1))

    print(f"  ‖W_gpu − W_np‖ / ‖W_np‖ = {wr:.3e}")
    print(f"  ‖H_gpu − H_np‖ / ‖H_np‖ = {hr:.3e}")
    print(f"  err_gpu = {eg:.4f}   err_np = {en:.4f}   err_sklearn = {err_sklearn:.4f}")
    print(f"  monotone = {mono_ok}")

    w_rel_all.append(wr)
    h_rel_all.append(hr)
    err_gpu_all.append(eg)
    err_np_all.append(en)
    err_sk_all.append(err_sklearn)
    mono_all.append(mono_ok)

w_rel_arr = np.asarray(w_rel_all)
h_rel_arr = np.asarray(h_rel_all)
err_gpu_arr = np.asarray(err_gpu_all)
err_np_arr = np.asarray(err_np_all)
err_sk_arr = np.asarray(err_sk_all)
mono_arr = np.asarray(mono_all)
n_outer_arr = cases["n_outer"].to_numpy()

# For 1-iter runs we can require tight W/H match against numpy. Once we
# iterate, tiny float32 order-of-op differences propagate through the update
# so W/H drift even though both reach the same objective — check err instead.
wh_ok = np.where(
    n_outer_arr == 1,
    (w_rel_arr < TOL_WH_1ITER) & (h_rel_arr < TOL_WH_1ITER),
    True,
)
err_diff = np.abs(err_gpu_arr - err_np_arr)
sk_gap = np.abs(err_gpu_arr - err_sk_arr) / np.maximum(err_sk_arr, 1e-30)

cases = cases.assign(
    w_rel=w_rel_arr,
    h_rel=h_rel_arr,
    err_gpu=err_gpu_arr,
    err_np=err_np_arr,
    err_sklearn=err_sk_arr,
    err_diff=err_diff,
    sk_gap=sk_gap,
    monotone=mono_arr,
    status=np.where(
        wh_ok & (err_diff < TOL_ERR) & mono_arr & (sk_gap < TOL_SKLEARN),
        "OK", "FAIL",
    ),
)

print()
print(cases.to_string(index=False))

if (cases["status"] == "FAIL").any():
    raise SystemExit(1)
