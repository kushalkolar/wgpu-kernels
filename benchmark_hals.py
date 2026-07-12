"""
NMF (HALS) benchmark: WGSL vs numpy vs cupy vs sklearn.

Each backend runs N_OUTER outer iterations from the same initial (W0, H0)
on a shape sweep. Wall-clock times and final relative Frobenius errors are
recorded. All four backends run the same algorithm (HALS / coordinate
descent on the Frobenius NMF objective); iteration semantics differ
slightly across implementations:

- WGSL, numpy, cupy: fused-sweep semantics — precompute (P, Q) or (R, S)
  once per side, then n_inner sweeps of r column updates.
- sklearn: its own solver='cd' scheduling. `max_iter` counts outer BCD
  iterations; setting `tol=0` disables early stopping.

Data conversion (float32 ↔ float64, host → device) is outside the timed
region so the times measure compute only.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import fastplotlib as fpl

from sklearn.decomposition import NMF
from sklearn.datasets import fetch_openml

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from project import HALS


adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)

DEVICE_NAME = adapter.info.device
F32_EPS = float(np.finfo(np.float32).eps)


N_OUTER = 50
N_INNER_H = 4
N_INNER_W = 4
N_REPEATS = 3


def pick_wg_size(r: int) -> int:
    """Choose the largest wg_size (power of two) that keeps hals_sweep_W's
    workgroup shared memory (2·r·(wg+1)·4 + r²·4 bytes) under 16 KB."""
    for wg in (128, 64, 32):
        if 2 * r * (wg + 1) * 4 + r * r * 4 <= 16 * 1024:
            return wg
    return 32


def numpy_hals_iter(X, W, H, eps, n_inner_H, n_inner_W):
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


def cupy_hals_iter(X, W, H, eps, n_inner_H, n_inner_W):
    """Cupy port of numpy_hals_iter. Keeps everything on the GPU; no
    per-iteration device→host syncs."""
    r = W.shape[1]
    P = W.T @ X
    Q = W.T @ W
    for _ in range(n_inner_H):
        for l in range(r):
            q_ll = Q[l, l]
            num = P[l, :] - Q[l, :] @ H + q_ll * H[l, :]
            H[l, :] = cp.maximum(eps, num / q_ll)
    R = X @ H.T
    S = H @ H.T
    for _ in range(n_inner_W):
        for l in range(r):
            s_ll = S[l, l]
            num = R[:, l] - W @ S[:, l] + s_ll * W[:, l]
            W[:, l] = cp.maximum(eps, num / s_ll)
    return W, H


def bench_wgsl(X, W0, H0, r):
    hals = HALS(X, W0.copy(), H0.copy(), wg_size=pick_wg_size(r))
    hals.fit_batched(1, N_INNER_H, N_INNER_W)  # warmup / pipeline compile
    t0 = time.perf_counter()
    hals.fit_batched(N_OUTER, N_INNER_H, N_INNER_W)
    W = hals.get_W()  # forces GPU sync via readback
    H = hals.get_H()
    elapsed = (time.perf_counter() - t0) * 1000.0
    err = float(np.linalg.norm(X - W @ H) / np.linalg.norm(X))
    return elapsed, err


def bench_numpy(X, W0, H0, r):
    W, H = W0.copy(), H0.copy()
    t0 = time.perf_counter()
    for _ in range(N_OUTER):
        W, H = numpy_hals_iter(X, W, H, F32_EPS, N_INNER_H, N_INNER_W)
    elapsed = (time.perf_counter() - t0) * 1000.0
    err = float(np.linalg.norm(X - W @ H) / np.linalg.norm(X))
    return elapsed, err


def bench_cupy(X, W0, H0, r):
    X_cp = cp.asarray(X)
    W = cp.asarray(W0)
    H = cp.asarray(H0)
    # warmup + cuBLAS kernel selection
    W, H = cupy_hals_iter(X_cp, W, H, F32_EPS, N_INNER_H, N_INNER_W)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_OUTER):
        W, H = cupy_hals_iter(X_cp, W, H, F32_EPS, N_INNER_H, N_INNER_W)
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000.0
    W_np = cp.asnumpy(W)
    H_np = cp.asnumpy(H)
    err = float(np.linalg.norm(X - W_np @ H_np) / np.linalg.norm(X))
    return elapsed, err


def bench_sklearn(X, W0, H0, r):
    X_sk = X.T.astype(np.float64)
    W_init = H0.T.astype(np.float64)
    H_init = W0.T.astype(np.float64)
    model = NMF(
        n_components=r,
        init="custom",
        solver="cd",
        beta_loss="frobenius",
        max_iter=N_OUTER,
        tol=0.0,
    )
    t0 = time.perf_counter()
    W_sk = model.fit_transform(X_sk, W=W_init.copy(), H=H_init.copy())
    H_sk = model.components_
    elapsed = (time.perf_counter() - t0) * 1000.0
    err = float(np.linalg.norm(X_sk - W_sk @ H_sk) / np.linalg.norm(X_sk))
    return elapsed, err


def repeat(fn, *args):
    times = np.zeros(N_REPEATS)
    err = 0.0
    for i in range(N_REPEATS):
        elapsed, err_i = fn(*args)
        times[i] = elapsed
        if i == 0:
            err = err_i
    return {
        "mean_ms":   times.mean(),
        "median_ms": float(np.median(times)),
        "std_ms":    times.std(),
        "min_ms":    times.min(),
        "max_ms":    times.max(),
        "err":       err,
    }


print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
# Pixels-as-features convention: X is (features, samples), columns of W
# become interpretable digit parts. fetch_openml returns train+test combined.
X_mnist = (mnist.data.astype(np.float32).T) / 255.0
mnist_m, mnist_n = X_mnist.shape
print(f"MNIST X: {X_mnist.shape}")


shapes = pd.DataFrame(
    [
        {"regime": "small", "m":     512, "n":     256, "r": 16, "dataset": "random"},
        {"regime": "mnist", "m": mnist_m, "n": mnist_n, "r": 20, "dataset": "mnist"},
        {"regime": "large", "m":    8192, "n":    1024, "r": 32, "dataset": "random"},
    ]
)

rng = np.random.default_rng(0)
records = []

for row in shapes.itertuples(index=False):
    m, n, r = row.m, row.n, row.r
    print(f"\n=== {row.regime}: m={m}, n={n}, r={r} ===")

    if row.dataset == "mnist":
        X = X_mnist
    else:
        X = rng.uniform(0.0, 1.0, size=(m, n)).astype(np.float32)
    W0 = (rng.uniform(0.0, 1.0, size=(m, r)) + 0.1).astype(np.float32)
    H0 = (rng.uniform(0.0, 1.0, size=(r, n)) + 0.1).astype(np.float32)

    backends = [
        ("wgsl",    bench_wgsl),
        ("numpy",   bench_numpy),
        ("sklearn", bench_sklearn),
    ]
    if HAS_CUPY:
        backends.append(("cupy", bench_cupy))
    else:
        print("  (cupy not installed, skipping)")

    for name, fn in backends:
        print(f"  {name}...", end=" ", flush=True)
        try:
            stats = repeat(fn, X, W0, H0, r)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(f"median={stats['median_ms']:.1f} ms   err={stats['err']:.4f}")
        records.append({
            "device": DEVICE_NAME,
            "backend": name,
            "regime": row.regime,
            "m": m, "n": n, "r": r,
            "n_outer": N_OUTER,
            "n_inner_H": N_INNER_H,
            "n_inner_W": N_INNER_W,
            **stats,
        })

df = pd.DataFrame(records)

# Report factor vs cupy per regime (or numpy if cupy unavailable).
baseline_backend = "cupy" if HAS_CUPY else "numpy"
baseline = (
    df[df.backend == baseline_backend]
    .set_index("regime")["median_ms"]
    .rename("baseline_ms")
)
df = df.merge(baseline, left_on="regime", right_index=True, how="left")
df[f"factor_vs_{baseline_backend}"] = df["median_ms"] / df["baseline_ms"]
df = df.drop(columns="baseline_ms")

print()
print(df.to_string(index=False))

out_path = Path(__file__).parent.joinpath("benchmark_hals.csv")
if out_path.is_file():
    df.to_csv(out_path, index=False, header=False, mode="a")
else:
    df.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
