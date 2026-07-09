"""
Dense GEMV benchmark: WGSL kernels vs torch.mv on CUDA / CPU.

Both backends batch iterations to amortize per-call CPU-side overhead so the
comparison reflects GPU throughput per matvec rather than per-call latency:

- WGSL: GEMV.dispatch_batch(N_INNER) submits N_INNER dispatches inside one
  command encoder and one _poll_wait, returning per-iter ms.
- torch: N_INNER A @ v calls between two torch.cuda.synchronize() calls,
  per-iter ms = total / N_INNER.

We repeat this N_BATCHES times to get mean/std/min/max across batches.
"""

from itertools import product
from pathlib import Path
import time

import fastplotlib as fpl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from project import GEMV


adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)

DEVICE_NAME = adapter.info.device
HAS_CUDA = torch.cuda.is_available()

N_BATCHES = 10
N_INNER = 200
N_WARMUP = 50


shapes = pd.DataFrame(
    [
        {"regime": "square",  "M":   4096, "N":  4096},
        {"regime": "tall",    "M": 262144, "N":    32},
        {"regime": "fat",     "M":    128, "N": 65536},
        {"regime": "small",   "M":   1024, "N":  1024},
    ]
)


def _stats(per_iter: np.ndarray) -> dict:
    return {
        "mean":   per_iter.mean(),
        "median": np.median(per_iter),
        "std":    per_iter.std(),
        "min":    per_iter.min(),
        "max":    per_iter.max(),
    }


def bench_wgsl(A: np.ndarray, v: np.ndarray, kernel: str, **kw) -> dict:
    gemv = GEMV(A, v, kernel=kernel, benchmark=True, **kw)
    for _ in range(N_WARMUP):
        gemv.dispatch()
    per_iter = np.array([gemv.dispatch_batch(N_INNER) for _ in range(N_BATCHES)])
    return _stats(per_iter)


def bench_torch(A: np.ndarray, v: np.ndarray, cuda: bool) -> dict:
    dev = "cuda" if cuda else "cpu"
    A_t = torch.from_numpy(A).to(dev)
    v_t = torch.from_numpy(v).to(dev)
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    for _ in range(N_WARMUP):
        _ = A_t @ v_t
    sync()

    per_iter = np.zeros(N_BATCHES)
    for b in range(N_BATCHES):
        sync()
        t0 = time.perf_counter()
        for _ in range(N_INNER):
            _ = A_t @ v_t
        sync()
        per_iter[b] = (time.perf_counter() - t0) * 1000.0 / N_INNER
    return _stats(per_iter)


def fat_ok(M: int, N: int, chunk_size: int = 4096) -> bool:
    """Fat kernel needs dispatch (K_chunks, M, 1) to stay under the 65535 per-dim cap."""
    return M <= 65535


rng = np.random.default_rng(seed=0)

records = []
for shape_row in shapes.itertuples(index=False):
    M, N = shape_row.M, shape_row.N
    print(f"\n=== {shape_row.regime}: A[{M}, {N}] ===")
    A = rng.standard_normal((M, N), dtype=np.float32)
    v = rng.standard_normal(N, dtype=np.float32)

    kernels = ["row", "strip"]
    if fat_ok(M, N):
        kernels.append("fat")
    # thread_per_row keeps v in workgroup shared memory (~16 KB budget on
    # typical WGPU limits), so cap N at 2048 f32.
    if N <= 2048:
        kernels.append("thread_per_row")

    for kernel in tqdm(kernels, desc="wgsl"):
        result = bench_wgsl(A, v, kernel)
        records.append({
            "device": DEVICE_NAME,
            "backend": "wgsl",
            "kernel": kernel,
            "regime": shape_row.regime,
            "M": M,
            "N": N,
            **result,
        })

    for cuda in ([False, True] if HAS_CUDA else [False]):
        result = bench_torch(A, v, cuda=cuda)
        records.append({
            "device": DEVICE_NAME,
            "backend": f"torch-{'cuda' if cuda else 'cpu'}",
            "kernel": None,
            "regime": shape_row.regime,
            "M": M,
            "N": N,
            **result,
        })

df = pd.DataFrame(records)

# Report time as a factor vs torch-cuda per shape regime. <1 means faster than
# torch-cuda, >1 means slower. Rows with no torch-cuda baseline in their regime
# get NaN.
baseline = (
    df[df.backend == "torch-cuda"]
    .set_index("regime")["mean"]
    .rename("cuda_mean")
)
df = df.merge(baseline, left_on="regime", right_index=True, how="left")
df["factor_vs_cuda"] = df["mean"] / df["cuda_mean"]
df = df.drop(columns="cuda_mean")

print()
print(df.to_string(index=False))

out_path = Path(__file__).parent.joinpath("benchmark_gemv.csv")
if out_path.is_file():
    df.to_csv(out_path, index=False, header=False, mode="a")
else:
    df.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
