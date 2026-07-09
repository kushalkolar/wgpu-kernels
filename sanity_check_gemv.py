import numpy as np
import pandas as pd
from tqdm import tqdm

import fastplotlib as fpl

from project import GEMV


adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)


cases = pd.DataFrame(
    [
        {"name": "square-row",   "M":   4096, "N":  4096, "kernel": "row"},
        {"name": "square-auto",  "M":   4096, "N":  4096, "kernel": "auto"},
        {"name": "tall-strip",   "M": 262144, "N":    32, "kernel": "strip"},
        {"name": "tall-row",     "M": 262144, "N":    32, "kernel": "row"},
        {"name": "tall-tpr",     "M": 262144, "N":    32, "kernel": "thread_per_row"},
        {"name": "tall-auto",    "M": 262144, "N":    32, "kernel": "auto"},
        {"name": "fat-fat",      "M":    128, "N": 65536, "kernel": "fat"},
        {"name": "fat-auto",     "M":    128, "N": 65536, "kernel": "auto"},
        {"name": "small-row",    "M":   1024, "N":  1024, "kernel": "row"},
        {"name": "small-auto",   "M":   1024, "N":  1024, "kernel": "auto"},
    ]
)

N_TRIALS = 20
TOL = 1e-4  # ~2^-23 accumulated over long reductions, plus slack
rng = np.random.default_rng(seed=0)

selected = []
worst_err = []

for row in cases.itertuples(index=False):
    A = rng.standard_normal((row.M, row.N), dtype=np.float32)
    gemv = GEMV(A, np.zeros(row.N, dtype=np.float32), kernel=row.kernel)
    print(f"[{row.name}] shape=({row.M}, {row.N}) kernel={row.kernel} -> selected={gemv.kernel}")

    worst = 0.0
    for _ in tqdm(range(N_TRIALS), desc=row.name):
        v = rng.standard_normal(row.N).astype(np.float32)
        gemv.set_v(v)
        got = gemv.to_numpy()
        truth = A @ v
        worst = max(worst, float(np.linalg.norm(got - truth) / np.linalg.norm(truth)))

    selected.append(gemv.kernel)
    worst_err.append(worst)

cases = cases.assign(
    selected=selected,
    worst_err=worst_err,
    status=np.where(np.asarray(worst_err) < TOL, "OK", "FAIL"),
)

print()
print(cases.to_string(index=False))

if (cases["status"] == "FAIL").any():
    raise SystemExit(1)
