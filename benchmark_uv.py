from pathlib import Path

from scipy import sparse
import numpy as np
import pandas as pd

import fastplotlib as fpl

from project import select_adapter, load_compression, CompressionBuffers, UVImage, SpMVImage

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)

dmr_path = "./demix_new.hdf5"

# U as dense cell tiles
uv_cells = UVImage(CompressionBuffers(dmr_path), benchmark=True)

# U as CSR, with the CSR vector kernel, for comparison
compression = load_compression(dmr_path)
u_csr_torch = compression["u"].coalesce().to_sparse_csr()
u_csr = sparse.csr_matrix(
    (u_csr_torch.values(), u_csr_torch.col_indices(), u_csr_torch.crow_indices()),
    shape=u_csr_torch.size(),
)
uv_csr = SpMVImage(
    u_csr,
    compression["v"].numpy(),
    shape=compression["shape"][1:],
    scale_factor=compression["var_img"].flatten().numpy(),
    scale_add=compression["mean_img"].flatten().numpy(),
    benchmark=True,
    spmv_mode="vector",
)

n = 5_000


def benchmark(obj: UVImage | SpMVImage, n: int) -> dict:
    obj.clear_timings()
    for i in range(n):
        obj.t = i

    timings = obj.get_timings()

    return {
        "mean": timings.mean(),
        "median": np.median(timings),
        "std": timings.std(),
        "min": timings.min(),
        "max": timings.max(),
    }


df = pd.DataFrame(
    columns=["device", "kernel", "mean", "median", "std", "min", "max"]
)

for kernel, obj in [("cell_tiles", uv_cells), ("csr_vector", uv_csr)]:
    df.loc[df.index.size] = {
        "device": adapter.info.device,
        "kernel": kernel,
        **benchmark(obj, n),
    }

print(df)

if not Path(__file__).parent.joinpath("benchmark_uv.csv").is_file():
    df.to_csv("benchmark_uv.csv", index=False)
else:
    df.to_csv("benchmark_uv.csv", index=False, header=False, mode="a")
