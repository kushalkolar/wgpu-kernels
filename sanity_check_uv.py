import numpy as np
import torch
from tqdm import tqdm

import fastplotlib as fpl

from project import select_adapter, load_compression, CompressionBuffers, UVImage

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)

# path to the demixing results file, only the compression results are used
dmr_path = "./demix_new.hdf5"

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
mean_img = compression["mean_img"].flatten()
var_img = compression["var_img"].flatten()
n_frames, height, width = compression["shape"]

uv = UVImage(CompressionBuffers(dmr_path))

n_samples = 100
err = np.zeros(n_samples)
# compare against the torch reconstruction, same as PMDArray with rescale=True
for i, t in enumerate(tqdm(np.random.randint(0, n_frames, size=n_samples))):
    uv.t = t
    frame_torch = (
        (torch.sparse.mm(u, v[:, [t]]).squeeze(1) * var_img + mean_img)
        .reshape(height, width)
        .numpy()
    )

    # relative error using the Frobenius norm
    err[i] = np.linalg.norm(uv.to_numpy() - frame_torch, ord="fro") / np.linalg.norm(
        frame_torch, ord="fro"
    )

print(f"max relative error: {err.max():.3e}")
