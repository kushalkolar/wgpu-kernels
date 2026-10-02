"""
Compare the WGSL merge test and merge (CorrelationImages.get_merge_pairs, SignalMerger.merge) against masknmf's
_compute_indices_to_merge and merge_components, with the real compression results U, V and synthetic spatial
footprints (gaussian discs), every fourth followed by a near-duplicate so that pairs merge. The temporal trace of each
footprint is the mean movie trace over it, so that its correlation image is high around it. Both use the same robust
noise term.

The pairs to merge must be the same. The areas and overlaps of the thresholded images of the candidate pairs are
compared as well, and the pixels where WGSL or masknmf and the float64 images are on different sides of the threshold
are reported with their distance to it. The merged signals must be the same: the preserved ones identical, the
supports of the merged ones equal, and their spatial and temporal components at least as close to masknmf's
rank_1_NMF_fit in float64 as masknmf's float32 ones are, within a factor of 2.
"""

import contextlib
import io

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

import fastplotlib as fpl
from masknmf.demixing.signal_demixer import (
    _compute_indices_to_merge,
    _compute_standard_correlation_image,
    merge_components,
)

from project import select_adapter, load_compression, CompressionBuffers
from project._hals import SignalBuffers, read_buffer, round_up
from project._correlation import CorrelationImages
from project._merge import SignalMerger

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
select_adapter(adapter)
torch.set_num_threads(16)

dmr_path = "./demix_new.hdf5"
merge_threshold = 0.8
overlap_threshold = 0.4
n_signals = 400

compression = load_compression(dmr_path)
u = compression["u"].coalesce()
v = compression["v"]
n_frames, height, width = compression["shape"]
rank = v.shape[0]
n_pixels = height * width

gpu_compression = CompressionBuffers(dmr_path)
correlation = CorrelationImages(gpu_compression)
noise = correlation.set_robust_noise(np.random.default_rng(0).choice(n_frames, size=5000, replace=False))

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

# the mean movie traces over the footprints, a^T U V / sum(a)
u_csr = sp.csr_matrix((u.values().numpy().astype(np.float64), u.indices().numpy()), shape=(n_pixels, rank))
a_csr = sp.csr_matrix((values, (pixels, signal_indices)), shape=(n_pixels, n_signals))
weights = np.asarray((a_csr.T @ u_csr).todense()) / np.asarray(a_csr.sum(axis=0)).T
c = (torch.from_numpy(weights.astype(np.float32)) @ v).numpy().T.copy()
signals = SignalBuffers(gpu_compression, a, c, np.zeros(n_pixels, dtype=np.float32))

pairs_wgsl = correlation.get_merge_pairs(signals, merge_threshold, overlap_threshold)
candidates, overlaps_wgsl, areas_wgsl = correlation.get_merge_candidates()

standard = _compute_standard_correlation_image(
    u, v, torch.from_numpy(c), (height, width), noise_std=torch.full((height, width), noise), frame_batch_size=1000
)
rows_masknmf, cols_masknmf = _compute_indices_to_merge(
    a, standard, merge_threshold, overlap_threshold, frame_batch_size=500
)
pairs_masknmf = np.stack([rows_masknmf.numpy(), cols_masknmf.numpy()], axis=1)

# masknmf's thresholded images, pixels in row-major order, and their areas and overlaps
images = standard.getitem_tensor(np.arange(n_signals)).reshape(n_signals, -1).numpy()
above_masknmf = images > merge_threshold
areas_masknmf = np.maximum(above_masknmf.sum(axis=1), 1)
overlaps_masknmf = np.array([np.sum(above_masknmf[i] & above_masknmf[j]) for i, j in candidates])
paired = np.unique(candidates)

# the WGSL bits in row-major pixel order
n_cells = gpu_compression.n_cells[0] * gpu_compression.n_cells[1]
cell_size = gpu_compression.cell_size
n_words = -(-(cell_size**2) // 32)
words = read_buffer(
    correlation._buffers["bits"], np.uint32, (n_signals, round_up(n_cells * n_words, 4))
)[:, : n_cells * n_words]
cell_bits = np.unpackbits(words.reshape(n_signals, n_cells, n_words).view(np.uint8), axis=-1, bitorder="little")
cell_bits = cell_bits[:, :, : cell_size**2].astype(bool)
n_cells_x = gpu_compression.n_cells[1]
cell_rows = (np.arange(n_cells) // n_cells_x)[:, None] * cell_size + np.arange(cell_size**2) // cell_size
cell_cols = (np.arange(n_cells) % n_cells_x)[:, None] * cell_size + np.arange(cell_size**2) % cell_size
above_wgsl = np.zeros((n_signals, n_pixels), dtype=bool)
above_wgsl[:, (cell_rows * width + cell_cols).ravel()] = cell_bits.reshape(n_signals, -1)

# float64 images
c64 = c.astype(np.float64)
c_centered = c64 - c64.mean(axis=0)
c_tilde = c_centered / np.linalg.norm(c_centered, axis=0)
mean64 = standard.std_corr_img_mean.numpy().astype(np.float64)
normalizer64 = standard.std_corr_img_normalizer.numpy().astype(np.float64)
images64 = (u_csr @ (v.numpy().astype(np.float64) @ c_tilde) - mean64[:, None] * c_tilde.sum(axis=0)).T
images64 /= normalizer64[None, :]
above64 = images64 > merge_threshold

print(
    f"{candidates.shape[0]} candidate pairs, to merge: {pairs_wgsl.shape[0]} (wgsl), {pairs_masknmf.shape[0]} (masknmf)"
)
print(
    f"areas differ for {np.sum(areas_wgsl[paired] != areas_masknmf[paired])} of {paired.size} signals, overlaps for "
    f"{np.sum(overlaps_wgsl != overlaps_masknmf)} of {candidates.shape[0]} pairs"
)
for name, above in (("wgsl", above_wgsl), ("masknmf", above_masknmf)):
    differ = above != above64
    distance = np.abs(images64[differ] - merge_threshold).max() if differ.any() else 0.0
    print(f"{name}: {differ.sum()} pixels on the other side of the threshold than float64, within {distance:.2e} of it")

if not np.array_equal(pairs_wgsl, pairs_masknmf):
    raise AssertionError("WGSL merge pairs differ from masknmf")

# the merge, masknmf prints each component it merges
merged, preserved = SignalMerger(gpu_compression).merge(signals, pairs_wgsl)
with contextlib.redirect_stdout(io.StringIO()):
    a_masknmf, c_masknmf, _, _ = merge_components(
        a, torch.from_numpy(c), standard, merge_threshold, overlap_threshold, frame_batch_size=500
    )
c_masknmf = c_masknmf.numpy()
a_wgsl = merged.get_a()
c_wgsl = merged.get_c()
print(f"{n_signals} signals merged into {a_wgsl.shape[1]} (wgsl), {a_masknmf.shape[1]} (masknmf)")
if a_wgsl.shape[1] != a_masknmf.shape[1]:
    raise AssertionError("WGSL merge gives a different number of signals than masknmf")


def get_columns(x: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """(pixels, values) of each column"""
    x = x.coalesce()
    x_rows, x_cols = x.indices().numpy()
    return [(x_rows[x_cols == j], x.values().numpy()[x_cols == j]) for j in range(x.shape[1])]


columns_wgsl, columns_masknmf = get_columns(a_wgsl), get_columns(a_masknmf)
n_preserved = preserved.size
preserved_equal = all(
    np.array_equal(columns_wgsl[j][0], columns_masknmf[j][0])
    and np.array_equal(columns_wgsl[j][1], columns_masknmf[j][1])
    for j in range(n_preserved)
) and np.array_equal(c_wgsl[:, :n_preserved], c_masknmf[:, :n_preserved])
print(f"preserved signals identical: {preserved_equal}")


def rank_1_fit_float64(members: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """masknmf's rank_1_NMF_fit in float64, the union of the members' pixels, s over them and t"""
    a_members = a_csr.tocsc()[:, members]
    union = np.unique(a_members.indices)
    a_members = a_members[union].toarray()
    c_members = c[:, members].T.astype(np.float64)
    s = a_members.mean(axis=1)
    mask = s > 0
    for _ in range(5):
        spatial_norm = s @ s
        t = np.maximum((a_members.T @ s) @ c_members / (spatial_norm if spatial_norm != 0 else 1.0), 0.0)
        temporal_norm = t @ t
        s = np.maximum(a_members @ (c_members @ t) / (temporal_norm if temporal_norm != 0 else 1.0) * mask, 0.0)
    return union, s, t


# the merged signals, in the order of masknmf's components
graph = nx.Graph()
graph.add_edges_from(list(zip(pairs_wgsl[:, 0], pairs_wgsl[:, 1])))
components = [np.array(list(component)) for component in nx.connected_components(graph)]
supports_differ = 0
errors = {"wgsl s": [], "masknmf s": [], "wgsl t": [], "masknmf t": []}
for g, members in enumerate(components):
    union, s64, t64 = rank_1_fit_float64(members)
    j = n_preserved + g
    if not np.array_equal(columns_wgsl[j][0], columns_masknmf[j][0]):
        supports_differ += 1
        differ = np.setxor1d(columns_wgsl[j][0], columns_masknmf[j][0])
        print(f"  merged signal {j}: supports differ at {differ}, float64 s {s64[np.searchsorted(union, differ)]}")
    for name, (column_pixels, column_values) in (("wgsl", columns_wgsl[j]), ("masknmf", columns_masknmf[j])):
        s_union = np.zeros(union.size)
        s_union[np.searchsorted(union, column_pixels)] = column_values
        errors[f"{name} s"].append(np.abs(s_union - s64).max() / np.abs(s64).max())
    errors["wgsl t"].append(np.abs(c_wgsl[:, j] - t64).max() / np.abs(t64).max())
    errors["masknmf t"].append(np.abs(c_masknmf[:, j] - t64).max() / np.abs(t64).max())
print(f"{len(components)} merged signals, supports differ for {supports_differ}")
print("errors of the merged signals relative to rank_1_NMF_fit in float64, max over the merged signals")
for name, e in errors.items():
    print(f"  {name}: {np.max(e):.2e}")

if not preserved_equal or supports_differ > 0:
    raise AssertionError("WGSL merged signals differ from masknmf")
if np.max(errors["wgsl s"]) > 2 * np.max(errors["masknmf s"]) or np.max(errors["wgsl t"]) > 2 * np.max(
    errors["masknmf t"]
):
    raise AssertionError("WGSL merged signals are less accurate than masknmf's")
print("passed")
