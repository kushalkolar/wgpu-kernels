"""
masknmf's pure superpixels (superpixel_init): the superpixels of each patch and the selection of the pure ones by
successive projection from the Gram matrix of their traces
"""

import numpy as np

from ._hals import concatenated_ranges


def get_patches(peaks: np.ndarray, fov_shape: tuple[int, int], patch_size: tuple[int, int]) -> tuple:
    """
    masknmf's patches of patch_size pixels of the superpixels at the pixels peaks (row-major, their label order), in
    row-major order as superpixel_init visits them.

    Returns (the patch of each superpixel, patch_ptr [n_patches + 1], the superpixels of each patch in label order)
    """
    height, width = fov_shape
    patch_height, patch_width = patch_size
    n_patch_cols = -(-width // patch_width)
    n_patches = -(-height // patch_height) * n_patch_cols
    patches = (peaks // width // patch_height) * n_patch_cols + (peaks % width) // patch_width
    patch_ptr = np.concatenate([[0], np.cumsum(np.bincount(patches, minlength=n_patches))])
    return patches, patch_ptr, np.argsort(patches, kind="stable")


def get_patch_pairs(patches: np.ndarray, patch_ptr: np.ndarray, members: np.ndarray) -> tuple:
    """
    The pairs of superpixels of a patch as the graph of trace_products.wgsl: for each superpixel the superpixels of its
    patch with a larger label, in label order.

    Returns (neighbor_ptr [n_superpixels + 1], neighbors)
    """
    n = patches.size
    rank = np.empty(n, dtype=np.int64)
    rank[members] = np.arange(n) - patch_ptr[patches[members]]
    counts = patch_ptr[patches + 1] - patch_ptr[patches] - 1 - rank
    neighbor_ptr = np.concatenate([[0], np.cumsum(counts)])
    return neighbor_ptr, members[concatenated_ranges(patch_ptr[patches] + rank + 1, counts)]


def successive_projection(gram: np.ndarray, l1_norms: np.ndarray, threshold: float) -> np.ndarray:
    """
    masknmf's successive_projection of the traces of the superpixels of a patch, from their Gram matrix and L1 norms,
    in float64: the traces normalized to an L1 norm of 1, then as long as the largest norm of a trace after the
    projection out of the traces selected, relative to its norm, is above threshold, the trace with it is selected, of
    traces within 1e-6 of it the one with the largest norm. A trace of L1 norm 0 makes masknmf's normalized traces nan,
    which selects nothing.

    Returns the indices of the selected traces, in the order of the selection.
    """
    k = gram.shape[0]
    if k == 0 or np.any(l1_norms == 0):
        return np.zeros(0, dtype=np.int64)
    g = gram / np.outer(l1_norms, l1_norms)
    squared_norms = np.diag(g).copy()
    squared_norms_orig = squared_norms.copy()
    norms_orig = np.sqrt(squared_norms_orig)
    # u_j^T m for the orthonormal directions u_j of the selected traces, of masknmf's Gram-Schmidt
    projections = np.zeros((k, k))
    selected = []
    while len(selected) < k and (np.sqrt(squared_norms) / norms_orig).max() > threshold:
        relative = squared_norms / squared_norms_orig
        largest = relative.max()
        position = int(np.argmax(relative))
        ties = np.flatnonzero((largest - relative) / largest <= 1e-6)
        if ties.size > 1:
            position = int(ties[np.argmax(squared_norms_orig[ties])])
        j = len(selected)
        selected.append(position)
        overlaps = projections[:j, position]
        projections[j] = (g[position] - overlaps @ projections[:j]) / np.sqrt(
            g[position, position] - overlaps @ overlaps
        )
        squared_norms = np.maximum(0.0, squared_norms - projections[j] ** 2)
    return np.array(selected, dtype=np.int64)


def select_pure_superpixels(
    patch_ptr: np.ndarray,
    members: np.ndarray,
    neighbor_ptr: np.ndarray,
    products: np.ndarray,
    self_products: np.ndarray,
    l1_norms: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    masknmf's pure superpixels: successive_projection in each patch, the Gram matrix of its traces from the products
    of trace_products.wgsl on the graph of get_patch_pairs.

    Returns the pure superpixels, sorted.
    """
    products, self_products, l1_norms = (np.asarray(x, dtype=np.float64) for x in (products, self_products, l1_norms))
    pure = [np.zeros(0, dtype=np.int64)]
    for p in range(patch_ptr.size - 1):
        patch = members[patch_ptr[p] : patch_ptr[p + 1]]
        k = patch.size
        rows, cols = np.triu_indices(k, 1)
        gram = np.diag(self_products[patch])
        gram[rows, cols] = gram[cols, rows] = products[neighbor_ptr[patch[rows]] + cols - rows - 1]
        pure.append(patch[successive_projection(gram, l1_norms[patch], threshold)])
    return np.unique(np.concatenate(pure))
