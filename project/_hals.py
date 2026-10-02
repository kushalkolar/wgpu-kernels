import functools
from pathlib import Path

import numpy as np
import torch
import wgpu
import pygfx

from ._spmv import ComputeShader
from ._compression import CompressionBuffers


SHADER_DIR = Path(__file__).parent


def create_empty_buffer(device: wgpu.GPUDevice, nbytes: int) -> wgpu.GPUBuffer:
    """
    zero-initialized storage buffer that can also be read back and copied to, at least 16 bytes so empty
    arrays can be bound
    """
    return device.create_buffer(
        size=max(-(-nbytes // 16) * 16, 16),
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_DST
        | wgpu.BufferUsage.COPY_SRC,
    )


def create_buffer(device: wgpu.GPUDevice, data: np.ndarray) -> wgpu.GPUBuffer:
    """storage buffer with the given data, see ``create_empty_buffer``"""
    buffer = create_empty_buffer(device, data.nbytes)
    if data.nbytes > 0:
        device.queue.write_buffer(buffer, 0, np.ascontiguousarray(data))
    return buffer


def check_buffer_size(buffer: wgpu.GPUBuffer | None, nbytes: int) -> wgpu.GPUBuffer:
    """
    buffer if it has at least nbytes, otherwise a new empty buffer: of nbytes if buffer is None, otherwise of
    1.5 * nbytes so that a buffer whose size changes often is not reallocated every time
    """
    if buffer is not None and buffer.size >= nbytes:
        return buffer
    device = pygfx.renderers.wgpu.get_shared().device
    return create_empty_buffer(device, nbytes if buffer is None else 3 * nbytes // 2)


def read_buffer(buffer: wgpu.GPUBuffer, dtype, shape) -> np.ndarray:
    """download a buffer from the GPU"""
    device = pygfx.renderers.wgpu.get_shared().device
    n = int(np.prod(shape))
    return np.frombuffer(device.queue.read_buffer(buffer), dtype=dtype)[:n].reshape(shape)


def load_shader(
    filename: str,
    entry_point: str | None = None,
    reductions: bool = False,
    label: str | None = None,
) -> ComputeShader:
    """ComputeShader from a wgsl file in this directory, optionally with the workgroup_sum and workgroup_max helpers"""
    wgsl = SHADER_DIR.joinpath(filename).read_text()
    if reductions:
        wgsl = SHADER_DIR.joinpath("reductions.wgsl").read_text() + wgsl

    return ComputeShader(wgsl, entry_point=entry_point or Path(filename).stem, label=label)


def set_constants_and_resources(
    shader: ComputeShader,
    constants: dict[str, int | bool],
    resources: dict[int, wgpu.GPUBuffer],
):
    """the pipeline is only recreated if a constant changed, the bind group if a resource changed"""
    for name, value in constants.items():
        shader.set_constant(name, value)
    for index, resource in resources.items():
        shader.set_resource(index, resource)


def dispatch_grid(n: int, limit: int = 65535) -> tuple[int, int]:
    """2D dispatch for n workgroups, index them with wid.y * nwg.x + wid.x"""
    nx = min(n, limit)
    return nx, -(-n // nx)


def next_power_of_two(n: int) -> int:
    """smallest power of two >= n, for n >= 1"""
    return 1 << (n - 1).bit_length()


def round_up(n: int, m: int) -> int:
    """n rounded up to a multiple of m"""
    return -(-n // m) * m


@functools.cache
def get_subgroup_size() -> int | None:
    """
    subgroup size of compute shaders on the shared device, None if the device does not have the subgroups feature (see
    ``select_adapter``). With 64 (RDNA on RADV) the ring term uses gemm_nn.wgsl instead of gemm_nt.wgsl
    """
    device = pygfx.renderers.wgpu.get_shared().device
    if not {"subgroup", "subgroups"} & set(device.features):
        return None
    shader = ComputeShader(
        """
        @group(0) @binding(0) var<storage, read_write> out: array<u32>;
        @compute @workgroup_size(64)
        fn subgroup_size(@builtin(subgroup_size) size: u32) {
            out[0] = size;
        }
        """,
        entry_point="subgroup_size",
    )
    out = create_empty_buffer(device, 4)
    shader.set_resource(0, out)
    shader.dispatch(1)
    return int(read_buffer(out, np.uint32, (1,))[0])


def compute_hals_schedule(
    has_support: np.ndarray,
    neighbor_ptr: np.ndarray,
    neighbors: np.ndarray,
    frame_batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Groups of signals that are updated together, the ``blocks`` of masknmf's HALS updates.

    Same as masknmf's ``_compute_hals_schedule`` on ``mask_ab``: ``networkx.coloring.greedy_color(graph,
    strategy="largest_first")`` on the overlap graph of the masks, nodes are sorted by degree, ties in node
    order, where a self loop adds 2 to the degree of every signal with a non-empty mask. Each node gets the
    smallest color not used by its neighbors. The signals of each color, in coloring order, are split into
    groups of at most ``frame_batch_size`` signals. Groups are in color order.

    Parameters
    ----------
    has_support: np.ndarray
        [n_signals] bool, whether the mask of each signal has any entries

    neighbor_ptr, neighbors: np.ndarray
        overlap graph of the masks, see ``compute_overlap_graph``

    frame_batch_size: int, optional
        max number of signals of a group, ``None`` does not split the colors

    Returns
    -------
    group_ptr: np.ndarray
        [n_groups + 1] uint32, the signals of group g are ``group_signals[group_ptr[g]:group_ptr[g + 1]]``

    group_signals: np.ndarray
        [n_signals] uint32
    """
    n_signals = has_support.size
    if n_signals == 0:
        return np.zeros(1, dtype=np.uint32), np.zeros(0, dtype=np.uint32)

    degree = np.diff(neighbor_ptr).astype(np.int64) + 2 * has_support

    # stable sort, descending degree
    order = np.argsort(-degree, kind="stable")

    colors = np.full(n_signals, -1, dtype=np.int64)
    for u in order:
        used = set(colors[neighbors[neighbor_ptr[u] : neighbor_ptr[u + 1]]].tolist())
        color = 0
        while color in used:
            color += 1
        colors[u] = color

    # signals of each color in coloring order
    colored = colors[order]
    by_color = np.argsort(colored, kind="stable")
    group_signals = order[by_color].astype(np.uint32)
    color_ptr = np.searchsorted(colored[by_color], np.arange(colored.max() + 2))

    # split each color into groups of at most frame_batch_size signals
    group_ptr = []
    for start, end in zip(color_ptr[:-1], color_ptr[1:]):
        group_ptr.extend(range(start, end, frame_batch_size or end - start))
    group_ptr.append(n_signals)

    return np.asarray(group_ptr, dtype=np.uint32), group_signals


def concatenated_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """concatenation of arange(start, start + length) for each (start, length)"""
    starts = np.asarray(starts, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(starts, lengths) + offsets


def compute_overlap_graph(
    pixels: np.ndarray, signals: np.ndarray, n_signals: int, n_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Signals whose supports share a pixel, from the (pixel, signal) coordinates of the entries.

    Returns
    -------
    neighbor_ptr: np.ndarray
        [n_signals + 1] int64, the neighbors of signal i are ``neighbors[neighbor_ptr[i]:neighbor_ptr[i + 1]]``

    neighbors: np.ndarray
        int64, every edge in both directions, sorted by (i, j), without self edges
    """
    order = np.lexsort((signals, pixels))
    pixels_sorted = pixels[order]
    signals_sorted = signals[order]
    pixel_counts = np.bincount(pixels, minlength=n_pixels)
    pixel_ptr = np.concatenate([[0], np.cumsum(pixel_counts)])

    # pair every entry of a pixel shared by several signals with every entry of the same pixel
    entry_counts = pixel_counts[pixels_sorted]
    shared = entry_counts > 1
    i = np.repeat(signals_sorted[shared], entry_counts[shared])
    j = signals_sorted[
        concatenated_ranges(pixel_ptr[pixels_sorted[shared]], entry_counts[shared])
    ]
    keep = i != j
    edge_keys = np.unique(i[keep] * n_signals + j[keep])

    neighbor_ptr = np.searchsorted(edge_keys // n_signals, np.arange(n_signals + 1))
    return neighbor_ptr, edge_keys % n_signals


def compute_signal_structures(
    pixels: np.ndarray,
    signals: np.ndarray,
    n_signals: int,
    compression: CompressionBuffers,
    mask_pixels: np.ndarray | None = None,
    mask_signals: np.ndarray | None = None,
    frame_batch_size: int | None = None,
) -> dict[str, np.ndarray | int]:
    """
    Index structures of the spatial footprints ``a``, from the (pixel, signal) coordinates of its entries.
    Recomputed whenever the support of ``a`` changes.

    The groups of signals that are updated together are computed from the (pixel, signal) coordinates of the
    entries of ``mask_ab`` if given, otherwise from those of ``a``. masknmf's ``mask_ab`` has the support of
    ``a`` at initialization, after a support update it is smaller: the signals of a group never overlap in
    ``mask_ab`` but they can overlap in ``a``.

    Returns a dict with:

    - "order": permutation of the given entries into signal-major order
    - "a_ptr", "a_pixels": signal-major entries, sorted by (signal, pixel)
    - "pixel_ptr", "pixel_entries", "pixel_signals": pixel-major view, ``pixel_entries`` indexes the
      signal-major entries
    - "neighbor_ptr", "neighbors", "neighbor_reverse": overlap graph, signals whose supports share a pixel,
      sorted by (i, j), without self edges. ``neighbor_reverse[e]`` is the index of edge (j, i)
    - "upper_ptr", "upper_edges": edges (i, j) with j > i grouped by i
    - "pair_ptr", "pair_blocks", "pair_signals", "pair_value_offsets": (signal, block) pairs where the
      block overlaps the support of the signal, sorted by (signal, block), with the start of each pair's
      values in the per-pair buffers (one value per column of U in the block)
    - "n_pair_values"
    - "group_ptr", "group_signals": see ``compute_hals_schedule``
    - "group_overlaps": [n_groups] bool, whether any two signals of the group overlap in ``a``
    - "max_overlapping_group": number of signals of the largest group that overlaps in ``a``, 0 if none does
    - "max_signal_pairs", "max_neighbors"

    """
    height, width = compression.fov_shape
    cell_size = compression.cell_size
    n_cells_x = width // cell_size
    cell_col_ptr, cell_cols = compression.get_cell_structure()
    block_col_ptr, col_block, _ = compression.get_block_structure()
    n_blocks = block_col_ptr.size - 1

    pixels = np.asarray(pixels, dtype=np.int64)
    signals = np.asarray(signals, dtype=np.int64)
    n_pixels = height * width

    order = np.lexsort((pixels, signals))
    a_pixels = pixels[order]
    a_signals = signals[order]
    a_ptr = np.searchsorted(a_signals, np.arange(n_signals + 1))

    # pixel-major view of the signal-major entries
    pixel_order = np.lexsort((a_signals, a_pixels))
    pixel_entries = pixel_order
    pixel_signals = a_signals[pixel_order]
    pixel_ptr = np.concatenate([[0], np.cumsum(np.bincount(a_pixels, minlength=n_pixels))])

    # overlap graph of a
    neighbor_ptr, edge_j = compute_overlap_graph(a_pixels, a_signals, n_signals, n_pixels)
    edge_i = np.repeat(np.arange(n_signals), np.diff(neighbor_ptr))
    neighbor_reverse = np.searchsorted(edge_i * n_signals + edge_j, edge_j * n_signals + edge_i)

    upper = np.flatnonzero(edge_j > edge_i)
    upper_ptr = np.searchsorted(edge_i[upper], np.arange(n_signals + 1))

    # (signal, block) pairs: blocks of the columns of the cells that the signal's pixels lie in
    rows, cols = a_pixels // width, a_pixels % width
    entry_cells = (rows // cell_size) * n_cells_x + cols // cell_size
    signal_cells = np.unique(a_signals * (n_cells_x * (height // cell_size)) + entry_cells)
    sc_signals = signal_cells // (n_cells_x * (height // cell_size))
    sc_cells = signal_cells % (n_cells_x * (height // cell_size))

    cell_n_cols = np.diff(cell_col_ptr).astype(np.int64)[sc_cells]
    tile_cols = concatenated_ranges(cell_col_ptr[sc_cells], cell_n_cols)
    pair_keys = np.unique(
        np.repeat(sc_signals, cell_n_cols) * n_blocks + col_block[cell_cols[tile_cols]]
    )
    pair_signals = pair_keys // n_blocks
    pair_blocks = pair_keys % n_blocks
    pair_ptr = np.searchsorted(pair_signals, np.arange(n_signals + 1))
    pair_n_values = np.diff(block_col_ptr).astype(np.int64)[pair_blocks]
    pair_value_offsets = np.concatenate([[0], np.cumsum(pair_n_values)])

    # groups of signals that are updated together, from the overlap graph of mask_ab
    if mask_pixels is None:
        has_support = np.diff(a_ptr) > 0
        mask_neighbor_ptr, mask_neighbors = neighbor_ptr, edge_j
    else:
        mask_pixels = np.asarray(mask_pixels, dtype=np.int64)
        mask_signals = np.asarray(mask_signals, dtype=np.int64)
        has_support = np.bincount(mask_signals, minlength=n_signals) > 0
        mask_neighbor_ptr, mask_neighbors = compute_overlap_graph(
            mask_pixels, mask_signals, n_signals, n_pixels
        )
    group_ptr, group_signals = compute_hals_schedule(
        has_support, mask_neighbor_ptr, mask_neighbors, frame_batch_size
    )

    # groups whose signals overlap in a
    n_groups = group_ptr.size - 1
    group_sizes = np.diff(group_ptr).astype(np.int64)
    signal_group = np.empty(n_signals, dtype=np.int64)
    signal_group[group_signals] = np.repeat(np.arange(n_groups), group_sizes)
    same_group = signal_group[edge_i] == signal_group[edge_j]
    group_overlaps = np.zeros(n_groups, dtype=bool)
    group_overlaps[signal_group[edge_i[same_group]]] = True

    u32 = lambda x: np.asarray(x, dtype=np.uint32)

    return {
        "order": order,
        "a_ptr": u32(a_ptr),
        "a_pixels": u32(a_pixels),
        "pixel_ptr": u32(pixel_ptr),
        "pixel_entries": u32(pixel_entries),
        "pixel_signals": u32(pixel_signals),
        "neighbor_ptr": u32(neighbor_ptr),
        "neighbors": u32(edge_j),
        "neighbor_reverse": u32(neighbor_reverse),
        "upper_ptr": u32(upper_ptr),
        "upper_edges": u32(upper),
        "pair_ptr": u32(pair_ptr),
        "pair_blocks": u32(pair_blocks),
        "pair_signals": u32(pair_signals),
        "pair_value_offsets": u32(pair_value_offsets[:-1]),
        "n_pair_values": int(pair_value_offsets[-1]),
        "group_ptr": group_ptr,
        "group_signals": group_signals,
        "group_overlaps": group_overlaps,
        "max_overlapping_group": int(group_sizes[group_overlaps].max(initial=0)),
        "max_signal_pairs": int(max(np.diff(pair_ptr).max(initial=0), 1)),
        "max_neighbors": int(max(np.diff(neighbor_ptr).max(initial=0), 1)),
    }


def compute_block_superblocks(
    compression: CompressionBuffers, superblock_size: int
) -> tuple[np.ndarray, int]:
    """
    Superblock of each block of U and the number of superblocks, see ``compute_superblock_structures``. From the
    position of the block in the grid of blocks at a stride of one cell.
    """
    _, _, block_rects = compression.get_block_structure()
    superblock_rows = block_rects[:, 0].astype(np.int64) // compression.cell_size // superblock_size
    superblock_cols = block_rects[:, 1].astype(np.int64) // compression.cell_size // superblock_size
    block_superblock = superblock_rows * (superblock_cols.max() + 1) + superblock_cols
    return block_superblock, int(block_superblock.max()) + 1


def compute_superblock_structures(
    structures: dict, compression: CompressionBuffers, superblock_size: int
) -> dict[str, np.ndarray | int]:
    """
    Index structures of the temporal partial products, from the (signal, block) pairs of
    ``compute_signal_structures``.

    A superblock is a square of ``superblock_size`` x ``superblock_size`` blocks of U, the superblocks partition
    the grid of blocks, and the columns of U of a superblock are the columns of its blocks. temporal_hals_partials
    writes one partial row per (signal, superblock) pair where the signal overlaps a block of the superblock. Its
    work items are a superblock and up to 16 of its signals, with dense weights: [column of the superblock,
    16 signals], zero where a signal does not overlap the column's block.

    Returns a dict with:

    - "superblock_block_ptr", "superblock_blocks": blocks of each superblock, in the order of its columns
    - "items": [n_items, 4] (superblock, first partial row, number of partial rows, start of the weights in vec4s)
      of each work item, the partial rows of a work item are consecutive
    - "w_offsets": [n_pairs] start of the weights of each (signal, block) pair in the weights of the work items,
      the weight of column r of the block is at ``w_offsets + 16 * r``
    - "n_w": number of weights of all work items
    - "signal_partial_ptr", "signal_partials": partial rows of each signal
    - "n_partial_rows"

    """
    block_col_ptr, _, _ = compression.get_block_structure()
    n_block_cols = np.diff(block_col_ptr).astype(np.int64)
    n_blocks = n_block_cols.size
    n_signals = structures["pair_ptr"].size - 1
    pair_signals = structures["pair_signals"].astype(np.int64)
    pair_blocks = structures["pair_blocks"].astype(np.int64)
    block_superblock, n_superblocks = compute_block_superblocks(compression, superblock_size)

    # blocks of each superblock, and where the columns of each block start in the columns of its superblock
    superblock_blocks = np.argsort(block_superblock, kind="stable")
    superblock_block_ptr = np.searchsorted(
        block_superblock[superblock_blocks], np.arange(n_superblocks + 1)
    )
    n_superblock_cols = np.bincount(
        block_superblock, weights=n_block_cols, minlength=n_superblocks
    ).astype(np.int64)
    sorted_col_starts = np.cumsum(n_block_cols[superblock_blocks]) - n_block_cols[superblock_blocks]
    superblock_col_starts = np.cumsum(n_superblock_cols) - n_superblock_cols
    block_col_offsets = np.empty(n_blocks, dtype=np.int64)
    block_col_offsets[superblock_blocks] = (
        sorted_col_starts - superblock_col_starts[block_superblock[superblock_blocks]]
    )

    # partial rows: (signal, superblock) pairs sorted by (superblock, signal)
    partial_keys, pair_partial_rows = np.unique(
        block_superblock[pair_blocks] * n_signals + pair_signals, return_inverse=True
    )
    partial_superblocks = partial_keys // n_signals
    partial_signals = partial_keys % n_signals

    # work items of each superblock: its partial rows in groups of 16
    superblock_row_ptr = np.searchsorted(partial_superblocks, np.arange(n_superblocks + 1))
    n_superblock_items = -(-np.diff(superblock_row_ptr) // 16)
    item_superblocks = np.repeat(np.arange(n_superblocks), n_superblock_items)
    item_first_rows = (
        superblock_row_ptr[item_superblocks]
        + concatenated_ranges(np.zeros(n_superblocks), n_superblock_items) * 16
    )
    item_n_rows = np.minimum(16, superblock_row_ptr[item_superblocks + 1] - item_first_rows)
    item_n_w = 16 * n_superblock_cols[item_superblocks]
    item_w_starts = np.cumsum(item_n_w) - item_n_w

    # weights of each (signal, block) pair: work item and slot of its partial row, column of its block
    row_items = np.repeat(np.arange(item_superblocks.size), item_n_rows)
    pair_items = row_items[pair_partial_rows]
    w_offsets = (
        item_w_starts[pair_items]
        + 16 * block_col_offsets[pair_blocks]
        + pair_partial_rows
        - item_first_rows[pair_items]
    )

    signal_partials = np.argsort(partial_signals, kind="stable")
    signal_partial_ptr = np.searchsorted(
        partial_signals[signal_partials], np.arange(n_signals + 1)
    )

    u32 = lambda x: np.asarray(x, dtype=np.uint32)

    return {
        "superblock_block_ptr": u32(superblock_block_ptr),
        "superblock_blocks": u32(superblock_blocks),
        "items": u32(
            np.stack([item_superblocks, item_first_rows, item_n_rows, item_w_starts // 4], axis=1)
        ),
        "w_offsets": u32(w_offsets),
        "n_w": int(item_n_w.sum()),
        "signal_partial_ptr": u32(signal_partial_ptr),
        "signal_partials": u32(signal_partials),
        "n_partial_rows": int(partial_keys.size),
    }


def compute_diff_structures(
    structures: dict, compression: CompressionBuffers
) -> dict[str, np.ndarray | int]:
    """
    Work items of spatial_hals_diff, from the (signal, block) pairs of ``compute_signal_structures``.

    A work item is a superblock of 2 x 2 blocks, see ``compute_superblock_structures``, and up to 32 of the signals
    that overlap its blocks. The rows of V of a work item are the columns of U of the blocks that overlap one of its
    signals, block after block in block order.

    Returns a dict with:

    - "items": [n_items, 12] (first column of U of each of up to 4 blocks, first row of each block in the rows of
      the work item, number of rows, number of signals, 2 zeros) of each work item, unused blocks start at the
      number of rows
    - "item_signals": [n_items, 32] signals of each work item, 0 in unused slots
    - "item_pair_offsets": [n_items, 32, 4] start of the values in diff of the (signal, block) pair of each signal
      and block of each work item, 0xFFFFFFFF where the signal does not overlap the block
    - "max_superblock_signals": max number of signals that overlap the blocks of a superblock, a superblock with
      more than 32 has several work items

    """
    block_col_ptr, _, _ = compression.get_block_structure()
    n_block_cols = np.diff(block_col_ptr).astype(np.int64)
    n_blocks = n_block_cols.size
    n_signals = structures["pair_ptr"].size - 1
    pair_signals = structures["pair_signals"].astype(np.int64)
    pair_blocks = structures["pair_blocks"].astype(np.int64)
    block_superblock, n_superblocks = compute_block_superblocks(compression, 2)

    # signals of each superblock: (superblock, signal) keys sorted by (superblock, signal), in work items of up to 32
    keys, pair_keys = np.unique(
        block_superblock[pair_blocks] * n_signals + pair_signals, return_inverse=True
    )
    superblock_key_ptr = np.searchsorted(keys // n_signals, np.arange(n_superblocks + 1))
    n_superblock_items = -(-np.diff(superblock_key_ptr) // 32)
    item_superblocks = np.repeat(np.arange(n_superblocks), n_superblock_items)
    item_first_keys = (
        superblock_key_ptr[item_superblocks]
        + concatenated_ranges(np.zeros(n_superblocks), n_superblock_items) * 32
    )
    item_n_signals = np.minimum(32, superblock_key_ptr[item_superblocks + 1] - item_first_keys)
    n_items = item_superblocks.size
    key_items = np.repeat(np.arange(n_items), item_n_signals)
    key_slots = np.arange(keys.size) - item_first_keys[key_items]

    # blocks of each work item: the blocks that overlap one of its signals, in block order
    pair_items = key_items[pair_keys]
    item_block_keys, pair_item_blocks = np.unique(
        pair_items * n_blocks + pair_blocks, return_inverse=True
    )
    ib_items = item_block_keys // n_blocks
    ib_blocks = item_block_keys % n_blocks
    item_block_ptr = np.searchsorted(ib_items, np.arange(n_items + 1))
    if np.diff(item_block_ptr).max(initial=0) > 4:
        raise ValueError(
            "spatial_hals_diff supports work items of up to 4 blocks, the blocks of U must start at different cells"
        )
    ib_slots = np.arange(ib_items.size) - item_block_ptr[ib_items]

    # first row of each block in the rows of its work item
    ib_n_rows = n_block_cols[ib_blocks]
    ib_row_starts = np.cumsum(ib_n_rows) - ib_n_rows
    ib_row_starts = ib_row_starts - ib_row_starts[item_block_ptr[ib_items]]
    item_n_rows = np.bincount(ib_items, weights=ib_n_rows, minlength=n_items).astype(np.int64)

    items = np.zeros((n_items, 12), dtype=np.int64)
    items[ib_items, ib_slots] = block_col_ptr[ib_blocks]
    items[:, 4:8] = item_n_rows[:, None]
    items[ib_items, 4 + ib_slots] = ib_row_starts
    items[:, 8] = item_n_rows
    items[:, 9] = item_n_signals

    item_signals = np.zeros((n_items, 32), dtype=np.int64)
    item_signals[key_items, key_slots] = keys % n_signals

    item_pair_offsets = np.full((n_items, 32, 4), 0xFFFFFFFF, dtype=np.int64)
    item_pair_offsets[pair_items, key_slots[pair_keys], ib_slots[pair_item_blocks]] = structures[
        "pair_value_offsets"
    ]

    u32 = lambda x: np.asarray(x, dtype=np.uint32)

    return {
        "items": u32(items),
        "item_signals": u32(item_signals),
        "item_pair_offsets": u32(item_pair_offsets),
        "max_superblock_signals": int(np.diff(superblock_key_ptr).max(initial=0)),
    }


class SignalBuffers:
    """
    Spatial footprints ``a``, temporal traces ``c``, static baseline ``b`` and the factorized ring term on
    the GPU, with the index structures of ``a``.

    Parameters
    ----------
    compression: CompressionBuffers

    a: torch.Tensor
        sparse COO tensor [n_pixels, n_signals], entries (including explicit zeros) define the support

    c: np.ndarray | torch.Tensor
        [n_frames, n_signals]

    b: np.ndarray | torch.Tensor
        [n_pixels] or [n_pixels, 1]

    factorized_ring_term: tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor] | None
        (q0 [rank, ring_rank], q1 [ring_rank, n_frames]), the fluctuating background is U q0 q1

    mask_ab: torch.Tensor, optional
        sparse COO tensor [n_pixels, n_signals], masknmf's ``mask_ab``, its entries define the groups of
        signals that are updated together. Defaults to the support of ``a``

    frame_batch_size: int, optional
        masknmf's ``frame_batch_size``, the max number of signals of a group. ``None`` does not split
        the groups

    """

    def __init__(
        self,
        compression: CompressionBuffers,
        a: torch.Tensor,
        c: np.ndarray | torch.Tensor,
        b: np.ndarray | torch.Tensor,
        factorized_ring_term: tuple | None = None,
        mask_ab: torch.Tensor | None = None,
        frame_batch_size: int | None = None,
    ):
        device = pygfx.renderers.wgpu.get_shared().device

        a = a.coalesce()
        pixels, signals = a.indices().cpu().numpy()
        values = a.values().cpu().numpy().astype(np.float32)
        mask_pixels, mask_signals = (
            (None, None) if mask_ab is None else mask_ab.coalesce().indices().cpu().numpy()
        )
        self._set_structures(compression, pixels, signals, a.shape[1], mask_pixels, mask_signals, frame_batch_size)
        self._buffers["a_values"] = create_buffer(device, values[self._structures["order"]])

        n_frames, n_frames_padded = compression.n_frames, compression.n_frames_padded
        c = np.asarray(c, dtype=np.float32)
        c_padded = np.zeros((self.n_signals, n_frames_padded), dtype=np.float32)
        c_padded[:, :n_frames] = c.T
        self._buffers["temporal_demixed"] = create_buffer(device, c_padded)
        self._buffers["b"] = create_buffer(
            device, np.asarray(b, dtype=np.float32).reshape(-1)
        )

        self.set_factorized_ring_term(factorized_ring_term)

    @classmethod
    def from_buffers(
        cls,
        compression: CompressionBuffers,
        pixels: np.ndarray,
        signals: np.ndarray,
        n_signals: int,
        a_values: wgpu.GPUBuffer,
        temporal_demixed: wgpu.GPUBuffer,
        b: wgpu.GPUBuffer,
        mask_pixels: np.ndarray | None = None,
        mask_signals: np.ndarray | None = None,
        frame_batch_size: int | None = None,
    ) -> "SignalBuffers":
        """
        Signals from the (pixel, signal) coordinates of the entries of ``a``, sorted by (signal, pixel), and its
        values, ``c`` and ``b`` on the GPU as ``buffers`` holds them: a_values in the order of the entries,
        temporal_demixed [n_signals, n_frames_padded] and b [n_pixels]. The groups are computed from the (pixel,
        signal) coordinates (mask_pixels, mask_signals) of the entries of masknmf's ``mask_ab`` if given, otherwise
        from those of ``a``. Without a ring term, see ``set_ring_term_buffers``.
        """
        signal_buffers = cls.__new__(cls)
        signal_buffers._set_structures(
            compression, pixels, signals, n_signals, mask_pixels, mask_signals, frame_batch_size
        )
        if not np.array_equal(signal_buffers._structures["order"], np.arange(len(pixels))):
            raise ValueError("the entries must be sorted by (signal, pixel), the order of a_values")
        signal_buffers._buffers["a_values"] = a_values
        signal_buffers._buffers["temporal_demixed"] = temporal_demixed
        signal_buffers._buffers["b"] = b
        signal_buffers.set_factorized_ring_term(None)
        return signal_buffers

    def _set_structures(
        self,
        compression: CompressionBuffers,
        pixels: np.ndarray,
        signals: np.ndarray,
        n_signals: int,
        mask_pixels: np.ndarray | None,
        mask_signals: np.ndarray | None,
        frame_batch_size: int | None,
    ):
        """the index structures of a (see compute_signal_structures) on the host and in buffers"""
        device = pygfx.renderers.wgpu.get_shared().device
        self._compression = compression
        self._n_signals = n_signals
        # the (pixel, signal) coordinates of the entries of mask_ab, None for the support of a
        self._mask_entries = (
            None
            if mask_pixels is None
            else (np.asarray(mask_pixels, dtype=np.int64), np.asarray(mask_signals, dtype=np.int64))
        )
        self._structures = compute_signal_structures(
            pixels,
            signals,
            n_signals,
            compression,
            mask_pixels=mask_pixels,
            mask_signals=mask_signals,
            frame_batch_size=frame_batch_size,
        )
        self._buffers = {
            name: create_buffer(device, self._structures[name])
            for name in (
                "a_ptr",
                "a_pixels",
                "pixel_ptr",
                "pixel_entries",
                "pixel_signals",
                "neighbor_ptr",
                "neighbors",
                "neighbor_reverse",
                "upper_ptr",
                "upper_edges",
                "pair_ptr",
                "pair_blocks",
                "pair_signals",
                "pair_value_offsets",
                "group_signals",
            )
        }

    def set_factorized_ring_term(self, factorized_ring_term: tuple | None):
        """
        (q0 [rank, ring_rank], q1 [ring_rank, n_frames]) or None. Stored with the ring rank padded to a
        multiple of 4, as q0, q1 and q1^T, in new buffers: a HALS that uses these signals needs
        ``HALS.set_signals`` afterwards.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        compression = self._compression

        if factorized_ring_term is None:
            self._ring_rank = 0
            for name in ("ring_left", "ring_right", "ring_right_t"):
                self._buffers[name] = create_empty_buffer(device, 0)
            return

        q0, q1 = (np.asarray(q, dtype=np.float32) for q in factorized_ring_term)
        ring_rank = q0.shape[1]
        self._ring_rank = ring_rank
        rrp = self.ring_rank_padded

        q0_padded = np.zeros((compression.rank, rrp), dtype=np.float32)
        q0_padded[:, :ring_rank] = q0
        q1_padded = np.zeros((rrp, compression.n_frames_padded), dtype=np.float32)
        q1_padded[:ring_rank, : compression.n_frames] = q1

        self._buffers["ring_left"] = create_buffer(device, q0_padded)
        self._buffers["ring_right"] = create_buffer(device, q1_padded)
        self._buffers["ring_right_t"] = create_buffer(
            device, np.ascontiguousarray(q1_padded.T)
        )

    def set_ring_term_buffers(
        self,
        ring_left: wgpu.GPUBuffer,
        ring_right: wgpu.GPUBuffer,
        ring_right_t: wgpu.GPUBuffer,
        ring_rank: int,
    ):
        """
        The factorized ring term of rank ring_rank in GPU buffers: q0 [rank, ring_rank_padded], q1
        [ring_rank_padded, n_frames_padded] and q1^T, as ``set_factorized_ring_term`` stores them. A HALS that
        uses these signals needs ``HALS.set_signals`` afterwards if the buffers or ``ring_rank_padded`` changed.
        """
        self._buffers["ring_left"] = ring_left
        self._buffers["ring_right"] = ring_right
        self._buffers["ring_right_t"] = ring_right_t
        self._ring_rank = ring_rank

    def reset_mask(self):
        """
        masknmf's mask_ab = a.bool() with the groups unchanged, as after a merge test without merges: merge_components
        returns a.bool(), the blocks are recomputed only after merges. A later ``index_select`` computes the groups from
        the support of a.
        """
        self._mask_entries = None

    def index_select(self, indices: np.ndarray, frame_batch_size: int | None = None) -> "SignalBuffers":
        """
        The signals ``indices`` (sorted), as masknmf's index_select of a, mask_ab and c: their entries of a and of
        mask_ab and their traces in new buffers, with the same b and ring term buffers. The groups are computed from
        the selected entries of mask_ab, or of a without one.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        indices = np.asarray(indices, dtype=np.int64)
        a_ptr = self._structures["a_ptr"].astype(np.int64)
        counts = np.diff(a_ptr)[indices]
        pixels = self._structures["a_pixels"][concatenated_ranges(a_ptr[indices], counts)].astype(np.int64)
        entry_signals = np.repeat(np.arange(indices.size), counts)

        mask_pixels, mask_signals = None, None
        if self._mask_entries is not None:
            new_indices = np.full(self._n_signals, -1, dtype=np.int64)
            new_indices[indices] = np.arange(indices.size)
            mask_pixels, mask_signals = self._mask_entries
            selected = new_indices[mask_signals] >= 0
            mask_pixels, mask_signals = mask_pixels[selected], new_indices[mask_signals[selected]]

        a_values = create_empty_buffer(device, 4 * pixels.size)
        temporal_demixed = create_empty_buffer(device, 4 * indices.size * self._compression.n_frames_padded)
        encoder = device.create_command_encoder()
        self.encode_copy(encoder, indices, a_values, temporal_demixed)
        device.queue.submit([encoder.finish()])

        signal_buffers = SignalBuffers.from_buffers(
            self._compression,
            pixels,
            entry_signals,
            indices.size,
            a_values,
            temporal_demixed,
            self._buffers["b"],
            mask_pixels,
            mask_signals,
            frame_batch_size,
        )
        signal_buffers.set_ring_term_buffers(
            self._buffers["ring_left"],
            self._buffers["ring_right"],
            self._buffers["ring_right_t"],
            self._ring_rank,
        )
        return signal_buffers

    def encode_copy(
        self,
        encoder: wgpu.GPUCommandEncoder,
        indices: np.ndarray,
        a_values: wgpu.GPUBuffer,
        temporal_demixed: wgpu.GPUBuffer,
    ):
        """copy the entries of a and the traces of the signals ``indices`` (sorted) to the start of a_values and
        temporal_demixed, in runs of consecutive signals"""
        a_ptr = self._structures["a_ptr"].astype(np.int64)
        row_size = 4 * self._compression.n_frames_padded
        runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if indices.size else []
        entry = 0
        row = 0
        for run in runs:
            first, last = run[0], run[-1] + 1
            n_entries = a_ptr[last] - a_ptr[first]
            if n_entries > 0:
                encoder.copy_buffer_to_buffer(
                    self._buffers["a_values"], 4 * a_ptr[first], a_values, 4 * entry, 4 * n_entries
                )
            encoder.copy_buffer_to_buffer(
                self._buffers["temporal_demixed"],
                row_size * first,
                temporal_demixed,
                row_size * row,
                row_size * run.size,
            )
            entry += n_entries
            row += run.size

    @property
    def n_signals(self) -> int:
        return self._n_signals

    @property
    def ring_rank(self) -> int:
        """rank of the factorized ring term, 0 if there is no ring term"""
        return self._ring_rank

    @property
    def ring_rank_padded(self) -> int:
        """rank of the factorized ring term padded to a multiple of 4, 0 if there is no ring term"""
        return round_up(self._ring_rank, 4)

    @property
    def structures(self) -> dict:
        """host index structures, see ``compute_signal_structures``"""
        return self._structures

    @property
    def buffers(self) -> dict[str, wgpu.GPUBuffer]:
        return self._buffers

    def get_a(self) -> torch.Tensor:
        """spatial footprints as a sparse COO tensor [n_pixels, n_signals]"""
        s = self._structures
        height, width = self._compression.fov_shape
        values = read_buffer(self._buffers["a_values"], np.float32, (s["a_pixels"].size,))
        signals = np.repeat(np.arange(self.n_signals), np.diff(s["a_ptr"]))
        return torch.sparse_coo_tensor(
            np.stack([s["a_pixels"].astype(np.int64), signals]),
            values.copy(),
            (height * width, self.n_signals),
        ).coalesce()

    def get_c(self) -> np.ndarray:
        """temporal traces [n_frames, n_signals]"""
        c = read_buffer(
            self._buffers["temporal_demixed"],
            np.float32,
            (self.n_signals, self._compression.n_frames_padded),
        )
        return np.ascontiguousarray(c[:, : self._compression.n_frames].T)

    def get_b(self) -> np.ndarray:
        """static baseline [n_pixels]"""
        height, width = self._compression.fov_shape
        return read_buffer(self._buffers["b"], np.float32, (height * width,)).copy()

    def get_factorized_ring_term(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(q0 [rank, ring_rank], q1 [ring_rank, n_frames]), None if there is no ring term"""
        if self._ring_rank == 0:
            return None
        compression = self._compression
        rrp = self.ring_rank_padded
        q0 = read_buffer(self._buffers["ring_left"], np.float32, (compression.rank, rrp))
        q1 = read_buffer(self._buffers["ring_right"], np.float32, (rrp, compression.n_frames_padded))
        return q0[:, : self._ring_rank].copy(), q1[: self._ring_rank, : compression.n_frames].copy()


class GEMM:
    """
    out = a @ b^T with gemm_nt.wgsl, a is [m, k] and b is [n, k], both row-major. Split-K is chosen to reach
    ``target_workgroups``, or ``n_splits`` if given (shorter sums in each split, for accuracy), gemm_nt_reduce sums the
    partial products of the splits.
    """

    def __init__(self, label: str, target_workgroups: int = 512, n_splits: int | None = None):
        device = pygfx.renderers.wgpu.get_shared().device
        self._target_workgroups = target_workgroups
        self._n_splits = n_splits

        self._gemm = load_shader("gemm_nt.wgsl", entry_point="gemm_nt", label=f"gemm_nt {label}")
        self._reduce = load_shader(
            "gemm_nt.wgsl", entry_point="gemm_nt_reduce", label=f"gemm_nt_reduce {label}"
        )

        self._params = device.create_buffer(
            size=48, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._gemm.set_resource(4, self._params)
        self._reduce.set_resource(4, self._params)

        # partial products of the splits
        self._partial = None
        self._gemm_grid = None
        self._reduce_grid = None

    def set_arguments(
        self,
        a: wgpu.GPUBuffer,
        b: wgpu.GPUBuffer,
        out: wgpu.GPUBuffer,
        m: int,
        n: int,
        k: int,
    ):
        """n and k must be multiples of 4"""
        device = pygfx.renderers.wgpu.get_shared().device

        tiles = -(-m // 128) * -(-n // 128)
        n_splits = self._n_splits if self._n_splits is not None else self._target_workgroups // tiles
        n_splits = int(min(max(1, n_splits), max(1, k // 256)))
        k_per_split = -(-(-(-k // n_splits)) // 32) * 32
        n_splits = -(-k // k_per_split)

        # consecutive workgroups use the same tile of the larger operand, so that each of its tiles is read from
        # memory once instead of once per workgroup
        m_fastest = n > m

        # M, N, K, lda, ldb, ldc, k_per_split, n_splits, m_fastest
        params = np.array(
            [m, n, k, k, k, n, k_per_split, n_splits, m_fastest], dtype=np.uint32
        )
        device.queue.write_buffer(self._params, 0, params)

        self._gemm.set_resource(0, a)
        self._gemm.set_resource(1, b)
        m_tiles, n_tiles = -(-m // 128), -(-n // 128)
        self._gemm_grid = (
            (m_tiles, n_tiles, n_splits) if m_fastest else (n_tiles, m_tiles, n_splits)
        )

        if n_splits == 1:
            self._gemm.set_resource(2, out)
            self._reduce_grid = None
            return

        self._partial = check_buffer_size(self._partial, 4 * n_splits * m * n)
        self._gemm.set_resource(2, self._partial)
        self._reduce.set_resource(2, self._partial)
        self._reduce.set_resource(3, out)
        self._reduce_grid = (*dispatch_grid(-(-(m * n // 4) // 256)), 1)

    def encode(self, command_encoder: wgpu.GPUCommandEncoder):
        self._gemm.encode(command_encoder, *self._gemm_grid)
        if self._reduce_grid is not None:
            self._reduce.encode(command_encoder, *self._reduce_grid)


class GEMMNN:
    """
    out = a @ b with gemm_nn.wgsl, a is [m, k] and b is [k, n], both row-major. Only for subgroups of 64, see
    ``get_subgroup_size``
    """

    def __init__(self, label: str):
        device = pygfx.renderers.wgpu.get_shared().device
        self._gemm = load_shader("gemm_nn.wgsl", label=f"gemm_nn {label}")
        self._params = device.create_buffer(
            size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        self._gemm.set_resource(3, self._params)
        self._grid = None

    def set_arguments(
        self,
        a: wgpu.GPUBuffer,
        b: wgpu.GPUBuffer,
        out: wgpu.GPUBuffer,
        m: int,
        n: int,
        k: int,
    ):
        """n and k must be multiples of 4"""
        device = pygfx.renderers.wgpu.get_shared().device
        # M, N, K
        device.queue.write_buffer(self._params, 0, np.array([m, n, k, 0], dtype=np.uint32))
        self._gemm.set_resource(0, a)
        self._gemm.set_resource(1, b)
        self._gemm.set_resource(2, out)
        # tiles of 32 rows and 512 columns
        self._grid = (-(-m // 32), -(-n // 512), 1)

    def encode(self, command_encoder: wgpu.GPUCommandEncoder):
        self._gemm.encode(command_encoder, *self._grid)


class GPUComputation:
    """
    Base of the computations of several kernels (FluctuatingBaseline, CorrelationImages): their shaders, GEMMs and
    buffers by name, created when first used, the command encoders they record into, and the small kernels they share.
    Subclasses that time the compute passes override ``_new_encoder`` and ``_submit`` (see benchmark_background.py).
    """

    def __init__(self):
        self._shaders = {}
        self._gemms = {}
        self._buffers = {}

    def _shader(self, name: str, key=None, reductions: bool = False) -> ComputeShader:
        """one shader object per kernel and key, so that sets of constants that alternate (e.g. for the sketch and
        the rank of the background) do not recreate each other's pipelines"""
        if (name, key) not in self._shaders:
            self._shaders[(name, key)] = load_shader(f"{name}.wgsl", reductions=reductions)
        return self._shaders[(name, key)]

    def _gemm(self, name: str, target_workgroups: int = 512, n_splits: int | None = None) -> GEMM:
        if name not in self._gemms:
            self._gemms[name] = GEMM(name, target_workgroups, n_splits)
        return self._gemms[name]

    def _buffer(self, name: str, nbytes: int) -> wgpu.GPUBuffer:
        self._buffers[name] = check_buffer_size(self._buffers.get(name), nbytes)
        return self._buffers[name]

    def _encode_axpy(
        self,
        encoder: wgpu.GPUCommandEncoder,
        key: str,
        x: wgpu.GPUBuffer,
        y: wgpu.GPUBuffer,
        out: wgpu.GPUBuffer,
        n: int,
        y_stride: int,
        alpha: float = -1.0,
    ):
        """out = x + alpha y[::y_stride] over n entries, see axpy.wgsl"""
        shader = self._shader("axpy", key)
        set_constants_and_resources(shader, {"alpha": alpha}, {0: x, 1: y, 2: out})
        shader.set_uniform(3, np.array([n, y_stride, 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(-(-n // 256)))

    def _encode_transpose(
        self,
        encoder: wgpu.GPUCommandEncoder,
        key: str,
        src: wgpu.GPUBuffer,
        n_rows: int,
        n_cols: int,
        dst: wgpu.GPUBuffer,
        dst_stride: int,
    ):
        """dst [n_cols, dst_stride] = src^T for src [n_rows, n_cols], see transpose.wgsl"""
        shader = self._shader("transpose", key)
        set_constants_and_resources(
            shader, {"n_rows": n_rows, "n_cols": n_cols, "dst_stride": dst_stride}, {0: src, 1: dst}
        )
        shader.encode(encoder, -(-n_cols // 16), -(-n_rows // 16))

    def _encode_row_sums(
        self,
        encoder: wgpu.GPUCommandEncoder,
        key: str,
        x: wgpu.GPUBuffer,
        stride: int,
        first: int,
        last: int,
        out: wgpu.GPUBuffer,
        n_rows: int,
    ):
        """out [n_rows] = sums of the rows of x [n_rows, stride] over the columns first:last, see row_sums.wgsl"""
        shader = self._shader("row_sums", key, reductions=True)
        set_constants_and_resources(
            shader, {"x_stride4": stride // 4, "first": first, "last": last}, {0: x, 1: out}
        )
        shader.encode(encoder, n_rows)

    def _new_encoder(self) -> wgpu.GPUCommandEncoder:
        return pygfx.renderers.wgpu.get_shared().device.create_command_encoder()

    def _submit(self, encoder: wgpu.GPUCommandEncoder):
        pygfx.renderers.wgpu.get_shared().device.queue.submit([encoder.finish()])


class HALS:
    """
    Spatial and temporal HALS updates of masknmf's localNMF on the GPU

    Parameters
    ----------
    compression: CompressionBuffers

    signals: SignalBuffers

    tuning: dict, optional
        pipeline constants of the kernels, see ``HALS.default_tuning``

    """

    default_tuning = {
        # tuned for the Radeon 780M with tune_hals.py
        "partials_wg_size": 128,
        # side of the superblocks in blocks of U
        "partials_superblock_size": 4,
        # frames per chunk staged through workgroup memory, in vec4s, a power of two
        "diff_chunk4": 8,
        "temporal_hals_wg_size": 64,
    }

    def __init__(
        self,
        compression: CompressionBuffers,
        signals: SignalBuffers,
        tuning: dict | None = None,
    ):
        self._compression = compression
        self._tuning = {**self.default_tuning, **(tuning or {})}

        if compression.max_block_cols > 32:
            raise ValueError(
                f"spatial_hals_diff supports blocks of up to 32 columns of U, got: {compression.max_block_cols}"
            )
        # spatial_hals_diff stages the rows of V of up to 4 blocks and 32 rows of c, each invocation at most 8 vec4s
        chunk4 = self._tuning["diff_chunk4"]
        if -(-4 * compression.max_block_cols * chunk4 // 256) + -(-32 * chunk4 // 256) > 8:
            raise ValueError(
                f"spatial_hals_diff stages at most 8 vec4s per invocation, diff_chunk4 is too large: "
                f"{self._tuning['diff_chunk4']}"
            )

        # sizes of the workgroup arrays, see set_signals
        self._workgroup_array_sizes = {"max_signal_pairs": 1, "max_neighbors": 1}

        # temporary buffers that are reused by set_signals when they are large enough
        self._temp = {"ring_term": None, "c_group": None, "partial": None}

        self._temporal_gram = load_shader("temporal_gram.wgsl", reductions=True)
        self._spatial_hals_diff = load_shader("spatial_hals_diff.wgsl")
        self._spatial_hals = load_shader("spatial_hals.wgsl")
        self._temporal_hals_projection = load_shader(
            "temporal_hals_projection.wgsl", reductions=True
        )
        self._temporal_hals_partials = load_shader("temporal_hals_partials.wgsl")
        self._temporal_hals_shaders = {
            c_nonneg: load_shader("temporal_hals.wgsl") for c_nonneg in (True, False)
        }
        # writes the new rows of c of a group whose signals overlap in a into c
        self._temporal_hals_copy = load_shader(
            "temporal_hals.wgsl", entry_point="temporal_hals_copy"
        )
        # ring_c = c q1^T, reduction over frames, split-K
        self._ring_c_gemm = GEMM("ring_c")
        # ring term of the cumulator = ring_w q1, read by temporal_hals: gemm_nn with q1 for subgroups of 64,
        # otherwise gemm_nt with q1^T
        if get_subgroup_size() == 64:
            self._ring_term_gemm, self._ring_term_b = GEMMNN("ring_term"), "ring_right"
        else:
            self._ring_term_gemm, self._ring_term_b = GEMM("ring_term"), "ring_right_t"

        self.set_signals(signals)

    def set_signals(self, signals: SignalBuffers):
        """
        Use new signals, e.g. after the support of ``a`` changed, or the same signals after their factorized
        ring term was set. Pipelines are only recreated when a workgroup array has to grow.
        """
        self._signals = signals
        compression = self._compression
        tuning = self._tuning
        device = pygfx.renderers.wgpu.get_shared().device

        s = signals.structures
        buf = signals.buffers
        n_signals = signals.n_signals
        rrp = signals.ring_rank_padded
        n_frames_padded = compression.n_frames_padded
        tiles = compression.spatial_compressed

        # index structures of the temporal partial products
        self._superblock_structures = compute_superblock_structures(
            s, compression, tuning["partials_superblock_size"]
        )
        sb = self._superblock_structures
        sb_buf = {
            name: create_buffer(device, sb[name])
            for name in (
                "superblock_block_ptr",
                "superblock_blocks",
                "items",
                "w_offsets",
                "signal_partial_ptr",
                "signal_partials",
            )
        }

        # work items of spatial_hals_diff
        self._diff_structures = compute_diff_structures(s, compression)
        diff_buf = {
            name: create_buffer(device, self._diff_structures[name])
            for name in ("items", "item_signals", "item_pair_offsets")
        }

        self._temp.update(
            {
                "c_sum": create_empty_buffer(device, 4 * n_signals),
                "c_sq": create_empty_buffer(device, 4 * n_signals),
                "gram": create_empty_buffer(device, 4 * s["neighbors"].size),
                "ata": create_empty_buffer(device, 4 * s["neighbors"].size),
                "atb": create_empty_buffer(device, 4 * n_signals),
                "a_sq": create_empty_buffer(device, 4 * n_signals),
                "diff": create_empty_buffer(device, 4 * s["n_pair_values"]),
                # weights of the work items of temporal_hals_partials, a new buffer so that the weights
                # where a signal does not overlap a block are zero
                "w": create_empty_buffer(device, 4 * sb["n_w"]),
                "ring_c": create_empty_buffer(device, 4 * n_signals * max(rrp, 1)),
                "ring_w": create_empty_buffer(device, 4 * n_signals * max(rrp, 1)),
                # values of a before the group that is updated, read by spatial_hals
                "a_before": create_empty_buffer(device, 4 * s["a_pixels"].size),
            }
        )

        # the buffers with a row per signal and frame are only reallocated when they have to grow
        self._temp["ring_term"] = check_buffer_size(
            self._temp["ring_term"], 4 * n_signals * n_frames_padded
        )
        self._temp["partial"] = check_buffer_size(
            self._temp["partial"], 4 * sb["n_partial_rows"] * n_frames_padded
        )
        # new rows of c of a group whose signals overlap in a, written back after the whole group
        self._temp["c_group"] = check_buffer_size(
            self._temp["c_group"], 4 * s["max_overlapping_group"] * n_frames_padded
        )

        # uniform with a row per dispatch, so that an update is recorded without writing to buffers:
        # (start in group_signals, number of signals, whether the signals overlap in a) of each group
        group_ptr = s["group_ptr"].astype(np.int64)
        groups = np.zeros((group_ptr.size - 1, 4), dtype=np.uint32)
        groups[:, 0] = group_ptr[:-1]
        groups[:, 1] = np.diff(group_ptr)
        groups[:, 2] = s["group_overlaps"]

        # max_signal_pairs and max_neighbors set the sizes of workgroup arrays. They are
        # rounded up to a power of two and never decreased, so that the pipelines are only recreated when one
        # of them grows past a power of two
        sizes = self._workgroup_array_sizes
        for name in sizes:
            sizes[name] = max(sizes[name], next_power_of_two(s[name]))

        set_constants_and_resources(
            self._temporal_gram,
            {"n_frames_padded": n_frames_padded},
            {
                0: buf["temporal_demixed"],
                1: buf["upper_ptr"],
                2: buf["upper_edges"],
                3: buf["neighbors"],
                4: buf["neighbor_reverse"],
                5: self._temp["c_sum"],
                6: self._temp["c_sq"],
                7: self._temp["gram"],
            },
        )

        set_constants_and_resources(
            self._spatial_hals_diff,
            {
                "n_frames_padded": n_frames_padded,
                "max_rows": 4 * compression.max_block_cols,
                "chunk4": tuning["diff_chunk4"],
            },
            {
                0: compression.temporal_compressed,
                1: buf["temporal_demixed"],
                2: diff_buf["items"],
                3: diff_buf["item_signals"],
                4: diff_buf["item_pair_offsets"],
                5: buf["ring_left"],
                6: self._temp["ring_c"],
                7: self._temp["c_sq"],
                8: self._temp["diff"],
            },
        )
        self._spatial_hals_diff.set_uniform(9, np.array([rrp], dtype=np.uint32))

        set_constants_and_resources(
            self._spatial_hals,
            {
                "cell_size": compression.cell_size,
                "fov_width": compression.fov_shape[1],
                "max_signal_pairs": sizes["max_signal_pairs"],
                "max_neighbors": sizes["max_neighbors"],
            },
            {
                1: buf["group_signals"],
                2: buf["a_ptr"],
                3: buf["a_pixels"],
                4: buf["a_values"],
                5: buf["pixel_ptr"],
                6: buf["pixel_entries"],
                7: buf["pixel_signals"],
                8: tiles.cell_col_ptr,
                9: tiles.cell_cols,
                10: tiles.tiles,
                11: compression.col_block,
                12: compression.block_col_ptr,
                13: buf["pair_ptr"],
                14: buf["pair_blocks"],
                15: buf["pair_value_offsets"],
                16: self._temp["diff"],
                17: buf["neighbor_ptr"],
                18: buf["neighbors"],
                19: self._temp["gram"],
                20: self._temp["c_sum"],
                21: self._temp["c_sq"],
                22: buf["b"],
                23: self._temp["a_before"],
            },
        )
        self._spatial_hals.set_uniform(0, groups)

        set_constants_and_resources(
            self._temporal_hals_projection,
            {
                "cell_size": compression.cell_size,
                "fov_width": compression.fov_shape[1],
                "max_signal_pairs": sizes["max_signal_pairs"],
                "max_block_cols": compression.max_block_cols,
            },
            {
                0: buf["a_ptr"],
                1: buf["a_pixels"],
                2: buf["a_values"],
                3: buf["pixel_ptr"],
                4: buf["pixel_entries"],
                5: buf["pixel_signals"],
                6: tiles.cell_col_ptr,
                7: tiles.cell_cols,
                8: tiles.tiles,
                9: compression.block_col_ptr,
                10: compression.block_rects,
                11: buf["pair_ptr"],
                12: buf["pair_blocks"],
                13: sb_buf["w_offsets"],
                14: self._temp["w"],
                15: buf["ring_left"],
                16: self._temp["ring_w"],
                17: buf["b"],
                18: self._temp["atb"],
                19: self._temp["a_sq"],
                20: buf["neighbor_ptr"],
                21: buf["neighbors"],
                22: self._temp["ata"],
            },
        )
        self._temporal_hals_projection.set_uniform(23, np.array([rrp], dtype=np.uint32))

        set_constants_and_resources(
            self._temporal_hals_partials,
            {
                "n_frames_padded": n_frames_padded,
                "wg_size": tuning["partials_wg_size"],
            },
            {
                0: sb_buf["items"],
                1: compression.temporal_compressed,
                2: compression.block_col_ptr,
                3: sb_buf["superblock_block_ptr"],
                4: sb_buf["superblock_blocks"],
                5: self._temp["w"],
                6: self._temp["partial"],
            },
        )

        for c_nonneg, shader in self._temporal_hals_shaders.items():
            set_constants_and_resources(
                shader,
                {
                    "n_frames": compression.n_frames,
                    "n_frames_padded": n_frames_padded,
                    "c_nonneg": c_nonneg,
                    "wg_size": tuning["temporal_hals_wg_size"],
                },
                {
                    1: buf["group_signals"],
                    2: buf["temporal_demixed"],
                    3: self._temp["ring_term"],
                    4: buf["neighbor_ptr"],
                    5: buf["neighbors"],
                    6: self._temp["ata"],
                    7: self._temp["a_sq"],
                    8: self._temp["c_group"],
                    9: self._temp["partial"],
                    10: sb_buf["signal_partial_ptr"],
                    11: sb_buf["signal_partials"],
                    12: self._temp["atb"],
                },
            )
            shader.set_uniform(0, groups)
            shader.set_uniform(13, np.array([rrp], dtype=np.uint32))

        set_constants_and_resources(
            self._temporal_hals_copy,
            {
                "n_frames": compression.n_frames,
                "n_frames_padded": n_frames_padded,
                "c_nonneg": True,
                "wg_size": tuning["temporal_hals_wg_size"],
            },
            {
                1: buf["group_signals"],
                2: buf["temporal_demixed"],
                8: self._temp["c_group"],
            },
        )
        self._temporal_hals_copy.set_uniform(0, groups)

        if rrp > 0:
            self._ring_c_gemm.set_arguments(
                buf["temporal_demixed"],
                buf["ring_right"],
                self._temp["ring_c"],
                m=n_signals,
                n=rrp,
                k=n_frames_padded,
            )
            self._ring_term_gemm.set_arguments(
                self._temp["ring_w"],
                buf[self._ring_term_b],
                self._temp["ring_term"],
                m=n_signals,
                n=n_frames_padded,
                k=rrp,
            )

    @property
    def superblock_structures(self) -> dict:
        """host index structures of the temporal partial products, see ``compute_superblock_structures``"""
        return self._superblock_structures

    @property
    def diff_structures(self) -> dict:
        """host index structures of spatial_hals_diff, see ``compute_diff_structures``"""
        return self._diff_structures

    def _groups(self):
        """(start in group_signals, number of signals, whether the signals overlap in a) of each group"""
        s = self._signals.structures
        for g in range(s["group_ptr"].size - 1):
            start, end = int(s["group_ptr"][g]), int(s["group_ptr"][g + 1])
            yield start, end - start, bool(s["group_overlaps"][g])

    def spatial_update(self, command_encoder: wgpu.GPUCommandEncoder | None = None):
        """
        masknmf ``spatial_update_hals``, updates the values of ``a`` on its support. Recorded into
        ``command_encoder`` if given, otherwise submitted.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        encoder = device.create_command_encoder() if command_encoder is None else command_encoder
        a_values = self._signals.buffers["a_values"]
        n_signals = self._signals.n_signals

        self._temporal_gram.encode(encoder, *dispatch_grid(n_signals))

        if self._signals.ring_rank_padded > 0:
            self._ring_c_gemm.encode(encoder)

        n_items = self._diff_structures["items"].shape[0]
        if n_items > 0:
            self._spatial_hals_diff.encode(encoder, *dispatch_grid(n_items))

        # like masknmf, every signal of a group is updated from the values of a before the group: if the
        # signals of a group overlap in a, spatial_hals reads them from a copy made before the group
        for group, (_, count, overlaps) in enumerate(self._groups()):
            if overlaps:
                encoder.copy_buffer_to_buffer(a_values, 0, self._temp["a_before"], 0, a_values.size)
            self._spatial_hals.encode(encoder, count, uniform_entry=group)

        if command_encoder is None:
            device.queue.submit([encoder.finish()])

    def temporal_update(
        self, c_nonneg: bool = True, command_encoder: wgpu.GPUCommandEncoder | None = None
    ):
        """
        masknmf ``temporal_update_hals``, updates ``c``. Recorded into ``command_encoder`` if given, otherwise
        submitted.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        encoder = device.create_command_encoder() if command_encoder is None else command_encoder
        compression = self._compression
        n_signals = self._signals.n_signals
        n4 = compression.n_frames_padded // 4

        self._temporal_hals_projection.encode(encoder, *dispatch_grid(n_signals))

        if self._signals.ring_rank_padded > 0:
            self._ring_term_gemm.encode(encoder)

        n_items = self._superblock_structures["items"].shape[0]
        if n_items > 0:
            self._temporal_hals_partials.encode(
                encoder, n_items, -(-n4 // self._tuning["partials_wg_size"])
            )

        # like masknmf, every signal of a group is updated from the values of c before the group: if the
        # signals of a group overlap in a their new rows are written to c_group and copied into c afterwards
        shader = self._temporal_hals_shaders[c_nonneg]
        n_workgroups_frames = -(-n4 // self._tuning["temporal_hals_wg_size"])
        for group, (_, count, overlaps) in enumerate(self._groups()):
            shader.encode(encoder, n_workgroups_frames, count, uniform_entry=group)
            if overlaps:
                self._temporal_hals_copy.encode(
                    encoder, n_workgroups_frames, count, uniform_entry=group
                )

        if command_encoder is None:
            device.queue.submit([encoder.finish()])
