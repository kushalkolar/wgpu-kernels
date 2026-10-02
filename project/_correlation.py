"""
masknmf's correlation images on the GPU, see CorrelationImages
"""

import numpy as np
import pygfx
import wgpu
from scipy.ndimage import maximum_filter1d

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    SignalBuffers,
    set_constants_and_resources,
    create_buffer,
    create_empty_buffer,
    read_buffer,
    dispatch_grid,
    compute_overlap_graph,
    concatenated_ranges,
    round_up,
)


def sample_noise_frames(n_frames: int, n_samples: int = 5000) -> np.ndarray:
    """the frames of masknmf's robust noise term: all frames if there are at most n_samples, else np.random.choice"""
    if n_frames <= n_samples:
        return np.arange(n_frames)
    return np.random.choice(n_frames, size=n_samples, replace=False)


def dilate_supports(
    pixels: np.ndarray, signals: np.ndarray, n_signals: int, fov_shape: tuple[int, int], radius: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    masknmf's sparse_dilation_routine: for each signal the pixels within the (2 radius + 1) x (2 radius + 1) box around
    the pixels of its support, clipped to the fov, as (pixel, signal) coordinates without duplicates, sorted by (signal,
    pixel), and whether each is a pixel of the support. With separable max filters on the bounding boxes of the
    signals plus the radius, stacked in one array per size rounded up to a multiple of 16.
    """
    height, width = fov_shape
    pixels = np.asarray(pixels, dtype=np.int64)
    signals = np.asarray(signals, dtype=np.int64)
    rows, cols = pixels // width, pixels % width
    first_row = np.full(n_signals, height, dtype=np.int64)
    first_col = np.full(n_signals, width, dtype=np.int64)
    end_row = np.zeros(n_signals, dtype=np.int64)
    end_col = np.zeros(n_signals, dtype=np.int64)
    np.minimum.at(first_row, signals, rows)
    np.minimum.at(first_col, signals, cols)
    np.maximum.at(end_row, signals, rows + 1)
    np.maximum.at(end_col, signals, cols + 1)
    box_heights = round_up(np.maximum(end_row - first_row, 0) + 2 * radius, 16)
    box_widths = round_up(np.maximum(end_col - first_col, 0) + 2 * radius, 16)

    # the signals of each box size and their entries, in signal order
    _, signal_sizes = np.unique(box_heights * (width + 2 * radius + 16) + box_widths, return_inverse=True)
    n_sizes = int(signal_sizes.max(initial=-1)) + 1
    size_signals = np.argsort(signal_sizes, kind="stable")
    size_signal_ptr = np.concatenate([[0], np.cumsum(np.bincount(signal_sizes, minlength=n_sizes))])
    size_entries = np.argsort(signal_sizes[signals], kind="stable")
    size_entry_ptr = np.concatenate([[0], np.cumsum(np.bincount(signal_sizes[signals], minlength=n_sizes))])

    # the dilated entries of each size: sorted by signal, and by pixel within a signal (the row-major box order)
    box_indices = np.empty(n_signals, dtype=np.int64)
    counts = np.zeros(n_signals, dtype=np.int64)
    dilated = []
    for size in range(n_sizes):
        members = size_signals[size_signal_ptr[size] : size_signal_ptr[size + 1]]
        box_indices[members] = np.arange(members.size)
        entries = size_entries[size_entry_ptr[size] : size_entry_ptr[size + 1]]
        entry_signals = signals[entries]
        support = np.zeros((members.size, box_heights[members[0]], box_widths[members[0]]), dtype=np.uint8)
        support[
            box_indices[entry_signals],
            rows[entries] - first_row[entry_signals] + radius,
            cols[entries] - first_col[entry_signals] + radius,
        ] = 1
        boxes = maximum_filter1d(support, 2 * radius + 1, axis=1, mode="constant")
        boxes = maximum_filter1d(boxes, 2 * radius + 1, axis=2, mode="constant")
        box, box_rows, box_cols = np.nonzero(boxes)
        on_support = support[box, box_rows, box_cols] > 0
        s = members[box]
        dilated_rows = box_rows + first_row[s] - radius
        dilated_cols = box_cols + first_col[s] - radius
        inside = (dilated_rows >= 0) & (dilated_rows < height) & (dilated_cols >= 0) & (dilated_cols < width)
        s = s[inside]
        dilated.append((s, dilated_rows[inside] * width + dilated_cols[inside], on_support[inside]))
        counts += np.bincount(s, minlength=n_signals)

    # placed in the order of all signals: an entry's place is the start of its signal plus its distance to the first
    # entry of its signal in its size
    signal_ptr = np.concatenate([[0], np.cumsum(counts)])
    dilated_pixels = np.empty(signal_ptr[-1], dtype=np.int64)
    dilated_on_support = np.empty(signal_ptr[-1], dtype=bool)
    for s, size_pixels, size_on_support in dilated:
        is_first = np.diff(s, prepend=-1) != 0
        places = signal_ptr[s] + np.arange(s.size) - np.flatnonzero(is_first)[np.cumsum(is_first) - 1]
        dilated_pixels[places] = size_pixels
        dilated_on_support[places] = size_on_support
    return dilated_pixels, np.repeat(np.arange(n_signals), counts), dilated_on_support


def get_candidate_pairs(
    pixels: np.ndarray, signals: np.ndarray, n_signals: int, fov_shape: tuple[int, int], radius: int = 5
) -> np.ndarray:
    """[n_pairs, 2] int64, the pairs (i, j), i < j, of signals whose supports dilated by the box overlap, the candidates
    of masknmf's merge test (_compute_indices_to_merge), sorted by (i, j)"""
    dilated_pixels, dilated_signals, _ = dilate_supports(pixels, signals, n_signals, fov_shape, radius)
    neighbor_ptr, neighbors = compute_overlap_graph(
        dilated_pixels, dilated_signals, n_signals, fov_shape[0] * fov_shape[1]
    )
    i = np.repeat(np.arange(n_signals), np.diff(neighbor_ptr))
    upper = neighbors > i
    return np.stack([i[upper], neighbors[upper]], axis=1).astype(np.int64)


def compute_dilated_overlap_graph(
    dilated_pixels: np.ndarray, dilated_signals: np.ndarray, structures: dict, n_signals: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each signal i the signals s != i with an entry of a at a pixel of the dilated support of i, from the (pixel,
    signal) coordinates of the dilated entries and the pixel-major view of a (see compute_signal_structures). Returns
    neighbor_ptr and neighbors as compute_overlap_graph does.
    """
    pixel_ptr = structures["pixel_ptr"].astype(np.int64)
    counts = np.diff(pixel_ptr)[dilated_pixels]
    i = np.repeat(dilated_signals, counts)
    s = structures["pixel_signals"][concatenated_ranges(pixel_ptr[dilated_pixels], counts)].astype(np.int64)
    other = i != s
    edge_keys = np.unique(i[other] * n_signals + s[other])
    neighbor_ptr = np.concatenate([[0], np.cumsum(np.bincount(edge_keys // n_signals, minlength=n_signals))])
    return neighbor_ptr, edge_keys % n_signals


class CorrelationImages(GPUComputation):
    """
    The parts of masknmf's correlation images that depend on the compression and the ring term only.

    Per dataset: the Gram matrix of the rows of V for the columns of U of each cell (see compute_cell_tiles), and the
    row sums of V. diag(U V V^T U^T) and U V 1 are quadratic and linear forms of the rows of U with them.

    Per pass (a DemixingState in masknmf): the robust noise term and with it the mean and normalizer of the standard
    correlation images (``set_robust_noise``), and the norms of the rows of U (V - q0 q1) with the ring term at the
    start of the pass (``set_uv_norms``), part of the normalizer of the residual correlation images. At the end of a
    pass: the mean and normalizer of the background movie U q0 q1 (``set_background_images``).

    Parameters
    ----------
    compression: CompressionBuffers
    """

    def __init__(self, compression: CompressionBuffers):
        super().__init__()
        device = pygfx.renderers.wgpu.get_shared().device
        self._compression = compression
        if compression.max_cell_cols > 96:
            raise ValueError(
                f"cell_gram.wgsl stages at most 96 columns of U per cell, got: {compression.max_cell_cols}"
            )

        # one n_pad x n_pad matrix per cell, n_pad = n_cols rounded up to a multiple of 4
        cell_col_ptr, _ = compression.get_cell_structure()
        n_pad = round_up(np.diff(cell_col_ptr).astype(np.int64), 4)
        self._gram_size = int(np.sum(n_pad**2))
        self._gram_ptr = create_buffer(device, np.concatenate([[0], np.cumsum(n_pad**2)[:-1]]).astype(np.uint32))

        n_pixels = compression.fov_shape[0] * compression.fov_shape[1]
        self._n_pixels = n_pixels
        self._std_corr_img_mean = create_empty_buffer(device, 4 * n_pixels)
        self._std_corr_img_normalizer = create_empty_buffer(device, 4 * n_pixels)
        self._uv_norms = create_empty_buffer(device, 4 * n_pixels)
        self._resid_corr_img_mean = create_empty_buffer(device, 4 * n_pixels)
        self._resid_corr_img_normalizer = create_empty_buffer(device, 4 * n_pixels)
        self._bkgd_corr_img_mean = create_empty_buffer(device, 4 * n_pixels)
        self._bkgd_corr_img_normalizer = create_empty_buffer(device, 4 * n_pixels)
        self._resid_corr_img_support_values = None
        # (max of the support values, brightness) per signal, see residual_support_values.wgsl
        self._signal_maxima = None
        self._residual_signals = None
        # c~, stats and W' of the last update_residual, for expand_masks
        self._residual_buffers = None
        # bound to the outputs that an epilogue of cell_quadratic_form does not write
        self._unused = create_empty_buffer(device, 16)
        self._robust_noise = None
        self._merge_candidates = None
        # W = V c~ of the signals of the last get_merge_pairs or update_residual and its columns, in one of two buffers
        self._w = None
        self._w_turn = 0

        # the Gram matrices of the rows of V and the row sums
        self._gram = create_empty_buffer(device, 4 * self._gram_size)
        self._v_sums = create_empty_buffer(device, 4 * compression.rank)
        encoder = self._new_encoder()
        v = compression.temporal_compressed
        n_frames_padded = compression.n_frames_padded
        self._encode_cell_gram(encoder, "v", v, v, n_frames_padded, n_frames_padded, self._gram, symmetric=True)
        self._encode_row_sums(encoder, "v", v, n_frames_padded, 0, compression.n_frames, self._v_sums, compression.rank)
        self._submit(encoder)

    def _encode_cell_gram(
        self,
        encoder: wgpu.GPUCommandEncoder,
        key: str,
        p: wgpu.GPUBuffer,
        q: wgpu.GPUBuffer,
        n_k: int,
        k_stride: int,
        out: wgpu.GPUBuffer,
        symmetric: bool,
        accumulate: bool = False,
        alpha: float = 1.0,
    ):
        """per cell out_c (+)= alpha * P_c Q_c^T over the first n_k columns of the rows, see cell_gram.wgsl"""
        comp = self._compression
        tiles = comp.spatial_compressed
        shader = self._shader("cell_gram", key)
        set_constants_and_resources(
            shader,
            {
                "max_cell_cols": comp.max_cell_cols,
                "n_k": n_k,
                "k_stride4": k_stride // 4,
                "symmetric": symmetric,
                "accumulate": accumulate,
                "alpha": alpha,
            },
            {0: tiles.cell_col_ptr, 1: tiles.cell_cols, 2: self._gram_ptr, 3: p, 4: q, 5: out},
        )
        # workgroups of 256 blocks per cell, for the cell with the most blocks
        nb = -(-comp.max_cell_cols // 4)
        max_blocks = nb * (nb + 1) // 2 if symmetric else nb * nb
        shader.encode(encoder, comp.n_cells[0] * comp.n_cells[1], -(-max_blocks // 256))

    def _encode_quadratic_form(
        self,
        encoder: wgpu.GPUCommandEncoder,
        key: str,
        gram: wgpu.GPUBuffer,
        sums: wgpu.GPUBuffer,
        epilogue: int,
        n_sum: int,
        out0: wgpu.GPUBuffer,
        out1: wgpu.GPUBuffer,
        noise_term: float = 0.0,
    ):
        """quadratic and linear forms of the rows of U with the matrices of the cells and a vector, see
        cell_quadratic_form.wgsl"""
        comp = self._compression
        tiles = comp.spatial_compressed
        shader = self._shader("cell_quadratic_form", key)
        set_constants_and_resources(
            shader,
            {
                "cell_size": comp.cell_size,
                "n_cells_x": comp.n_cells[1],
                "n_cells_y": comp.n_cells[0],
                "max_cell_cols": comp.max_cell_cols,
                "epilogue": epilogue,
                "n_sum": float(n_sum),
            },
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: self._gram_ptr,
                4: gram,
                5: sums,
                6: out0,
                7: out1,
            },
        )
        shader.set_uniform(8, np.array([noise_term, 0.0, 0.0, 0.0], dtype=np.float32))
        shader.encode(encoder, comp.n_cells[0] * comp.n_cells[1])

    def set_robust_noise(self, frames: np.ndarray, frame_batch_size: int = 5000) -> float:
        """
        masknmf's robust noise term (DemixingState._sketch_robust_variance_term), one value for all pixels: the 0.1
        quantile over the pixels of the std of U V over the given frames (see ``sample_noise_frames``), where masknmf
        takes the mean over the last batch of frame_batch_size frames only. Then the mean and normalizer of the standard
        correlation images, which include it (_compute_standard_correlation_image).

        Returns the robust noise term.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        frames = np.asarray(frames, dtype=np.int64)
        n = frames.size
        n4 = round_up(n, 4)
        # masknmf's last frame batch, columns last:n of the sampled frames in its order
        last = (-(-n // frame_batch_size) - 1) * frame_batch_size

        encoder = self._new_encoder()
        sampled = self._buffer("sampled", 4 * comp.rank * n4)
        shader = self._shader("gather_columns", "sampled")
        set_constants_and_resources(
            shader,
            {"n_rows": comp.rank, "x_stride": comp.n_frames_padded, "n_columns": n, "n_out_cols": n4},
            {0: comp.temporal_compressed, 1: create_buffer(device, frames.astype(np.uint32)), 2: sampled},
        )
        shader.encode(encoder, -(-n4 // 256), comp.rank)
        sampled_gram = self._buffer("sampled_gram", 4 * self._gram_size)
        self._encode_cell_gram(encoder, "sampled", sampled, sampled, n4, n4, sampled_gram, symmetric=True)
        last_sums = self._buffer("last_sums", 4 * comp.rank)
        self._encode_row_sums(encoder, "last", sampled, n4, last, n, last_sums, comp.rank)
        noise_std = self._buffer("noise_std", 4 * self._n_pixels)
        self._encode_quadratic_form(encoder, "noise_std", sampled_gram, last_sums, 1, n, noise_std, self._unused)
        self._submit(encoder)

        self._robust_noise = float(np.quantile(read_buffer(noise_std, np.float32, (self._n_pixels,)), 0.1))

        encoder = self._new_encoder()
        noise_term = np.float32(comp.n_frames) * np.float32(self._robust_noise) ** 2
        self._encode_quadratic_form(
            encoder,
            "standard",
            self._gram,
            self._v_sums,
            0,
            comp.n_frames,
            self._std_corr_img_normalizer,
            self._std_corr_img_mean,
            float(noise_term),
        )
        self._submit(encoder)
        return self._robust_noise

    def set_uv_norms(self, signals: SignalBuffers):
        """
        The norms of the rows of U (V - q0 q1) with the ring term of signals: masknmf's uv_norms of
        _compute_residual_correlation_image, which a pass computes once with the ring term at its start and reuses after
        the ring term changes (DemixingState.precompute_quantities resets it). From the Gram matrices of the cells:
        (V - q0 q1)(V - q0 q1)^T = V V^T + q0 (q0 H)^T - 2 q0 X^T on the rows that the quadratic forms read, with
        X = V q1^T and H = q1 q1^T.
        """
        comp = self._compression
        rrp = signals.ring_rank_padded
        encoder = self._new_encoder()
        gram = self._gram
        if rrp > 0:
            q0, q1 = signals.buffers["ring_left"], signals.buffers["ring_right"]
            # X sums all frames and the correction cancels much of V V^T, so more splits and shorter sums in each: for
            # a ring rank of 40 the default target gives 5 splits of 3904 frames, with twice masknmf's error of uv_norms
            # (sanity_check_correlation.py)
            x = self._buffer("x", 4 * comp.rank * rrp)
            g = self._gemm("x", target_workgroups=2048)
            g.set_arguments(comp.temporal_compressed, q1, x, comp.rank, rrp, comp.n_frames_padded)
            g.encode(encoder)
            h = self._buffer("h", 4 * rrp * rrp)
            g = self._gemm("h")
            g.set_arguments(q1, q1, h, rrp, rrp, comp.n_frames_padded)
            g.encode(encoder)
            # q0 H, as q0 H^T: H is exactly symmetric, both entries are the same products in the same order
            y = self._buffer("y", 4 * comp.rank * rrp)
            g = self._gemm("y")
            g.set_arguments(q0, h, y, comp.rank, rrp, rrp)
            g.encode(encoder)

            gram = self._buffer("ring_gram", 4 * self._gram_size)
            encoder.copy_buffer_to_buffer(self._gram, 0, gram, 0, 4 * self._gram_size)
            for key, operand, alpha in (("q0_h", y, 1.0), ("x", x, -2.0)):
                self._encode_cell_gram(
                    encoder, key, q0, operand, rrp, rrp, gram, symmetric=False, accumulate=True, alpha=alpha
                )
        self._encode_quadratic_form(encoder, "uv_norms", gram, self._v_sums, 2, 1, self._uv_norms, self._unused)
        self._submit(encoder)

    def set_background_images(self, signals: SignalBuffers):
        """
        masknmf's background-to-signal correlation images at the end of a pass (_compute_standard_correlation_image of
        U q0 q1 without the noise term) with the ring term of signals: the mean and normalizer of the background movie
        U q0 q1 at each pixel. From the Gram matrices of the cells: (q0 q1)(q0 q1)^T = q0 (q0 H)^T with H = q1 q1^T.
        Without a ring term both are 0.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        rrp = signals.ring_rank_padded
        encoder = self._new_encoder()
        if rrp == 0:
            encoder.clear_buffer(self._bkgd_corr_img_mean)
            encoder.clear_buffer(self._bkgd_corr_img_normalizer)
            self._submit(encoder)
            return
        q0, q1 = signals.buffers["ring_left"], signals.buffers["ring_right"]
        h = self._buffer("h", 4 * rrp * rrp)
        g = self._gemm("h")
        g.set_arguments(q1, q1, h, rrp, rrp, comp.n_frames_padded)
        g.encode(encoder)
        # q0 H, as q0 H^T: H is exactly symmetric, both entries are the same products in the same order
        y = self._buffer("y", 4 * comp.rank * rrp)
        g = self._gemm("y")
        g.set_arguments(q0, h, y, comp.rank, rrp, rrp)
        g.encode(encoder)
        gram = self._buffer("background_gram", 4 * self._gram_size)
        self._encode_cell_gram(encoder, "background", q0, y, rrp, rrp, gram, symmetric=False)

        # q0 q1 1: q1 1 in the first row of a [4, rrp] matrix, q0 q1 1 in the first column of [rank, 4]
        q1_sums = self._buffer("q1_sums", 16 * rrp)
        encoder.clear_buffer(q1_sums)
        self._encode_row_sums(encoder, "q1", q1, comp.n_frames_padded, 0, comp.n_frames, q1_sums, n_rows=rrp)
        ring_v = self._buffer("ring_v", 16 * comp.rank)
        g = self._gemm("ring_v")
        g.set_arguments(q0, q1_sums, ring_v, comp.rank, 4, rrp)
        g.encode(encoder)
        background_sums = self._buffer("background_sums", 4 * comp.rank)
        shader = self._shader("gather_columns", "background_sums")
        set_constants_and_resources(
            shader,
            {"n_rows": comp.rank, "x_stride": 4, "n_columns": 1, "n_out_cols": 1},
            {0: ring_v, 1: create_buffer(device, np.zeros(1, dtype=np.uint32)), 2: background_sums},
        )
        shader.encode(encoder, 1, comp.rank)

        self._encode_quadratic_form(
            encoder,
            "background",
            gram,
            background_sums,
            0,
            comp.n_frames,
            self._bkgd_corr_img_normalizer,
            self._bkgd_corr_img_mean,
        )
        self._submit(encoder)

    def _encode_standardize(
        self, encoder: wgpu.GPUCommandEncoder, signals: SignalBuffers
    ) -> tuple[wgpu.GPUBuffer, wgpu.GPUBuffer, wgpu.GPUBuffer]:
        """c~ [round_up(n_signals, 4), n_frames_padded], its rows beyond the signals 0, and the stats and max |c| of
        standardize.wgsl"""
        comp = self._compression
        n = signals.n_signals
        c_tilde = self._buffer("c_tilde", 4 * round_up(n, 4) * comp.n_frames_padded)
        encoder.clear_buffer(c_tilde)
        stats = self._buffer("stats", 16 * n)
        max_abs = self._buffer("max_abs", 4 * n)
        shader = self._shader("standardize", reductions=True)
        set_constants_and_resources(
            shader,
            {"n_frames": comp.n_frames, "n4": comp.n_frames_padded // 4},
            {0: signals.buffers["temporal_demixed"], 1: c_tilde, 2: stats, 3: max_abs},
        )
        shader.encode(encoder, n)
        return c_tilde, stats, max_abs

    def _next_w(self, nbytes: int) -> wgpu.GPUBuffer:
        """the other of the two buffers of W, so that the previous W can be read while the next one is written"""
        self._w_turn = 1 - self._w_turn
        return self._buffer(f"w_{self._w_turn}", nbytes)

    def get_merge_pairs(
        self, signals: SignalBuffers, merge_threshold: float = 0.8, overlap_threshold: float = 0.4
    ) -> np.ndarray:
        """
        masknmf's merge test (_compute_indices_to_merge): the pairs of signals whose supports dilated by an 11 x 11 box
        overlap and whose standard correlation images, thresholded at merge_threshold over the whole field of view,
        overlap by more than overlap_threshold of each thresholded area (an area of 0 counts as 1). Standardizes the
        traces and computes W = V c~ for all signals, which the residual correlation images reuse.

        Returns [n_pairs, 2] int64, pairs (i, j) with i < j, sorted.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        tiles = comp.spatial_compressed
        n = signals.n_signals
        n4 = round_up(n, 4)
        n_cells = comp.n_cells[0] * comp.n_cells[1]
        n_words = -(-comp.cell_size**2 // 32)
        row_words = round_up(n_cells * n_words, 4)

        encoder = self._new_encoder()
        c_tilde, stats, _ = self._encode_standardize(encoder, signals)
        # which overwrites those of the last update_residual
        self._residual_buffers = None
        w = self._next_w(4 * comp.rank * n4)
        # 8 splits of the frames: the residual normalizer cancels much of each pixel's energy, with 1 split its error
        # was 5 times masknmf's (sanity_check_residual.py)
        g = self._gemm("w", n_splits=8)
        g.set_arguments(comp.temporal_compressed, c_tilde, w, comp.rank, n4, comp.n_frames_padded)
        g.encode(encoder)
        self._w = (w, n4)
        bits = self._buffer("bits", 4 * n * row_words)
        encoder.clear_buffer(bits)
        shader = self._shader("standard_image_bitmasks")
        set_constants_and_resources(
            shader,
            {
                "cell_size": comp.cell_size,
                "n_cells_x": comp.n_cells[1],
                "n_cells_y": comp.n_cells[0],
                "max_cell_cols": comp.max_cell_cols,
                "threshold": merge_threshold,
                "row_words": row_words,
            },
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: w,
                4: self._std_corr_img_mean,
                5: self._std_corr_img_normalizer,
                6: stats,
                7: bits,
            },
        )
        shader.set_uniform(8, np.array([n, n4, 0, 0], dtype=np.uint32))
        shader.encode(encoder, n_cells, -(-n // 64))
        self._submit(encoder)

        # the candidate pairs on the host while the GPU computes, and each of their signals with itself for the areas
        s = signals.structures
        pairs = get_candidate_pairs(s["a_pixels"], np.repeat(np.arange(n), np.diff(s["a_ptr"])), n, comp.fov_shape)
        if pairs.shape[0] == 0:
            self._merge_candidates = (pairs, np.zeros(0, dtype=np.uint32), np.ones(n, dtype=np.float32))
            return pairs
        paired = np.unique(pairs)
        all_pairs = np.concatenate([pairs, np.stack([paired, paired], axis=1)]).astype(np.uint32)

        encoder = self._new_encoder()
        overlap_counts = self._buffer("overlap_counts", 4 * all_pairs.shape[0])
        shader = self._shader("bitmask_overlaps", reductions=True)
        set_constants_and_resources(
            shader, {"n_row_words": row_words}, {0: bits, 1: create_buffer(device, all_pairs), 2: overlap_counts}
        )
        shader.set_uniform(3, np.array([all_pairs.shape[0], 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(all_pairs.shape[0]))
        self._submit(encoder)
        counts = read_buffer(overlap_counts, np.uint32, (all_pairs.shape[0],))

        # masknmf's fractions in float32, areas of 0 as 1
        areas = np.ones(n, dtype=np.float32)
        areas[paired] = np.maximum(counts[pairs.shape[0] :], 1)
        overlaps = counts[: pairs.shape[0]].astype(np.float32)
        threshold = np.float32(overlap_threshold)
        merge = (overlaps / areas[pairs[:, 0]] > threshold) & (overlaps / areas[pairs[:, 1]] > threshold)
        self._merge_candidates = (pairs, counts[: pairs.shape[0]], areas)
        return pairs[merge]

    def get_merge_candidates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Of the last ``get_merge_pairs``: the candidate pairs [n_pairs, 2], the number of pixels where both of their
        thresholded images are above the threshold, and the number of pixels above it for each signal as masknmf
        divides by it, 0 as 1 (1 for the signals in no pair).
        """
        return self._merge_candidates

    def update_residual(self, signals: SignalBuffers, preserved: np.ndarray | None = None):
        """
        masknmf's residual correlation images (_compute_residual_correlation_image) of signals, with their ring term,
        the robust noise term of ``set_robust_noise`` and the uv_norms of ``set_uv_norms``: the mean and normalizer of
        the residual at each pixel and the values of each signal's image on its support, with their max per signal for
        the support update (``get_signals_to_keep``).

        W = V c~ of the signals of the previous ``get_merge_pairs`` or ``update_residual`` is reused: signal j <
        preserved.size is signal preserved[j] there, W is computed for the signals after them. preserved None: the same
        signals, an empty preserved: W is computed for all signals, also without a previous one.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        tiles = comp.spatial_compressed
        s = signals.structures
        b = signals.buffers
        n = signals.n_signals
        n4 = round_up(n, 4)
        rank = comp.rank
        n_frames_padded = comp.n_frames_padded
        preserved = np.arange(n) if preserved is None else np.asarray(preserved)
        n_added = n - preserved.size

        encoder = self._new_encoder()
        c_tilde, stats, max_abs = self._encode_standardize(encoder, signals)

        # W: the preserved signals' columns, a GEMM for the others
        source, source_n4 = (self._unused, 0) if preserved.size == 0 else self._w
        added_n4 = round_up(n_added, 4)
        w_added = self._unused
        if n_added > 0:
            added_c_tilde = self._buffer("added_c_tilde", 4 * added_n4 * n_frames_padded)
            encoder.clear_buffer(added_c_tilde)
            encoder.copy_buffer_to_buffer(
                c_tilde, 4 * preserved.size * n_frames_padded, added_c_tilde, 0, 4 * n_added * n_frames_padded
            )
            w_added = self._buffer("w_added", 4 * rank * added_n4)
            # 8 splits as for W in get_merge_pairs, the default target gives 1 split for 257 or more signals
            g = self._gemm("w_added", n_splits=8)
            g.set_arguments(comp.temporal_compressed, added_c_tilde, w_added, rank, added_n4, n_frames_padded)
            g.encode(encoder)
        w = self._next_w(4 * rank * n4)
        shader = self._shader("assemble_columns")
        set_constants_and_resources(
            shader,
            {"n_rows": rank},
            {0: source, 1: create_buffer(device, preserved.astype(np.uint32)), 2: w_added, 3: w},
        )
        shader.set_uniform(4, np.array([source_n4, added_n4, n4, preserved.size, n_added, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, -(-n4 // 256), rank)
        self._w = (w, n4)

        # W' = V' c~ and v' = V' 1 with V' = V - q0 q1
        rrp = signals.ring_rank_padded
        w_prime = w
        v_prime = self._v_sums
        if rrp > 0:
            q0, q1 = b["ring_left"], b["ring_right"]
            c_q1 = self._buffer("c_q1", 4 * n4 * rrp)
            g = self._gemm("c_q1")
            g.set_arguments(c_tilde, q1, c_q1, n4, rrp, n_frames_padded)
            g.encode(encoder)
            ring_w = self._buffer("ring_w", 4 * rank * n4)
            g = self._gemm("ring_w")
            g.set_arguments(q0, c_q1, ring_w, rank, n4, rrp)
            g.encode(encoder)
            w_prime = self._buffer("w_prime", 4 * rank * n4)
            self._encode_axpy(encoder, "w_prime", w, ring_w, w_prime, rank * n4, 1)

            # q1 1 in the first row of a [4, rrp] matrix, q0 q1 1 in the first column of [rank, 4]
            q1_sums = self._buffer("q1_sums", 16 * rrp)
            encoder.clear_buffer(q1_sums)
            self._encode_row_sums(encoder, "q1", q1, n_frames_padded, 0, comp.n_frames, q1_sums, n_rows=rrp)
            ring_v = self._buffer("ring_v", 16 * rank)
            g = self._gemm("ring_v")
            g.set_arguments(q0, q1_sums, ring_v, rank, 4, rrp)
            g.encode(encoder)
            v_prime = self._buffer("v_prime", 4 * rank)
            self._encode_axpy(encoder, "v_prime", self._v_sums, ring_v, v_prime, rank, 4)

        # c_s . c~_i for the signals that overlap
        products = self._buffer("products", 4 * s["neighbors"].size)
        self_products = self._buffer("self_products", 4 * n)
        shader = self._shader("trace_products", reductions=True)
        set_constants_and_resources(
            shader,
            {"n4": n_frames_padded // 4},
            {
                0: b["temporal_demixed"],
                1: c_tilde,
                2: b["neighbor_ptr"],
                3: b["neighbors"],
                4: products,
                5: self_products,
            },
        )
        shader.set_uniform(6, np.array([n, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(n))

        noise_term = np.float32(comp.n_frames) * np.float32(self._robust_noise) ** 2
        sizes = np.array([n4, 0], dtype=np.uint32)
        sizes[1:] = np.array([noise_term], dtype=np.float32).view(np.uint32)
        n_pixels = self._n_pixels
        norm2 = self._buffer("norm2", 4 * n_pixels)
        x_entries = self._buffer("x_entries", 4 * s["a_pixels"].size)
        shader = self._shader("residual_pixels")
        set_constants_and_resources(
            shader,
            {"cell_size": comp.cell_size, "n_cells_x": comp.n_cells[1], "n_frames": float(comp.n_frames)},
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: w_prime,
                4: v_prime,
                5: b["pixel_ptr"],
                6: b["pixel_entries"],
                7: b["pixel_signals"],
                8: b["a_values"],
                9: stats,
                10: b["neighbor_ptr"],
                11: b["neighbors"],
                12: products,
                13: self_products,
                14: self._uv_norms,
                15: self._resid_corr_img_mean,
                16: self._resid_corr_img_normalizer,
                17: norm2,
                18: x_entries,
            },
        )
        shader.set_uniform(19, sizes)
        shader.encode(encoder, comp.n_cells[0] * comp.n_cells[1])

        # the group of each signal
        signal_groups = np.empty(n, dtype=np.uint32)
        signal_groups[s["group_signals"]] = np.repeat(np.arange(s["group_ptr"].size - 1), np.diff(s["group_ptr"]))
        self._resid_corr_img_support_values = self._buffer("support_values", 4 * s["a_pixels"].size)
        self._signal_maxima = self._buffer("signal_maxima", 8 * n)
        shader = self._shader("residual_support_values", reductions=True)
        set_constants_and_resources(
            shader,
            {},
            {
                0: b["a_ptr"],
                1: b["a_pixels"],
                2: b["a_values"],
                3: b["pixel_ptr"],
                4: b["pixel_entries"],
                5: b["pixel_signals"],
                6: create_buffer(device, signal_groups),
                7: stats,
                8: max_abs,
                9: b["neighbor_ptr"],
                10: b["neighbors"],
                11: products,
                12: self_products,
                13: self._resid_corr_img_mean,
                14: norm2,
                15: x_entries,
                16: self._resid_corr_img_support_values,
                17: self._signal_maxima,
            },
        )
        shader.set_uniform(18, np.array([n, sizes[1]], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(n))
        self._submit(encoder)
        self._residual_signals = signals
        self._residual_buffers = {"c_tilde": c_tilde, "stats": stats, "w_prime": w_prime}

    def get_resid_corr_img_mean(self) -> np.ndarray:
        return read_buffer(self._resid_corr_img_mean, np.float32, (self._n_pixels,)).copy()

    def get_resid_corr_img_normalizer(self) -> np.ndarray:
        return read_buffer(self._resid_corr_img_normalizer, np.float32, (self._n_pixels,)).copy()

    def get_resid_corr_img_support_values(self) -> np.ndarray:
        """the values on the support of a of the signals of the last update_residual, in the order of their entries
        (signal-major, see SignalBuffers.structures)"""
        n_entries = self._residual_signals.structures["a_pixels"].size
        return read_buffer(self._resid_corr_img_support_values, np.float32, (n_entries,)).copy()

    def get_signals_to_keep(self, deletion_threshold: float = 0.2, min_brightness: float | None = None) -> np.ndarray:
        """
        masknmf's deletion test (_flag_components_for_deletion) on the signals of the last ``update_residual``: the
        signals with a value of their residual correlation image above deletion_threshold on their support and, if
        min_brightness is not None, a brightness max |a| max |c| of at least min_brightness. Compared in float32 with
        the thresholds rounded to float32, as torch compares them.

        Returns the indices of the signals to keep, sorted.
        """
        n = self._residual_signals.n_signals
        maxima = read_buffer(self._signal_maxima, np.float32, (n, 2))
        # a value above the threshold, as masknmf counts them: the max of the values is above it
        has_entries = np.diff(self._residual_signals.structures["a_ptr"]) > 0
        keep = has_entries & (maxima[:, 0] > np.float32(deletion_threshold))
        n_correlated = int(np.count_nonzero(keep))
        if min_brightness is not None:
            keep &= maxima[:, 1] >= np.float32(min_brightness)
        if not keep.any():
            if n_correlated == 0:
                raise ValueError(
                    f"all {n} remaining signal(s) were deleted: none had a residual correlation above "
                    f"deletion_threshold={deletion_threshold}"
                )
            raise ValueError(f"all {n} remaining signal(s) were deleted: none met min_brightness={min_brightness}")
        return np.flatnonzero(keep)

    def expand_masks(self, support_threshold: float = 0.9, frame_batch_size: int | None = None) -> SignalBuffers:
        """
        masknmf's mask expansion (_mask_expansion_routine) of the signals of the last ``update_residual``: the new
        mask_ab of each signal is the pixels of its support dilated by an 11 x 11 box where its residual correlation
        image is above support_threshold times the max of its support values, compared in float32 as torch does, and a
        has its old entries and zeros at the new pixels. The new signals share c, b and the ring term buffers with the
        old ones, their groups are computed from the new mask_ab with frame_batch_size, see ``SignalBuffers``.
        """
        if self._residual_buffers is None:
            raise RuntimeError("expand_masks needs an update_residual after get_merge_pairs")
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        tiles = comp.spatial_compressed
        signals = self._residual_signals
        s = signals.structures
        b = signals.buffers
        n = signals.n_signals
        a_ptr = s["a_ptr"].astype(np.int64)
        support_maxima = read_buffer(self._signal_maxima, np.float32, (n, 2))[:, 0]

        # the dilated supports, the entry of a at each of their pixels, and the signals with entries in them
        pixels, dilated_signals, on_support = dilate_supports(
            s["a_pixels"], np.repeat(np.arange(n), np.diff(a_ptr)), n, comp.fov_shape
        )
        dilated_ptr = np.concatenate([[0], np.cumsum(np.bincount(dilated_signals, minlength=n))])
        a_entries = np.full(pixels.size, 0xFFFFFFFF, dtype=np.uint32)
        a_entries[on_support] = np.arange(a_ptr[-1], dtype=np.uint32)
        neighbor_ptr, neighbors = compute_dilated_overlap_graph(pixels, dilated_signals, s, n)

        # their products with the standardized traces, then the residual correlation images at the dilated pixels
        encoder = self._new_encoder()
        neighbor_ptr_buffer = create_buffer(device, neighbor_ptr.astype(np.uint32))
        neighbors_buffer = create_buffer(device, neighbors.astype(np.uint32))
        products = self._buffer("dilated_products", 4 * neighbors.size)
        shader = self._shader("trace_products", "dilated", reductions=True)
        set_constants_and_resources(
            shader,
            {"n4": comp.n_frames_padded // 4, "compute_self_products": False},
            {
                0: b["temporal_demixed"],
                1: self._residual_buffers["c_tilde"],
                2: neighbor_ptr_buffer,
                3: neighbors_buffer,
                4: products,
                5: self._unused,
            },
        )
        shader.set_uniform(6, np.array([n, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(n))
        values = self._buffer("dilated_values", 4 * pixels.size)
        shader = self._shader("residual_image_values")
        set_constants_and_resources(
            shader,
            {"cell_size": comp.cell_size, "n_cells_x": comp.n_cells[1]},
            {
                0: create_buffer(device, dilated_ptr.astype(np.uint32)),
                1: create_buffer(device, pixels.astype(np.uint32)),
                2: create_buffer(device, a_entries),
                3: tiles.cell_col_ptr,
                4: tiles.cell_cols,
                5: tiles.tiles,
                6: self._residual_buffers["w_prime"],
                7: self._resid_corr_img_mean,
                8: self._resid_corr_img_normalizer,
                9: self._residual_buffers["stats"],
                10: b["pixel_ptr"],
                11: b["pixel_entries"],
                12: b["pixel_signals"],
                13: b["a_values"],
                14: neighbor_ptr_buffer,
                15: neighbors_buffer,
                16: products,
                17: self._resid_corr_img_support_values,
                18: values,
            },
        )
        shader.set_uniform(19, np.array([n, round_up(n, 4), 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(n))
        self._submit(encoder)

        # the new masks and supports, both among the dilated entries, and the place of each old entry among the new
        thresholds = support_maxima * np.float32(support_threshold)
        in_mask = read_buffer(values, np.float32, (pixels.size,)) > thresholds[dilated_signals]
        in_support = on_support | in_mask
        places = (np.cumsum(in_support) - 1)[on_support]

        # the old values of a at their places, zeros at the new entries
        a_values = create_empty_buffer(device, 4 * int(np.count_nonzero(in_support)))
        encoder = self._new_encoder()
        shader = self._shader("scatter")
        set_constants_and_resources(
            shader, {}, {0: b["a_values"], 1: create_buffer(device, places.astype(np.uint32)), 2: a_values}
        )
        shader.set_uniform(3, np.array([places.size, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, *dispatch_grid(-(-places.size // 256)))
        self._submit(encoder)

        expanded = SignalBuffers.from_buffers(
            comp,
            pixels[in_support],
            dilated_signals[in_support],
            n,
            a_values,
            b["temporal_demixed"],
            b["b"],
            pixels[in_mask],
            dilated_signals[in_mask],
            frame_batch_size,
        )
        expanded.set_ring_term_buffers(b["ring_left"], b["ring_right"], b["ring_right_t"], signals.ring_rank)
        return expanded

    @property
    def robust_noise(self) -> float | None:
        """masknmf's robust noise term, see ``set_robust_noise``"""
        return self._robust_noise

    @property
    def std_corr_img_mean(self) -> wgpu.GPUBuffer:
        """[n_pixels] mean of U V over the frames"""
        return self._std_corr_img_mean

    @property
    def std_corr_img_normalizer(self) -> wgpu.GPUBuffer:
        """[n_pixels] norm of U V - mean over the frames, with the robust noise term"""
        return self._std_corr_img_normalizer

    @property
    def uv_norms(self) -> wgpu.GPUBuffer:
        """[n_pixels] norms of the rows of U (V - q0 q1), see ``set_uv_norms``"""
        return self._uv_norms

    def get_std_corr_img_mean(self) -> np.ndarray:
        return read_buffer(self._std_corr_img_mean, np.float32, (self._n_pixels,)).copy()

    def get_std_corr_img_normalizer(self) -> np.ndarray:
        return read_buffer(self._std_corr_img_normalizer, np.float32, (self._n_pixels,)).copy()

    def get_uv_norms(self) -> np.ndarray:
        return read_buffer(self._uv_norms, np.float32, (self._n_pixels,)).copy()

    def get_bkgd_corr_img_mean(self) -> np.ndarray:
        return read_buffer(self._bkgd_corr_img_mean, np.float32, (self._n_pixels,)).copy()

    def get_bkgd_corr_img_normalizer(self) -> np.ndarray:
        return read_buffer(self._bkgd_corr_img_normalizer, np.float32, (self._n_pixels,)).copy()
