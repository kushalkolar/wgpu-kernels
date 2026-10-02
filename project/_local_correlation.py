"""
masknmf's local correlation image (get_local_correlation_structure) on the GPU, see LocalCorrelationImage
"""

import numpy as np
import pygfx
import wgpu

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    SignalBuffers,
    set_constants_and_resources,
    create_buffer,
    create_empty_buffer,
    read_buffer,
    dispatch_grid,
    get_subgroup_size,
    round_up,
)

SIGNS = {"unconstrained": 0, "positive": 1, "negative": 2}


class LocalCorrelationImage(GPUComputation):
    """
    masknmf's local correlation image (get_local_correlation_structure): the mean correlation of each pixel's residual
    trace with those of the adjacent pixels, the traces thresholded at their median absolute deviation and normalized
    with the robust noise term. Superpixel initialization finds its peaks, at the end of a pass it is the global
    residual correlation image.

    Band by band, one row of cells at a time: the residual traces of the band (residual_traces.wgsl, with
    V - q0 q1 from subtract_ring.wgsl in a buffer of the size of V for the call), the statistics of each trace
    (trace_statistics.wgsl) and the correlations with the pixels to the left and above (neighbor_correlations.wgsl,
    the row above from the previous band), then the means (neighbor_average.wgsl). Needs subgroups of at least 32
    invocations and at most 32768 frames.

    Parameters
    ----------
    compression: CompressionBuffers

    n_frame_groups: int
        the frames of the correlations are split into this many groups, summed by separate workgroups
    """

    def __init__(self, compression: CompressionBuffers, n_frame_groups: int = 8):
        super().__init__()
        device = pygfx.renderers.wgpu.get_shared().device
        self._compression = compression
        self._n_groups = n_frame_groups
        height, width = compression.fov_shape
        n_pixels = height * width
        if compression.n_frames_padded > 4 * 1024 * 8:
            raise ValueError(
                f"trace_statistics.wgsl holds at most 32768 frames per trace, got {compression.n_frames_padded}"
            )
        subgroup_size = get_subgroup_size()
        if subgroup_size is None or subgroup_size < 32:
            raise ValueError(f"trace_statistics.wgsl needs subgroups of at least 32 invocations, got {subgroup_size}")

        n_band_pixels = compression.cell_size * width
        self._traces = create_empty_buffer(device, 4 * n_band_pixels * compression.n_frames_padded)
        self._carry_traces = create_empty_buffer(device, 4 * width * compression.n_frames_padded)
        self._stats = create_empty_buffer(device, 16 * n_band_pixels)
        self._carry_stats = create_empty_buffer(device, 16 * width)
        self._partial = create_empty_buffer(device, 16 * n_frame_groups * n_pixels)
        self._image = create_empty_buffer(device, 4 * n_pixels)
        # the pixel-major view of a without signals
        self._no_entries = create_buffer(device, np.zeros(n_pixels + 1, dtype=np.uint32))

    def compute(
        self,
        noise: float,
        mad_threshold: int = 1,
        sign: str = "unconstrained",
        signals: SignalBuffers | None = None,
    ):
        """
        The local correlation image of the residual U V - a c^T - U q0 q1 of signals with their ring term, or of U V
        without signals, with masknmf's robust noise term noise (``CorrelationImages.robust_noise``). A mad_threshold of
        0 keeps all frames whatever the sign, as masknmf does.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        tiles = comp.spatial_compressed
        height, width = comp.fov_shape
        cell_size = comp.cell_size
        n_cells_y, n_cells_x = comp.n_cells
        n4 = comp.n_frames_padded // 4
        n_band_pixels = cell_size * width
        sign_index = SIGNS[sign] if mad_threshold != 0 else 0

        encoder = self._new_encoder()
        v = comp.temporal_compressed
        if signals is None:
            pixel_ptr = pixel_entries = pixel_signals = a_values = c = self._no_entries
        else:
            b = signals.buffers
            pixel_ptr, pixel_entries, pixel_signals = b["pixel_ptr"], b["pixel_entries"], b["pixel_signals"]
            a_values, c = b["a_values"], b["temporal_demixed"]
            rrp = signals.ring_rank_padded
            # V' = V - q0 q1 for the duration of the call
            if rrp > 0:
                v = create_empty_buffer(device, 16 * comp.rank * n4)
                shader = self._shader("subtract_ring")
                set_constants_and_resources(
                    shader,
                    {"rank": comp.rank, "n4": n4, "ring_rank_padded": rrp},
                    {0: comp.temporal_compressed, 1: b["ring_left"], 2: b["ring_right"], 3: v},
                )
                shader.encode(encoder, -(-n4 // 256), -(-comp.rank // 8))

        bands = np.zeros((n_cells_y, 4), dtype=np.uint32)
        bands[:, 0] = np.arange(n_cells_y)
        bands[1:, 1] = 1
        traces = self._shader("residual_traces")
        set_constants_and_resources(
            traces,
            {"cell_size": cell_size, "n_cells_x": n_cells_x, "max_cell_cols": comp.max_cell_cols, "n4": n4},
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: v,
                4: pixel_ptr,
                5: pixel_entries,
                6: pixel_signals,
                7: a_values,
                8: c,
                9: self._traces,
            },
        )
        traces.set_uniform(10, bands)
        statistics = self._shader("trace_statistics")
        set_constants_and_resources(
            statistics,
            {
                "n_frames": comp.n_frames,
                "n4": n4,
                "n_slots": -(-n4 // 1024),
                "mad_threshold": float(mad_threshold),
                "sign": sign_index,
            },
            {0: self._traces, 1: self._stats},
        )
        noise_term = np.array([np.float32(comp.n_frames) * np.float32(noise) ** 2], dtype=np.float32)
        statistics.set_uniform(2, np.array([n_band_pixels, noise_term.view(np.uint32)[0], 0, 0], dtype=np.uint32))
        correlations = self._shader("neighbor_correlations")
        set_constants_and_resources(
            correlations,
            {
                "cell_size": cell_size,
                "n_cells_x": n_cells_x,
                "n_cells_y": n_cells_y,
                "n4": n4,
                "n_frames": comp.n_frames,
                "group4": round_up(-(-n4 // self._n_groups), 4),
                "sign": sign_index,
            },
            {0: self._traces, 1: self._carry_traces, 2: self._stats, 3: self._carry_stats, 4: self._partial},
        )
        correlations.set_uniform(5, bands)

        last_row = (cell_size - 1) * width
        for band in range(n_cells_y):
            traces.encode(encoder, -(-n4 // 8), n_cells_x, uniform_entry=band)
            statistics.encode(encoder, *dispatch_grid(n_band_pixels))
            correlations.encode(encoder, self._n_groups, n_cells_x, uniform_entry=band)
            # the band's last row for the next band
            encoder.copy_buffer_to_buffer(self._traces, 16 * last_row * n4, self._carry_traces, 0, 16 * width * n4)
            encoder.copy_buffer_to_buffer(self._stats, 16 * last_row, self._carry_stats, 0, 16 * width)
        average = self._shader("neighbor_average")
        set_constants_and_resources(
            average,
            {"height": height, "width": width, "n_groups": self._n_groups},
            {0: self._partial, 1: self._image},
        )
        average.encode(encoder, *dispatch_grid(-(-height * width // 256)))
        self._submit(encoder)

    def get_image(self) -> np.ndarray:
        """[height, width] the local correlation image of the last ``compute``"""
        return read_buffer(self._image, np.float32, self._compression.fov_shape).copy()

    @property
    def image(self) -> wgpu.GPUBuffer:
        """[n_pixels] the local correlation image of the last ``compute``, pixels in row-major order"""
        return self._image
