"""
The GPU side of the demixing viewer: frames of masknmf's PMD, AC and residual arrays (DemixingFrames), computed from the
buffers of the compression and of the signals
"""

import numpy as np
import pygfx
import wgpu
import fastplotlib as fpl
from fastplotlib.graphics.features import TextureArray

from ._compression import CompressionBuffers
from ._hals import SignalBuffers, create_empty_buffer, load_shader, set_constants_and_resources


class DemixingFrames:
    """
    Frame t of masknmf's PMD, AC and residual arrays without rescaling, U V[:, t], a c[t] and
    U V[:, t] - U q0 q1[:, t] - a c[t] - b, computed into Textures that are shown as fastplotlib ImageGraphics. The
    three images have the vmin and vmax of the PMD frames, the min and max of 10 frames spread across the movie.

    Parameters
    ----------
    compression: CompressionBuffers

    """

    def __init__(self, compression: CompressionBuffers):
        device = pygfx.renderers.wgpu.get_shared().device
        self._compression = compression
        height, width = compression.fov_shape

        self._texture_arrays = tuple(
            TextureArray(
                data=np.zeros((height, width), dtype=np.float32),
                cpu_buffer=False,
                usage=(
                    wgpu.TextureUsage.STORAGE_BINDING
                    | wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.COPY_SRC
                    | wgpu.TextureUsage.COPY_DST
                ),
            )
            for _ in range(3)
        )

        # the ring term of the frame on the columns of U, 0 without a ring term
        self._x = create_empty_buffer(device, 4 * compression.rank)
        self._ring_frame = load_shader("ring_frame.wgsl")
        set_constants_and_resources(self._ring_frame, {"rank": compression.rank}, {3: self._x})

        tiles = compression.spatial_compressed
        self._demixing_frames = load_shader("demixing_frames.wgsl")
        set_constants_and_resources(
            self._demixing_frames,
            {
                "cell_size": compression.cell_size,
                "n_cells_x": compression.n_cells[1],
                "n_frames_padded": compression.n_frames_padded,
                "max_cell_cols": compression.max_cell_cols,
            },
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: compression.temporal_compressed,
                5: self._x,
                12: self._texture_arrays[0].buffer[0, 0],
                13: self._texture_arrays[1].buffer[0, 0],
                14: self._texture_arrays[2].buffer[0, 0],
            },
        )

        # a without entries and b = 0, the buffers for no signals
        self._no_signals = {
            "pixel_ptr": create_empty_buffer(device, 4 * (height * width + 1)),
            "pixel_entries": create_empty_buffer(device, 0),
            "pixel_signals": create_empty_buffer(device, 0),
            "a_values": create_empty_buffer(device, 0),
            "temporal_demixed": create_empty_buffer(device, 0),
            "b": create_empty_buffer(device, 4 * height * width),
        }

        self._t = 0
        self.set_signals(None)

        vmin, vmax = self._compute_vmin_vmax()
        self._image_graphics = tuple(
            fpl.ImageGraphic(texture_array, vmin=vmin, vmax=vmax) for texture_array in self._texture_arrays
        )

    @property
    def image_graphics(self) -> tuple[fpl.ImageGraphic, fpl.ImageGraphic, fpl.ImageGraphic]:
        """the PMD, AC and residual images"""
        return self._image_graphics

    def _compute_vmin_vmax(self, n_samples: int = 10) -> tuple[float, float]:
        """min and max of the PMD frames over ``n_samples`` frames spread across the movie"""
        vmin, vmax = np.inf, -np.inf
        for t in np.linspace(0, self._compression.n_frames - 1, n_samples).astype(int):
            self.t = t
            pmd = self.get_frames()[0]
            vmin, vmax = min(vmin, float(pmd.min())), max(vmax, float(pmd.max()))

        self.t = 0
        return vmin, vmax

    def set_signals(self, signals: SignalBuffers | None):
        """
        The signals of the AC and residual frames, None for no signals. Their buffers are read when a frame is
        computed, set the signals again after they changed. Computes frame t again.
        """
        b = self._no_signals if signals is None else signals.buffers
        set_constants_and_resources(
            self._demixing_frames,
            {},
            {
                6: b["pixel_ptr"],
                7: b["pixel_entries"],
                8: b["pixel_signals"],
                9: b["a_values"],
                10: b["temporal_demixed"],
                11: b["b"],
            },
        )

        self._ring_rank_padded = 0 if signals is None else signals.ring_rank_padded
        if self._ring_rank_padded > 0:
            set_constants_and_resources(self._ring_frame, {}, {0: b["ring_left"], 1: b["ring_right_t"]})
        else:
            device = pygfx.renderers.wgpu.get_shared().device
            encoder = device.create_command_encoder()
            encoder.clear_buffer(self._x)
            device.queue.submit([encoder.finish()])

        self._compute_frame()

    @property
    def t(self) -> int:
        """get or set the frame index"""
        return self._t

    @t.setter
    def t(self, value: int):
        self._t = int(value)
        self._compute_frame()

    def _compute_frame(self):
        """frame t into the textures"""
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        encoder = device.create_command_encoder()
        if self._ring_rank_padded > 0:
            self._ring_frame.set_uniform(2, np.array([self._t, self._ring_rank_padded // 4], dtype=np.uint32))
            self._ring_frame.encode(encoder, -(-comp.rank // 256))
        self._demixing_frames.set_uniform(4, np.array([self._t], dtype=np.uint32))
        n_cells_y, n_cells_x = comp.n_cells
        self._demixing_frames.encode(encoder, n_cells_x, n_cells_y)
        device.queue.submit([encoder.finish()])

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """download the current PMD, AC and residual frames from the GPU"""
        device = pygfx.renderers.wgpu.get_shared().device
        height, width = self._compression.fov_shape
        frames = []
        for texture_array in self._texture_arrays:
            wgpu_texture = pygfx.renderers.wgpu.engine.update.ensure_wgpu_object(texture_array.buffer[0, 0])
            data = device.queue.read_texture(
                source={"texture": wgpu_texture, "origin": (0, 0, 0), "mip_level": 0},
                data_layout={"offset": 0, "bytes_per_row": width * 4},
                size=(width, height, 1),
            )
            frames.append(np.frombuffer(data, dtype=np.float32).reshape(height, width))

        return tuple(frames)
