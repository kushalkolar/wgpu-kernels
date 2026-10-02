from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wgpu
import pygfx
import fastplotlib as fpl
from fastplotlib.graphics.features import TextureArray

from masknmf.utils._serialization import load_dict

from ._spmv import ComputeShader, create_storage_buffer


def select_adapter(adapter: wgpu.GPUAdapter):
    """
    Select the adapter used by fastplotlib, request the buffer size limits needed to store the
    compression results, the workgroup size limits of trace_statistics.wgsl (1024 invocations) and the
    subgroups feature if the adapter has it (see ``get_subgroup_size``).
    Must be called before any fastplotlib Figure is created.
    """
    fpl.select_adapter(adapter)
    pygfx.renderers.wgpu.set_wgpu_limits(
        **{
            limit: adapter.limits[limit]
            for limit in (
                "max-storage-buffer-binding-size",
                "max-buffer-size",
                "max-storage-buffers-per-shader-stage",
                "max-compute-invocations-per-workgroup",
                "max-compute-workgroup-size-x",
            )
        }
    )
    # wgpu-native has subgroups as the native feature "subgroup", a preferred feature is enabled if the adapter has it
    wgpu.preconfigure_default_device("select_adapter", preferred_features={"subgroup"})


def load_compression(path: str | Path) -> dict:
    """
    Load the compression results from a demixing results file.

    Returns a dict with the keys "shape", "u", "v", "mean_img", "var_img" and "u_local_projector"
    """
    d = load_dict(path, "DemixingResults")

    return {
        "shape": tuple(int(i) for i in d["shape"]),
        "u": d["u"],
        "v": d["v"],
        "mean_img": d["pmd_mean_img"],
        "var_img": d["pmd_var_img"],
        "u_local_projector": d["pmd_u_projector"],
    }


def get_column_blocks(u: torch.Tensor, fov_shape: tuple[int, int]) -> np.ndarray:
    """
    Each column of U is supported on one rectangular block of pixels.

    Returns
    -------
    np.ndarray
        [rank, 4] (row, col, row_end, col_end) of the block that supports each column of U, ends are exclusive
    """
    height, width = fov_shape
    rank = u.shape[1]

    u = u.coalesce()
    rows, cols = u.indices().cpu().numpy()
    pixel_rows, pixel_cols = rows // width, rows % width

    rects = np.zeros((rank, 4), dtype=np.int64)
    rects[:, 0], rects[:, 1] = height, width
    np.minimum.at(rects[:, 0], cols, pixel_rows)
    np.minimum.at(rects[:, 1], cols, pixel_cols)
    np.maximum.at(rects[:, 2], cols, pixel_rows + 1)
    np.maximum.at(rects[:, 3], cols, pixel_cols + 1)

    return rects


def compute_blocks(
    u: torch.Tensor, fov_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group the columns of U by the block that supports them.

    Returns
    -------
    block_col_ptr: np.ndarray
        [n_blocks + 1] uint32, the columns of block b are ``block_col_ptr[b]:block_col_ptr[b + 1]``

    col_block: np.ndarray
        [rank] uint32, block of each column of U

    block_rects: np.ndarray
        [n_blocks, 4] uint32, (row, col, row_end, col_end) of each block, ends are exclusive

    """
    col_rects = get_column_blocks(u, fov_shape)

    # blocks numbered in order of their first column
    _, first_cols, col_block = np.unique(
        col_rects, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_cols)
    rank_of_block = np.empty_like(order)
    rank_of_block[order] = np.arange(order.size)
    col_block = rank_of_block[col_block.ravel()]

    if not np.all(np.diff(col_block) >= 0):
        raise ValueError("the columns of U that belong to one block must be contiguous")

    block_col_ptr = np.searchsorted(col_block, np.arange(order.size + 1)).astype(
        np.uint32
    )
    block_rects = col_rects[block_col_ptr[:-1]].astype(np.uint32)

    return block_col_ptr, col_block.astype(np.uint32), block_rects


def compute_cell_tiles(
    u: torch.Tensor, fov_shape: tuple[int, int]
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Store the sparse spatial matrix U as one dense tile per cell.

    Each column of U is supported on one block of pixels and the blocks overlap. Every pixel of a
    cell, a square whose side is the gcd of the block starts, block sizes and fov dimensions, lies
    in the same set of blocks, so all pixels of a cell are supported on the same columns of U.

    Parameters
    ----------
    u: torch.Tensor
        sparse COO tensor of shape [n_pixels, rank], pixels flattened in row-major order

    fov_shape: (int, int)
        (height, width) of the field of view

    Returns
    -------
    cell_size: int
        side length of a cell in pixels

    cell_col_ptr: np.ndarray
        [n_cells + 1] uint32, start of each cell's columns in ``cell_cols``

    cell_cols: np.ndarray
        [n_tile_cols] uint32, the column of U for each tile column

    tiles: np.ndarray
        [n_tile_cols * cell_size**2] float32, the values of each tile column are contiguous,
        pixels within a cell are in row-major order

    """
    height, width = fov_shape
    rank = u.shape[1]

    u = u.coalesce()
    rows, cols = u.indices().cpu().numpy()
    values = u.values().cpu().numpy()

    pixel_rows, pixel_cols = rows // width, rows % width

    col_rects = get_column_blocks(u, fov_shape)
    cell_size = int(
        np.gcd.reduce(
            np.concatenate(
                [
                    col_rects[:, 0],
                    col_rects[:, 1],
                    col_rects[:, 2] - col_rects[:, 0],
                    col_rects[:, 3] - col_rects[:, 1],
                    [height, width],
                ]
            )
        )
    )

    n_cells_x = width // cell_size
    n_cells = (height // cell_size) * n_cells_x
    n_cell_pixels = cell_size**2

    cells = (pixel_rows // cell_size) * n_cells_x + pixel_cols // cell_size
    cell_pixels = (pixel_rows % cell_size) * cell_size + pixel_cols % cell_size

    # one tile column for each unique (cell, column of U) pair, sorted by cell
    tile_cols, tile_col_indices, counts = np.unique(
        cells * rank + cols, return_inverse=True, return_counts=True
    )

    if not np.all(counts == n_cell_pixels):
        raise ValueError(
            "the pixels of a cell are not supported on the same set of columns of U"
        )

    cell_col_ptr = np.searchsorted(tile_cols // rank, np.arange(n_cells + 1)).astype(
        np.uint32
    )
    cell_cols = (tile_cols % rank).astype(np.uint32)

    tiles = np.zeros(tile_cols.size * n_cell_pixels, dtype=np.float32)
    tiles[tile_col_indices * n_cell_pixels + cell_pixels] = values

    return cell_size, cell_col_ptr, cell_cols, tiles


@dataclass
class CellTiles:
    """spatial matrix stored as one dense tile per cell, see ``compute_cell_tiles``"""

    cell_col_ptr: wgpu.GPUBuffer
    cell_cols: wgpu.GPUBuffer
    tiles: wgpu.GPUBuffer


class CompressionBuffers:
    """
    The compression results on the GPU

    Parameters
    ----------
    path: str | Path
        path to the demixing results file

    """

    def __init__(self, path: str | Path):
        compression = load_compression(path)
        device = pygfx.renderers.wgpu.get_shared().device

        n_frames, height, width = compression["shape"]
        self._fov_shape = (height, width)
        self._n_frames = n_frames
        self._rank = compression["v"].shape[0]

        cell_size, cell_col_ptr, cell_cols, tiles = compute_cell_tiles(
            compression["u"], self.fov_shape
        )

        # the kernels process 4 horizontally adjacent pixels of a cell per invocation
        if cell_size % 4 != 0:
            raise ValueError(f"cell size must be a multiple of 4, got: {cell_size}")

        _, projector_col_ptr, projector_cols, projector_tiles = compute_cell_tiles(
            compression["u_local_projector"], self.fov_shape
        )

        if not (
            np.array_equal(projector_col_ptr, cell_col_ptr)
            and np.array_equal(projector_cols, cell_cols)
        ):
            raise ValueError("u_local_projector must have the same sparsity as u")

        self._cell_size = cell_size
        self._max_cell_cols = int(np.diff(cell_col_ptr).max())

        block_col_ptr, col_block, block_rects = compute_blocks(
            compression["u"], self.fov_shape
        )
        # kept on the host to build the signal structures
        self._cell_structure = (cell_col_ptr, cell_cols)
        self._block_structure = (block_col_ptr, col_block, block_rects)
        self._max_block_cols = int(np.diff(block_col_ptr).max())

        self._block_col_ptr = create_storage_buffer(device, block_col_ptr)
        self._col_block = create_storage_buffer(device, col_block)
        self._block_rects = create_storage_buffer(device, block_rects)

        # both matrices share the cell structure
        self._cell_col_ptr = create_storage_buffer(device, cell_col_ptr)
        self._cell_cols = create_storage_buffer(device, cell_cols)

        self._spatial_compressed = CellTiles(
            self._cell_col_ptr, self._cell_cols, create_storage_buffer(device, tiles)
        )
        self._spatial_compressed_local_projector = CellTiles(
            self._cell_col_ptr,
            self._cell_cols,
            create_storage_buffer(device, projector_tiles),
        )

        # pad time to a multiple of 4 so rows of V can be read with vec4 loads
        self._n_frames_padded = -(-n_frames // 4) * 4
        v = np.zeros((self.rank, self.n_frames_padded), dtype=np.float32)
        v[:, :n_frames] = compression["v"].numpy()
        self._temporal_compressed = create_storage_buffer(device, v)

        self._mean_image = create_storage_buffer(
            device, np.ascontiguousarray(compression["mean_img"].numpy(), np.float32)
        )
        self._noise_variance_image = create_storage_buffer(
            device, np.ascontiguousarray(compression["var_img"].numpy(), np.float32)
        )

    @property
    def fov_shape(self) -> tuple[int, int]:
        """(height, width) of the field of view"""
        return self._fov_shape

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def n_frames_padded(self) -> int:
        """number of frames rounded up to a multiple of 4, the row stride of the temporal buffers"""
        return self._n_frames_padded

    @property
    def rank(self) -> int:
        """number of columns of U"""
        return self._rank

    @property
    def cell_size(self) -> int:
        return self._cell_size

    @property
    def n_cells(self) -> tuple[int, int]:
        """number of cells along (height, width)"""
        return self.fov_shape[0] // self.cell_size, self.fov_shape[1] // self.cell_size

    @property
    def max_cell_cols(self) -> int:
        """max number of columns of U that support a cell"""
        return self._max_cell_cols

    def get_cell_structure(self) -> tuple[np.ndarray, np.ndarray]:
        """host copies of (cell_col_ptr, cell_cols), see ``compute_cell_tiles``"""
        return self._cell_structure

    def get_block_structure(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """host copies of (block_col_ptr, col_block, block_rects), see ``compute_blocks``"""
        return self._block_structure

    @property
    def n_blocks(self) -> int:
        """number of blocks that support the columns of U"""
        return self._block_structure[2].shape[0]

    @property
    def max_block_cols(self) -> int:
        """max number of columns of U in one block"""
        return self._max_block_cols

    @property
    def block_col_ptr(self) -> wgpu.GPUBuffer:
        """[n_blocks + 1], the columns of block b are ``block_col_ptr[b]:block_col_ptr[b + 1]``"""
        return self._block_col_ptr

    @property
    def col_block(self) -> wgpu.GPUBuffer:
        """[rank], block of each column of U"""
        return self._col_block

    @property
    def block_rects(self) -> wgpu.GPUBuffer:
        """[n_blocks, 4], (row, col, row_end, col_end) of each block, ends are exclusive"""
        return self._block_rects

    @property
    def spatial_compressed(self) -> CellTiles:
        return self._spatial_compressed

    @property
    def spatial_compressed_local_projector(self) -> CellTiles:
        return self._spatial_compressed_local_projector

    @property
    def temporal_compressed(self) -> wgpu.GPUBuffer:
        """V, [rank, n_frames_padded]"""
        return self._temporal_compressed

    @property
    def mean_image(self) -> wgpu.GPUBuffer:
        return self._mean_image

    @property
    def noise_variance_image(self) -> wgpu.GPUBuffer:
        return self._noise_variance_image


class UVImage:
    """
    Frame of the denoised movie, (U @ V[:, t]) * noise_variance_image + mean_image, computed into a
    Texture that is shown as a fastplotlib ImageGraphic.

    Parameters
    ----------
    compression: CompressionBuffers

    benchmark: bool, default False
        record the compute time of every frame

    """

    def __init__(self, compression: CompressionBuffers, benchmark: bool = False):
        self._compression = compression
        height, width = compression.fov_shape

        self._texture_array = TextureArray(
            data=np.zeros((height, width), dtype=np.float32),
            cpu_buffer=False,
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING
                | wgpu.TextureUsage.TEXTURE_BINDING
                | wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST
            ),
        )
        self._texture = self._texture_array.buffer[0, 0]

        self._benchmark = benchmark
        self._timings = list()

        tiles = compression.spatial_compressed
        self._compute_shader = ComputeShader(
            Path(__file__).parent.joinpath("uv_frame.wgsl").read_text(),
            entry_point="uv_frame",
            report_time=benchmark,
        )
        self._compute_shader.set_constant("cell_size", compression.cell_size)
        self._compute_shader.set_constant("n_cells_x", compression.n_cells[1])
        self._compute_shader.set_constant("n_frames_padded", compression.n_frames_padded)
        self._compute_shader.set_constant("max_cell_cols", compression.max_cell_cols)
        self._compute_shader.set_resource(0, tiles.cell_col_ptr)
        self._compute_shader.set_resource(1, tiles.cell_cols)
        self._compute_shader.set_resource(2, tiles.tiles)
        self._compute_shader.set_resource(3, compression.temporal_compressed)
        self._compute_shader.set_resource(5, self._texture)
        self._compute_shader.set_resource(6, compression.noise_variance_image)
        self._compute_shader.set_resource(7, compression.mean_image)

        self._t = np.array([0], dtype=np.uint32)
        self.t = 0

        vmin, vmax = self._compute_vmin_vmax()
        self._image_graphic = fpl.ImageGraphic(self._texture_array, vmin=vmin, vmax=vmax)

    @property
    def image_graphic(self) -> fpl.ImageGraphic:
        return self._image_graphic

    def _compute_vmin_vmax(self, n_samples: int = 10) -> tuple[float, float]:
        """min and max over ``n_samples`` frames spread across the movie"""
        vmin, vmax = np.inf, -np.inf
        for t in np.linspace(0, self._compression.n_frames - 1, n_samples).astype(int):
            self.t = t
            frame = self.to_numpy()
            vmin, vmax = min(vmin, float(frame.min())), max(vmax, float(frame.max()))

        self.t = 0
        return vmin, vmax

    def to_numpy(self) -> np.ndarray:
        """download the current frame from the GPU"""
        device = pygfx.renderers.wgpu.get_shared().device
        wgpu_texture = pygfx.renderers.wgpu.engine.update.ensure_wgpu_object(
            self._texture
        )
        height, width = self._compression.fov_shape

        buf = device.queue.read_texture(
            source={"texture": wgpu_texture, "origin": (0, 0, 0), "mip_level": 0},
            data_layout={"offset": 0, "bytes_per_row": width * 4},
            size=(width, height, 1),
        )

        return np.frombuffer(buf, dtype=np.float32).reshape(height, width)

    @property
    def t(self) -> int:
        """get or set the frame index"""
        return int(self._t[0])

    @t.setter
    def t(self, value: int):
        self._t[0] = int(value)
        self._compute_shader.set_uniform(4, self._t)

        n_cells_y, n_cells_x = self._compression.n_cells
        timing = self._compute_shader.dispatch(n_cells_x, n_cells_y)
        if self._benchmark:
            self._timings.append(timing)

    def get_timings(self) -> np.ndarray:
        """compute time of every frame in milliseconds"""
        if not self._benchmark:
            raise ValueError("Must create with benchmark=True to get timings")

        return np.asarray(self._timings)

    def clear_timings(self):
        self._timings = list()
