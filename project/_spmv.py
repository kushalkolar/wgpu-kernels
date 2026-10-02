from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse
import wgpu
import pygfx
import fastplotlib as fpl
from fastplotlib.graphics.features import TextureArray

import time

from typing import Optional, Union, Literal
import pygfx as gfx


# `ComputeShader` helper class adapted from pygfx to add uniform buffer support, recording several dispatches into one
# command encoder, and return timing information for benchmarks
# TODO: move this into wgpu/pygfx/new lib
class ComputeShader:
    """Abstraction for a compute shader.

    Parameters
    ----------
    wgsl : str
        The compute shader's code as WGSL.
    entry_point : str | None
        The name of the wgsl function that must be called.
        If the wgsl code has only one entry-point (a function marked with ``@compute``)
        this argument can be omitted.
    label : str | None
        The label for this shader. Used to set labels of underlying wgpu objects,
        and in debugging messages. If not set, use the entry_point.
    report_time : bool
        When set to True, will print the spent time to run the shader.
    """

    def __init__(
        self,
        wgsl,
        *,
        entry_point: Optional[str] = None,
        label: Optional[str] = None,
        report_time: bool = False,
    ):
        # Fixed
        self._wgsl = wgsl
        self._entry_point = entry_point
        self._label = label or entry_point or ""
        self._report_time = report_time

        # Things that can be changed via the API
        self._resources = {}
        self._constants = {}
        # (number of rows, stride between rows in bytes, row size in bytes) of each uniform
        self._uniform_rows = {}

        # Flag to keep track whether this object changed.
        # Note that this says nothing about the contents of buffers/textures used as input.
        self._changed = True

        # Internal variables
        self._device = gfx.renderers.wgpu.Shared.get_instance().device
        self._shader_module = None
        self._pipeline = None
        # one bind group per row of the uniforms that have a row per dispatch
        self._bind_groups = None

    @property
    def changed(self) -> bool:
        """Whether the shader has been changed.

        This can be a new value for a constant, or a different resource.
        Note that this says nothing about the values inside a buffer or texture resource.
        This value is reset when ``encode()`` or ``dispatch()`` is called.
        """
        return self._changed

    def set_resource(
        self,
        index: int,
        resource: Union[gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture],
        *,
        clear=False,
    ):
        """Set a resource.

        Parameters
        ----------
        index : int
            The binding index to connect this resource to. (The group is hardcoded to zero for now.)
        resource : buffer | texture
            The buffer or texture to attach. Can be a wgpu or pygfx resource.
        clear : bool
            When set to True (only possible for a buffer), the resource is cleared to zeros
            right before running the shader.
        """
        # Check
        if not isinstance(index, int):
            raise TypeError(f"ComputeShader resource index must be int, not {index!r}.")
        if not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, wgpu.GPUTexture)
        ):
            raise TypeError(
                f"ComputeShader resource value must be gfx.Buffer, gfx.Texture, wgpu.GPUBuffer, or wgpu.GPUTexture, not {resource!r}"
            )
        clear = bool(clear)
        if clear and not isinstance(
            resource, (gfx.Buffer, gfx.Texture, wgpu.GPUBuffer)
        ):
            raise ValueError("Can only clear a buffer, not a texture.")

        old_value = self._resources.get(index)
        if old_value is not None and old_value[2] == "uniform":
            raise ValueError(
                f"Binding index {index} is already used as a uniform; choose a different index."
            )

        # Value to store
        new_value = (resource, bool(clear), "resource")

        # Update if different
        if new_value != old_value:
            if resource is None:
                self._resources.pop(index, None)
            else:
                self._resources[index] = new_value
            self._bind_groups = None
            self._changed = True

    def set_uniform(self, index: int, data: np.ndarray):
        """
        Set a uniform buffer from a 1D array, or from a 2D array with one row per dispatch: ``encode()`` and
        ``dispatch()`` bind the row given by ``uniform_entry``.
        """
        rows = np.atleast_2d(data)
        n_rows = rows.shape[0]

        # rows of at least 16 bytes, at offsets that are multiples of the uniform buffer offset alignment
        row_size = max(-(-rows[0].nbytes // 16) * 16, 16)
        alignment = self._device.limits["min-uniform-buffer-offset-alignment"]
        stride = -(-row_size // alignment) * alignment
        size = (n_rows - 1) * stride + row_size

        old_value = self._resources.get(index)
        if old_value is not None and old_value[2] != "uniform":
            raise ValueError

        # create GPU buffer and write value, similar to set_resource()
        if old_value is None or self._uniform_rows[index] != (n_rows, stride, row_size):
            buffer = self._device.create_buffer(
                size=size,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._resources[index] = (buffer, False, "uniform")
            self._uniform_rows[index] = (n_rows, stride, row_size)
            self._bind_groups = None
            self._changed = True
        else:
            buffer = old_value[0]

        padded = np.zeros((n_rows, stride), dtype=np.uint8)
        padded[:, : rows[0].nbytes] = np.ascontiguousarray(rows).view(np.uint8).reshape(n_rows, -1)
        self._device.queue.write_buffer(buffer, 0, padded.reshape(-1)[:size])

    def set_constant(self, name: str, value: Union[bool, int, float, None]):
        """Set override constant.

        Setting override constants don't require shader recompilation, but does
        require re-creating the pipeline object. So it's less suited for things
        that change on every draw.
        """
        # NOTE: we could also provide support for uniform variables.
        # The override constants are nice and simple, but require the pipeline
        # to be re-created whenever a contant changes.

        # Check
        if not isinstance(name, str):
            raise TypeError(f"ComputeShader constant name must be str, not {name!r}.")
        if not (value is None or isinstance(value, (bool, int, float))):
            raise TypeError(
                f"ComputeShader constant value must be bool, int, float, or None, not {value!r}."
            )

        # Update if different
        old_value = self._constants.get(name)
        if value != old_value:
            if value is None:
                self._constants.pop(name, None)
            else:
                self._constants[name] = value
            self._pipeline = None
            self._changed = True

    def _get_native_resource(self, resource):
        if isinstance(resource, gfx.Resource):
            return gfx.renderers.wgpu.engine.update.ensure_wgpu_object(resource)
        return resource

    def _get_bindings_from_resources(self, entry: int = 0):
        bindings = []
        for index, (resource, _, kind) in self._resources.items():
            # Get wgpu.GPUBuffer or wgpu.GPUTexture
            wgpu_object = self._get_native_resource(resource)
            if kind == "uniform":
                # row `entry` of a uniform with a row per dispatch, otherwise its only row
                n_rows, stride, row_size = self._uniform_rows[index]
                bindings.append(
                    {
                        "binding": index,
                        "resource": {
                            "buffer": wgpu_object,
                            "offset": stride * entry if n_rows > 1 else 0,
                            "size": row_size,
                        },
                    }
                )
            elif isinstance(wgpu_object, wgpu.GPUBuffer):
                bindings.append(
                    {
                        "binding": index,
                        "resource": {
                            "buffer": wgpu_object,
                            "offset": 0,
                            "size": wgpu_object.size,
                        },
                    }
                )
            elif isinstance(wgpu_object, wgpu.GPUTexture):
                bindings.append(
                    {
                        "binding": index,
                        "resource": wgpu_object.create_view(
                            usage=wgpu.TextureUsage.STORAGE_BINDING
                        ),
                    }
                )
            else:
                raise RuntimeError(f"Unexpected resource: {resource}")
        return bindings

    def _check_bind_groups(self) -> list[wgpu.GPUBindGroup]:
        """compile the shader, create the pipeline and the bind groups if needed"""
        # Get device
        if self._device is None:
            self._shader_module = None
            self._device = gfx.renderers.wgpu.Shared.get_instance().device
        device = self._device

        # Compile the shader
        if self._shader_module is None:
            self._pipeline = None
            self._shader_module = device.create_shader_module(
                label=self._label, code=self._wgsl
            )

        # Get the pipeline object
        if self._pipeline is None:
            self._bind_groups = None
            self._pipeline = device.create_compute_pipeline(
                label=self._label,
                layout="auto",
                compute={
                    "module": self._shader_module,
                    "entry_point": self._entry_point,
                    "constants": self._constants,
                },
            )

        # Get the bind group objects, one per row of the uniforms that have a row per dispatch
        if self._bind_groups is None:
            n_entries = {n_rows for n_rows, _, _ in self._uniform_rows.values() if n_rows > 1}
            if len(n_entries) > 1:
                raise ValueError(
                    "uniforms with a row per dispatch must have the same number of rows"
                )
            bind_group_layout = self._pipeline.get_bind_group_layout(0)
            self._bind_groups = [
                device.create_bind_group(
                    label=self._label,
                    layout=bind_group_layout,
                    entries=self._get_bindings_from_resources(entry),
                )
                for entry in range(max(n_entries, default=1))
            ]

        return self._bind_groups

    def encode(
        self,
        command_encoder: wgpu.GPUCommandEncoder,
        nx,
        ny=1,
        nz=1,
        uniform_entry: int = 0,
    ):
        """
        Record the dispatch into a command encoder as one compute pass, so that several dispatches can be
        submitted together. ``uniform_entry`` is the row of the uniforms that have a row per dispatch.
        """
        nx, ny, nz = int(nx), int(ny), int(nz)

        # Reset
        self._changed = False

        bind_group = self._check_bind_groups()[uniform_entry]

        # Make sure that all used resources have a wgpu-representation, and are synced
        for resource, _, _ in self._resources.values():
            if isinstance(resource, gfx.Resource):
                gfx.renderers.wgpu.engine.update.update_resource(resource)

        # Maybe clear some buffers
        for resource, clear, _ in self._resources.values():
            if clear:
                command_encoder.clear_buffer(self._get_native_resource(resource))

        # Do the compute pass
        compute_pass = command_encoder.begin_compute_pass(label=self._label)
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(nx, ny, nz)
        compute_pass.end()

    def dispatch(self, nx, ny=1, nz=1, uniform_entry: int = 0) -> None | float:
        """
        Dispatch the workgroups, i.e. run the shader.


        """
        # compile the shader and create the pipeline before the timing starts
        self._check_bind_groups()

        t0 = time.perf_counter()

        # Start!
        command_encoder = self._device.create_command_encoder(label=self._label)
        self.encode(command_encoder, nx, ny, nz, uniform_entry)

        # Submit!
        self._device.queue.submit([command_encoder.finish()])

        # Timeit
        if self._report_time:
            self._device._poll_wait()  # wait for the GPU to finish
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0


def create_storage_buffer(device, array: np.ndarray) -> wgpu.GPUBuffer:
    """helper function to create our storage buffers on the GPU"""
    buf = device.create_buffer(
        size=array.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    device.queue.write_buffer(buf, 0, array)
    return buf


@dataclass
class MatrixCSR:
    """organize our CSR buffers"""

    indptr: wgpu.GPUBuffer
    indices: wgpu.GPUBuffer
    values: wgpu.GPUBuffer


class SpMVImage:
    """
    Sparse matrix-vector multiply with the result stored in a Texture that can
    be visualized as a fastplotlib ImageGraphic.

    Parameters
    ----------
    A: scipy.sparse.csr_matrix
        Spatial components, shape is [p, k]

    C: np.ndarray
        dense array of temporal components, shape is [k, T]

    shape: (int, int)
        2D shape of the image that A @ C[:, t] represents, i.e. FOV shape

    scale_factor: np.ndarray
        scale factor image from masknmf

    scale_add: np.ndarray
        scale addition from masknmf

    """

    def __init__(
        self,
        A: scipy.sparse.csr_matrix,
        C: np.ndarray,
        shape: tuple[int, int],
        scale_factor: np.ndarray = None,
        scale_add: np.ndarray = None,
        workgroup_size: int = 32,
        benchmark: bool = False,
        spmv_mode: Literal["scalar", "vector"] = "vector",
    ):
        # get the shapes of things
        p, k = A.shape
        _, T = C.shape
        m, n = shape
        self._workgroup_size = workgroup_size

        if spmv_mode == "vector" and workgroup_size != 32:
            raise ValueError

        self._spmv_mode = spmv_mode

        # we only use scaling for the PMD results
        if scale_factor is None:
            scale_factor = np.ones(shape=(p,), dtype=np.float32)

        if scale_add is None:
            scale_add = np.zeros(shape=(p,), dtype=np.float32)

        device = pygfx.renderers.wgpu.get_shared().device

        self._p, self._T = p, T
        self._m, self._n = m, n

        # create GPU buffers
        self._A = MatrixCSR(
            indptr=create_storage_buffer(
                device, np.ascontiguousarray(A.indptr, np.uint32)
            ),
            indices=create_storage_buffer(
                device, np.ascontiguousarray(A.indices, np.uint32)
            ),
            values=create_storage_buffer(
                device, np.ascontiguousarray(A.data, np.float32)
            ),
        )
        self._C = create_storage_buffer(device, np.ascontiguousarray(C))
        self._scale_factor = create_storage_buffer(device, scale_factor)
        self._scale_add = create_storage_buffer(device, scale_add)

        # fastplotlib TextureArray to write the data into so we can visualize it
        self._texture_array = TextureArray(
            data=np.zeros((m, n), dtype=np.float32),
            cpu_buffer=False,  # buffer only exists on the GPU
            usage=(
                wgpu.TextureUsage.STORAGE_BINDING  # used in compute kernel
                | wgpu.TextureUsage.TEXTURE_BINDING  # used as a texture
                | wgpu.TextureUsage.COPY_SRC  # allow compute kernel to write to it
                | wgpu.TextureUsage.COPY_DST
            ),
        )

        # fastplotlib uses a TextureArray which represents an array of textures, not used here but it's useful for
        # very large images. So we just get the actual texture by indexing the (0, 0) texture in the array
        self._texture = self._texture_array.buffer[0, 0]

        self._benchmark = benchmark
        self._timings = list()

        # create compute shader module, set resources
        self._compute_shader = ComputeShader(
            Path(__file__).parent.joinpath(f"spmv_csr_{self._spmv_mode}.wgsl").read_text(),
            entry_point="spmv_csr",
            report_time=self._benchmark,
        )
        self._compute_shader.set_constant("wg_size", self._workgroup_size)
        self._compute_shader.set_constant("T", T)
        self._compute_shader.set_constant("n_cols", n)
        self._compute_shader.set_resource(0, self._A.indptr)
        self._compute_shader.set_resource(1, self._A.indices)
        self._compute_shader.set_resource(2, self._A.values)
        self._compute_shader.set_resource(3, self._C)
        self._compute_shader.set_resource(5, self._texture)
        self._compute_shader.set_resource(6, self._scale_factor)
        self._compute_shader.set_resource(7, self._scale_add)

        self._t = np.array([0], dtype=np.uint32)

        vmin, vmax = self.estimate_vmin_vmax()

        self._image_graphic = fpl.ImageGraphic(
            self._texture_array, vmin=vmin, vmax=vmax, cmap="viridis"
        )

        self.dispatch()

    @property
    def image_graphic(self) -> fpl.ImageGraphic:
        return self._image_graphic

    def estimate_vmin_vmax(self) -> tuple[float, float]:
        """
        Estimate (vmin, vmax) by computing 10 frames
        """
        n_samples = min(10, self._T - 1)

        if n_samples < 1:
            return 0.0, 1.0

        timepoints = np.unique(
            np.linspace(1, self._T - 1, n_samples).round().astype(int)
        )

        vmin = np.inf
        vmax = -np.inf

        for t in timepoints:
            self.t = t
            frame = self.to_numpy()
            vmin = min(vmin, float(frame.min()))
            vmax = max(vmax, float(frame.max())) / 2

        return vmin, vmax

    def to_numpy(self) -> np.ndarray:
        """
        Download the current image from the GPU and get as a numpy array
        """
        device = pygfx.renderers.wgpu.get_shared().device
        wgpu_texture = pygfx.renderers.wgpu.engine.update.ensure_wgpu_object(
            self._texture
        )

        buf = device.queue.read_texture(
            source={"texture": wgpu_texture, "origin": (0, 0, 0), "mip_level": 0},
            data_layout={"offset": 0, "bytes_per_row": self._n * 4},
            size=(self._n, self._m, 1),
        )

        return np.frombuffer(buf, dtype=np.float32).reshape(self._m, self._n)

    @property
    def t(self) -> int:
        """get or set the current t index vlaue"""
        return self._t[0]

    @t.setter
    def t(self, value: int):
        self._t[0] = int(value)

        self._compute_shader.set_uniform(4, self._t)
        self._timings.append(self.dispatch())

    def dispatch(self):
        """run the compute shader"""

        if self._spmv_mode == "vector":
            n_wg = self._p
            nx = min(65535, n_wg)
            ny = (n_wg + nx - 1) // nx
        else:
            nx = (self._p + self._workgroup_size - 1) // self._workgroup_size
            ny = 1

        return self._compute_shader.dispatch(
            nx, ny, 1
        )

    def get_timings(self):
        if not self._benchmark:
            raise ValueError("Must create with benchmark=True to get stats")

        return np.asarray(self._timings)

    def clear_timings(self):
        self._timings = list()
