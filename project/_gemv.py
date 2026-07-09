import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pygfx
import wgpu

from ._spmv import ComputeShader, create_storage_buffer


KernelName = Literal["row", "strip", "fat", "thread_per_row", "auto"]

# Workgroup size used by every reducing kernel; must be a power of two.
_WG_SIZE = 128


class GEMV:
    """
    Dense matrix-vector multiply on WGPU.

    Computes ``y = A @ v`` on the GPU. Four kernel variants:

    - ``row``            one workgroup per row of ``A``. Good for square-ish
                         shapes.
    - ``strip``          one workgroup handles ``rows_per_wg`` rows sharing a
                         cached tile of ``v``. Good when ``M >> N``.
    - ``fat``            split-K two-pass reduction. Good when ``M`` is small
                         and ``N`` is large so the row-per-workgroup dispatch
                         cannot fill the GPU.
    - ``thread_per_row`` one lane computes one full row's dot product; no
                         reduction. Good when ``M >> N`` with small ``N``,
                         where the reduction cost dominates per-row work.
                         Requires ``A`` stored column-major on the GPU; the
                         driver holds ``A.T`` when this kernel is selected.

    Parameters
    ----------
    A : np.ndarray
        Dense matrix, shape ``[M, N]``, dtype float32, row-major.
    v : np.ndarray
        Vector, shape ``[N]``, dtype float32.
    kernel : {"row", "strip", "fat", "thread_per_row", "auto"}
    rows_per_wg, v_tile_size : int
        strip parameters.
    chunk_size : int
        fat pass-1 columns per workgroup.
    benchmark : bool
        If True, ``dispatch()`` waits for the GPU and appends elapsed ms.
    """

    def __init__(
        self,
        A: np.ndarray,
        v: np.ndarray,
        kernel: KernelName = "auto",
        rows_per_wg: int = 4,
        v_tile_size: int = 1024,
        chunk_size: int = 4096,
        benchmark: bool = False,
    ):
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if v.ndim != 1:
            raise ValueError(f"v must be 1D, got shape {v.shape}")
        if A.shape[1] != v.shape[0]:
            raise ValueError(
                f"A.shape[1] ({A.shape[1]}) must equal v.shape[0] ({v.shape[0]})"
            )

        A = np.ascontiguousarray(A, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)

        self._M, self._N = A.shape
        self._rows_per_wg = int(rows_per_wg)
        self._v_tile_size = int(v_tile_size)
        self._chunk_size = int(chunk_size)
        self._benchmark = benchmark

        self._kernel = self._resolve_kernel(kernel)
        self._device = pygfx.renderers.wgpu.get_shared().device

        # thread_per_row reads A in column-major layout for coalescing. Store
        # A.T (which is column-major A) on the GPU for that kernel.
        if self._kernel == "thread_per_row":
            A_gpu = np.ascontiguousarray(A.T, dtype=np.float32)
        else:
            A_gpu = A
        self._A_buf = create_storage_buffer(self._device, A_gpu)

        self._v_buf = self._device.create_buffer(
            size=v.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self._device.queue.write_buffer(self._v_buf, 0, v)

        self._y_buf = self._device.create_buffer(
            size=self._M * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

        self._partials_buf: Optional[wgpu.GPUBuffer] = None
        self._reduce_shader: Optional[ComputeShader] = None
        self._k_chunks = 0

        self._timings: list[float] = []
        self._compute_shader = self._build_kernel()

    def _resolve_kernel(self, kernel: KernelName) -> str:
        if kernel != "auto":
            if kernel not in ("row", "strip", "fat", "thread_per_row"):
                raise ValueError(f"unknown kernel {kernel!r}")
            return kernel
        # Refine after benchmarks; crossovers are hardware-dependent.
        if self._M < 512:
            return "fat"
        if self._N <= 128 and self._M > 16 * self._N:
            return "thread_per_row"
        if self._M > 16 * self._N:
            return "strip"
        return "row"

    def _load_wgsl(self, name: str) -> str:
        return Path(__file__).parent.joinpath(f"{name}.wgsl").read_text()

    def _build_kernel(self) -> ComputeShader:
        if self._kernel == "row":
            cs = ComputeShader(
                self._load_wgsl("gemv_row"),
                entry_point="gemv",
                report_time=self._benchmark,
                label="gemv_row",
            )
            cs.set_constant("wg_size", _WG_SIZE)
            cs.set_constant("N", self._N)
            cs.set_resource(0, self._A_buf)
            cs.set_resource(1, self._v_buf)
            cs.set_resource(2, self._y_buf)
            return cs

        if self._kernel == "strip":
            cs = ComputeShader(
                self._load_wgsl("gemv_strip"),
                entry_point="gemv",
                report_time=self._benchmark,
                label="gemv_strip",
            )
            cs.set_constant("wg_size", _WG_SIZE)
            cs.set_constant("N", self._N)
            cs.set_constant("rows_per_wg", self._rows_per_wg)
            cs.set_constant("v_tile_size", self._v_tile_size)
            cs.set_resource(0, self._A_buf)
            cs.set_resource(1, self._v_buf)
            cs.set_resource(2, self._y_buf)
            return cs

        if self._kernel == "thread_per_row":
            cs = ComputeShader(
                self._load_wgsl("gemv_thread_per_row"),
                entry_point="gemv",
                report_time=self._benchmark,
                label="gemv_thread_per_row",
            )
            cs.set_constant("wg_size", _WG_SIZE)
            cs.set_constant("N", self._N)
            cs.set_resource(0, self._A_buf)
            cs.set_resource(1, self._v_buf)
            cs.set_resource(2, self._y_buf)
            return cs

        # fat: two-pass split-K
        self._k_chunks = (self._N + self._chunk_size - 1) // self._chunk_size
        self._partials_buf = self._device.create_buffer(
            size=self._M * self._k_chunks * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )

        cs = ComputeShader(
            self._load_wgsl("gemv_fat_partial"),
            entry_point="gemv_partial",
            report_time=self._benchmark,
            label="gemv_fat_partial",
        )
        cs.set_constant("wg_size", _WG_SIZE)
        cs.set_constant("N", self._N)
        cs.set_constant("chunk_size", self._chunk_size)
        cs.set_constant("K_chunks", self._k_chunks)
        cs.set_resource(0, self._A_buf)
        cs.set_resource(1, self._v_buf)
        cs.set_resource(2, self._partials_buf)

        reduce_cs = ComputeShader(
            self._load_wgsl("reduce_partials"),
            entry_point="reduce",
            report_time=self._benchmark,
            label="reduce_partials",
        )
        reduce_cs.set_constant("wg_size", _WG_SIZE)
        reduce_cs.set_constant("K_chunks", self._k_chunks)
        reduce_cs.set_resource(0, self._partials_buf)
        reduce_cs.set_resource(1, self._y_buf)
        self._reduce_shader = reduce_cs
        return cs

    def set_v(self, new_v: np.ndarray) -> None:
        new_v = np.ascontiguousarray(new_v, dtype=np.float32)
        if new_v.shape != (self._N,):
            raise ValueError(f"v must have shape ({self._N},), got {new_v.shape}")
        self._device.queue.write_buffer(self._v_buf, 0, new_v)
        self.dispatch()

    def _primary_geom(self) -> tuple[int, int, int]:
        if self._kernel == "row":
            n_wg = self._M
            nx = min(65535, n_wg)
            ny = (n_wg + nx - 1) // nx
            return (nx, ny, 1)
        if self._kernel == "strip":
            n_wg = (self._M + self._rows_per_wg - 1) // self._rows_per_wg
            nx = min(65535, n_wg)
            ny = (n_wg + nx - 1) // nx
            return (nx, ny, 1)
        if self._kernel == "thread_per_row":
            n_wg = (self._M + _WG_SIZE - 1) // _WG_SIZE
            nx = min(65535, n_wg)
            ny = (n_wg + nx - 1) // nx
            return (nx, ny, 1)
        return (self._k_chunks, self._M, 1)

    def dispatch(self) -> Optional[float]:
        nx, ny, nz = self._primary_geom()
        if self._reduce_shader is None:
            elapsed = self._compute_shader.dispatch(nx, ny, nz)
        else:
            t1 = self._compute_shader.dispatch(nx, ny, nz)
            t2 = self._reduce_shader.dispatch(self._M, 1, 1)
            elapsed = None if (t1 is None or t2 is None) else (t1 + t2)

        if elapsed is not None:
            self._timings.append(elapsed)
        return elapsed

    def dispatch_batch(self, count: int) -> float:
        """
        Submit ``count`` dispatches inside a single command encoder and one
        _poll_wait. Returns the amortized per-iter wall-clock time in ms.

        This isolates GPU throughput from per-dispatch CPU-side overhead
        (encoder creation, submit, poll) which dominates individual dispatch
        latency for short kernels.
        """
        # Prime pipelines so _pipeline / _bind_group exist on both shaders.
        self.dispatch()

        device = self._device
        cs = self._compute_shader
        reduce = self._reduce_shader
        primary_dims = self._primary_geom()
        reduce_dims = (self._M, 1, 1)

        encoder = device.create_command_encoder(label="gemv_batch")
        pass_ = encoder.begin_compute_pass()

        t0 = time.perf_counter()
        for _ in range(count):
            pass_.set_pipeline(cs._pipeline)
            pass_.set_bind_group(0, cs._bind_group)
            pass_.dispatch_workgroups(*primary_dims)
            if reduce is not None:
                pass_.set_pipeline(reduce._pipeline)
                pass_.set_bind_group(0, reduce._bind_group)
                pass_.dispatch_workgroups(*reduce_dims)
        pass_.end()
        device.queue.submit([encoder.finish()])
        device._poll_wait()
        total_ms = (time.perf_counter() - t0) * 1000.0
        return total_ms / count

    def to_numpy(self) -> np.ndarray:
        buf = self._device.queue.read_buffer(self._y_buf)
        return np.frombuffer(buf, dtype=np.float32).reshape(self._M).copy()

    def get_timings(self) -> np.ndarray:
        if not self._benchmark:
            raise ValueError("Must construct with benchmark=True to get timings")
        return np.asarray(self._timings)

    def clear_timings(self) -> None:
        self._timings = []

    @property
    def kernel(self) -> str:
        return self._kernel

    @property
    def shape(self) -> tuple[int, int]:
        return (self._M, self._N)
