from pathlib import Path
from typing import Optional

import numpy as np
import pygfx
import wgpu

from ._spmv import ComputeShader, create_storage_buffer


_F32_EPS = float(np.finfo(np.float32).eps)


class HALS:
    """
    Hierarchical Alternating Least Squares NMF on WGPU.

    Factors ``X ∈ R^{m × n}_+`` as ``X ≈ W H`` with ``W ∈ R^{m × r}_+``,
    ``H ∈ R^{r × n}_+``. Each outer iteration:

    1. Precompute ``P = W^T X`` and ``Q = W^T W``.
    2. Fused H sweep: ``n_inner_H`` full row-sweeps in one dispatch, keeping
       H and Q in workgroup shared memory across all inner sweeps.
    3. Precompute ``R = X H^T`` and ``S = H H^T``  (split-K, two dispatches
       each: gemm_ABT_partial then reduce_ABT).
    4. Fused W sweep: ``n_inner_W`` full column-sweeps in one dispatch.

    Split-K on the ``gemm_ABT`` GEMMs adds parallelism along the reduction
    axis ``n`` so ``gemm_ABT`` doesn't starve when ``m`` (or ``r`` for the
    S GEMM) is smaller than the number of SMs.

    Parameters
    ----------
    X : np.ndarray
        Nonnegative data matrix, shape ``(m, n)``, dtype float32, row-major.
    W_init, H_init : np.ndarray
        Nonnegative initial factors of shape ``(m, r)`` and ``(r, n)``.
    eps : float
        Machine-precision lower bound to avoid zero-locking.
    wg_size : int
        Workgroup size for all kernels; must be a power of two.
    m_tile, rows_per_wg, j_tile : int
        GEMM tile parameters.
    target_wgs : int
        Auto-tune target for total workgroups per gemm_ABT dispatch.
        ``k_chunks`` is picked so ``(m/rows_per_wg) × k_chunks ≥ target_wgs``
        when feasible. Defaults to 256 (~4 workgroups per SM on typical
        dGPUs; more on smaller GPUs).
    """

    def __init__(
        self,
        X: np.ndarray,
        W_init: np.ndarray,
        H_init: np.ndarray,
        eps: float = _F32_EPS,
        wg_size: int = 64,
        m_tile: int = 16,
        rows_per_wg: int = 8,
        j_tile: int = 64,
        target_wgs: int = 256,
    ):
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        m, n = X.shape
        if W_init.ndim != 2 or W_init.shape[0] != m:
            raise ValueError(f"W_init shape {W_init.shape} incompatible with X {X.shape}")
        r = W_init.shape[1]
        if H_init.shape != (r, n):
            raise ValueError(f"H_init shape {H_init.shape} must be ({r}, {n})")
        if wg_size & (wg_size - 1):
            raise ValueError(f"wg_size ({wg_size}) must be a power of two")

        X = np.ascontiguousarray(X, dtype=np.float32)
        W_init = np.ascontiguousarray(W_init, dtype=np.float32)
        H_init = np.ascontiguousarray(H_init, dtype=np.float32)

        self._m, self._n, self._r = m, n, r
        self._eps = float(eps)
        self._wg_size = int(wg_size)
        self._m_tile = int(m_tile)
        self._rows_per_wg = int(rows_per_wg)
        self._j_tile = int(j_tile)
        self._target_wgs = int(target_wgs)

        # k_chunks auto-tuned per side. R = X H^T reduces along n; the "m"
        # dim of gemm_ABT for R is X's row dim (self._m). S = H H^T uses
        # r as the "m" dim.
        self._k_chunks_R = self._pick_k_chunks(m, self._rows_per_wg, target_wgs)
        self._k_chunks_S = self._pick_k_chunks(r, self._rows_per_wg, target_wgs)
        self._chunk_size_R = self._ceil_div(n, self._k_chunks_R)
        self._chunk_size_S = self._ceil_div(n, self._k_chunks_S)

        self._device = pygfx.renderers.wgpu.get_shared().device

        # Keep X on CPU too so error() doesn't need a device readback.
        self._X_np = X

        self._X_buf = create_storage_buffer(self._device, X)
        self._W_buf = self._make_rw_buffer(m * r * 4, initial=W_init.tobytes())
        self._H_buf = self._make_rw_buffer(r * n * 4, initial=H_init.tobytes())
        self._P_buf = self._make_rw_buffer(r * n * 4)
        self._Q_buf = self._make_rw_buffer(r * r * 4)
        self._R_buf = self._make_rw_buffer(m * r * 4)
        self._S_buf = self._make_rw_buffer(r * r * 4)
        # split-K partials, one buffer per side.
        self._partials_R_buf = self._make_rw_buffer(self._k_chunks_R * m * r * 4)
        self._partials_S_buf = self._make_rw_buffer(self._k_chunks_S * r * r * 4)

        self._cur_n_inner_H: Optional[int] = None
        self._cur_n_inner_W: Optional[int] = None

        self._build_kernels()

    @staticmethod
    def _pick_k_chunks(m_out: int, rows_per_wg: int, target_wgs: int) -> int:
        """Pick k_chunks (power of two) so
        (m_out / rows_per_wg) * k_chunks >= target_wgs when feasible."""
        row_strips = (m_out + rows_per_wg - 1) // rows_per_wg
        if row_strips >= target_wgs:
            return 1
        needed = (target_wgs + row_strips - 1) // row_strips
        k = 1
        while k < needed:
            k *= 2
        return k

    def _make_rw_buffer(self, nbytes: int, initial: Optional[bytes] = None) -> wgpu.GPUBuffer:
        buf = self._device.create_buffer(
            size=nbytes,
            usage=(
                wgpu.BufferUsage.STORAGE
                | wgpu.BufferUsage.COPY_SRC
                | wgpu.BufferUsage.COPY_DST
            ),
        )
        if initial is not None:
            self._device.queue.write_buffer(buf, 0, initial)
        return buf

    def _load_wgsl(self, name: str) -> str:
        return Path(__file__).parent.joinpath(f"{name}.wgsl").read_text()

    def _build_kernels(self):
        m, n, r = self._m, self._n, self._r
        wg = self._wg_size

        atb_src = self._load_wgsl("gemm_ATB")
        abt_partial_src = self._load_wgsl("gemm_ABT_partial")
        reduce_abt_src = self._load_wgsl("reduce_ABT")

        # P = W^T X   — gemm_ATB with output shape (r, n).
        self._cs_P = ComputeShader(atb_src, entry_point="gemm_ATB", label="gemm_ATB_P")
        for name, val in (
            ("r", r), ("wg_size", wg), ("m_tile", self._m_tile), ("m", m), ("n", n),
        ):
            self._cs_P.set_constant(name, val)
        self._cs_P.set_resource(0, self._W_buf)
        self._cs_P.set_resource(1, self._X_buf)
        self._cs_P.set_resource(2, self._P_buf)

        # Q = W^T W   — gemm_ATB with output shape (r, r).
        self._cs_Q = ComputeShader(atb_src, entry_point="gemm_ATB", label="gemm_ATB_Q")
        for name, val in (
            ("r", r), ("wg_size", wg), ("m_tile", self._m_tile), ("m", m), ("n", r),
        ):
            self._cs_Q.set_constant(name, val)
        self._cs_Q.set_resource(0, self._W_buf)
        self._cs_Q.set_resource(1, self._W_buf)
        self._cs_Q.set_resource(2, self._Q_buf)

        # R = X H^T pass 1 — gemm_ABT_partial. Output written to partials_R.
        self._cs_R = ComputeShader(
            abt_partial_src, entry_point="gemm_ABT_partial", label="gemm_ABT_R",
        )
        for name, val in (
            ("r", r), ("wg_size", wg), ("rows_per_wg", self._rows_per_wg),
            ("j_tile", self._j_tile), ("m", m), ("n", n),
            ("k_chunks", self._k_chunks_R), ("chunk_size", self._chunk_size_R),
        ):
            self._cs_R.set_constant(name, val)
        self._cs_R.set_resource(0, self._X_buf)
        self._cs_R.set_resource(1, self._H_buf)
        self._cs_R.set_resource(2, self._partials_R_buf)

        # R pass 2 — reduce along k_chunks_R into R.
        self._cs_R_reduce = ComputeShader(
            reduce_abt_src, entry_point="reduce", label="reduce_ABT_R",
        )
        for name, val in (
            ("wg_size", wg), ("m", m), ("r", r), ("k_chunks", self._k_chunks_R),
        ):
            self._cs_R_reduce.set_constant(name, val)
        self._cs_R_reduce.set_resource(0, self._partials_R_buf)
        self._cs_R_reduce.set_resource(1, self._R_buf)

        # S = H H^T pass 1 — same kernel, different dims.
        self._cs_S = ComputeShader(
            abt_partial_src, entry_point="gemm_ABT_partial", label="gemm_ABT_S",
        )
        for name, val in (
            ("r", r), ("wg_size", wg), ("rows_per_wg", self._rows_per_wg),
            ("j_tile", self._j_tile), ("m", r), ("n", n),
            ("k_chunks", self._k_chunks_S), ("chunk_size", self._chunk_size_S),
        ):
            self._cs_S.set_constant(name, val)
        self._cs_S.set_resource(0, self._H_buf)
        self._cs_S.set_resource(1, self._H_buf)
        self._cs_S.set_resource(2, self._partials_S_buf)

        # S pass 2 — reduce along k_chunks_S into S.
        self._cs_S_reduce = ComputeShader(
            reduce_abt_src, entry_point="reduce", label="reduce_ABT_S",
        )
        for name, val in (
            ("wg_size", wg), ("m", r), ("r", r), ("k_chunks", self._k_chunks_S),
        ):
            self._cs_S_reduce.set_constant(name, val)
        self._cs_S_reduce.set_resource(0, self._partials_S_buf)
        self._cs_S_reduce.set_resource(1, self._S_buf)

        self._cs_sweep_H = ComputeShader(
            self._load_wgsl("hals_sweep_H"),
            entry_point="hals_sweep",
            label="hals_sweep_H",
        )
        for name, val in (
            ("r", r), ("wg_size", wg), ("n", n), ("eps", self._eps),
        ):
            self._cs_sweep_H.set_constant(name, val)
        self._cs_sweep_H.set_resource(0, self._P_buf)
        self._cs_sweep_H.set_resource(1, self._Q_buf)
        self._cs_sweep_H.set_resource(2, self._H_buf)

        self._cs_sweep_W = ComputeShader(
            self._load_wgsl("hals_sweep_W"),
            entry_point="hals_sweep",
            label="hals_sweep_W",
        )
        for name, val in (
            ("r", r), ("wg_size", wg), ("m", m), ("eps", self._eps),
        ):
            self._cs_sweep_W.set_constant(name, val)
        self._cs_sweep_W.set_resource(0, self._R_buf)
        self._cs_sweep_W.set_resource(1, self._S_buf)
        self._cs_sweep_W.set_resource(2, self._W_buf)

    def sweep_H(self, n_inner: int = 1) -> None:
        """Compute P, Q then run ``n_inner`` fused H sweeps."""
        self._cs_P.dispatch(self._ceil_div(self._n, self._wg_size), 1, 1)
        self._cs_Q.dispatch(self._ceil_div(self._r, self._wg_size), 1, 1)

        if self._cur_n_inner_H != n_inner:
            self._cs_sweep_H.set_constant("n_inner_sweeps", n_inner)
            self._cur_n_inner_H = n_inner

        self._cs_sweep_H.dispatch(self._ceil_div(self._n, self._wg_size), 1, 1)

    def sweep_W(self, n_inner: int = 1) -> None:
        """Compute R, S then run ``n_inner`` fused W sweeps."""
        # R = X H^T  (split-K, two passes)
        self._cs_R.dispatch(
            self._ceil_div(self._m, self._rows_per_wg), self._k_chunks_R, 1,
        )
        self._cs_R_reduce.dispatch(
            self._ceil_div(self._m * self._r, self._wg_size), 1, 1,
        )
        # S = H H^T  (split-K, two passes)
        self._cs_S.dispatch(
            self._ceil_div(self._r, self._rows_per_wg), self._k_chunks_S, 1,
        )
        self._cs_S_reduce.dispatch(
            self._ceil_div(self._r * self._r, self._wg_size), 1, 1,
        )

        if self._cur_n_inner_W != n_inner:
            self._cs_sweep_W.set_constant("n_inner_sweeps", n_inner)
            self._cur_n_inner_W = n_inner

        self._cs_sweep_W.dispatch(self._ceil_div(self._m, self._wg_size), 1, 1)

    def fit(
        self,
        n_outer: int,
        n_inner_H: int = 1,
        n_inner_W: int = 1,
    ) -> None:
        for _ in range(n_outer):
            self.sweep_H(n_inner_H)
            self.sweep_W(n_inner_W)

    def fit_batched(
        self,
        n_outer: int,
        n_inner_H: int = 1,
        n_inner_W: int = 1,
    ) -> None:
        """
        Record all dispatches for ``n_outer`` iterations inside a single
        command encoder, submit once, ``_poll_wait`` once. Amortizes
        per-dispatch CPU-side overhead across the whole run.

        First call primes pipelines; subsequent calls (with the same
        ``n_inner`` values) are pure record-and-submit.
        """
        needs_prime = (
            self._cur_n_inner_H != n_inner_H
            or self._cur_n_inner_W != n_inner_W
            or self._cs_P._pipeline is None
            or self._cs_sweep_H._pipeline is None
            or self._cs_sweep_W._pipeline is None
        )
        if needs_prime:
            self.sweep_H(n_inner_H)
            self.sweep_W(n_inner_W)
            if n_outer == 0:
                return
            n_outer -= 1

        device = self._device

        n_wg_P = self._ceil_div(self._n, self._wg_size)
        n_wg_Q = self._ceil_div(self._r, self._wg_size)
        n_wg_R = self._ceil_div(self._m, self._rows_per_wg)
        n_wg_S = self._ceil_div(self._r, self._rows_per_wg)
        n_wg_R_reduce = self._ceil_div(self._m * self._r, self._wg_size)
        n_wg_S_reduce = self._ceil_div(self._r * self._r, self._wg_size)
        n_wg_sweep_H = self._ceil_div(self._n, self._wg_size)
        n_wg_sweep_W = self._ceil_div(self._m, self._wg_size)

        encoder = device.create_command_encoder(label="hals_fit_batched")
        pass_ = encoder.begin_compute_pass()
        for _ in range(n_outer):
            pass_.set_pipeline(self._cs_P._pipeline)
            pass_.set_bind_group(0, self._cs_P._bind_group)
            pass_.dispatch_workgroups(n_wg_P, 1, 1)

            pass_.set_pipeline(self._cs_Q._pipeline)
            pass_.set_bind_group(0, self._cs_Q._bind_group)
            pass_.dispatch_workgroups(n_wg_Q, 1, 1)

            pass_.set_pipeline(self._cs_sweep_H._pipeline)
            pass_.set_bind_group(0, self._cs_sweep_H._bind_group)
            pass_.dispatch_workgroups(n_wg_sweep_H, 1, 1)

            # R = X H^T split-K, two passes.
            pass_.set_pipeline(self._cs_R._pipeline)
            pass_.set_bind_group(0, self._cs_R._bind_group)
            pass_.dispatch_workgroups(n_wg_R, self._k_chunks_R, 1)
            pass_.set_pipeline(self._cs_R_reduce._pipeline)
            pass_.set_bind_group(0, self._cs_R_reduce._bind_group)
            pass_.dispatch_workgroups(n_wg_R_reduce, 1, 1)

            # S = H H^T split-K, two passes.
            pass_.set_pipeline(self._cs_S._pipeline)
            pass_.set_bind_group(0, self._cs_S._bind_group)
            pass_.dispatch_workgroups(n_wg_S, self._k_chunks_S, 1)
            pass_.set_pipeline(self._cs_S_reduce._pipeline)
            pass_.set_bind_group(0, self._cs_S_reduce._bind_group)
            pass_.dispatch_workgroups(n_wg_S_reduce, 1, 1)

            pass_.set_pipeline(self._cs_sweep_W._pipeline)
            pass_.set_bind_group(0, self._cs_sweep_W._bind_group)
            pass_.dispatch_workgroups(n_wg_sweep_W, 1, 1)
        pass_.end()
        device.queue.submit([encoder.finish()])
        device._poll_wait()

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        return (a + b - 1) // b

    def get_W(self) -> np.ndarray:
        buf = self._device.queue.read_buffer(self._W_buf)
        return np.frombuffer(buf, dtype=np.float32).reshape(self._m, self._r).copy()

    def get_H(self) -> np.ndarray:
        buf = self._device.queue.read_buffer(self._H_buf)
        return np.frombuffer(buf, dtype=np.float32).reshape(self._r, self._n).copy()

    def error(self) -> float:
        """Relative Frobenius error ``‖X − W H‖_F / ‖X‖_F``."""
        W = self.get_W()
        H = self.get_H()
        return float(np.linalg.norm(self._X_np - W @ H) / np.linalg.norm(self._X_np))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self._m, self._n, self._r)
