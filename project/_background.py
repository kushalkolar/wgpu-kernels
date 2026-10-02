"""
masknmf's fluctuating_baseline_update on the GPU: the ring model of the fluctuating background, the factorized ring
term (q0, q1) of the background U q0 q1 that HALS reads, and the multiunit factorization of the residual at the end of a
pass, see FluctuatingBaseline
"""

import numpy as np
import pygfx
import torch
import wgpu

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    SignalBuffers,
    load_shader,
    set_constants_and_resources,
    create_buffer,
    create_empty_buffer,
    read_buffer,
    dispatch_grid,
    round_up,
)


def ring_taps(radius: int) -> np.ndarray:
    """[n_taps, 2] int32 (dy, dx) of masknmf's ring kernel, the pixels at distance radius +- 0.5"""
    r = np.arange(-radius, radius + 1)
    dy, dx = np.meshgrid(r, r, indexing="ij")
    dist = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2)
    ring = (dist >= radius - 0.5) & (dist <= radius + 0.5)
    return np.stack([dy[ring], dx[ring]], axis=1).astype(np.int32)


def downsample_bins(pixels: np.ndarray, fov_shape: tuple[int, int], factor: int) -> np.ndarray:
    """bin of each pixel of masknmf's spatial downsampling by factor x factor bins, -1 beyond the last full bin"""
    height, width = fov_shape
    rows, cols = pixels // width, pixels % width
    w = width // factor
    inside = (rows // factor < height // factor) & (cols // factor < w)
    return np.where(inside, (rows // factor) * w + cols // factor, -1)


class FluctuatingBaseline(GPUComputation):
    """
    masknmf's fluctuating_baseline_update: a randomized SVD of the downsampled residual UV - a c^T (b is 0 in
    masknmf's demix loop) gives a temporal basis of the background, the residual projected onto it at full resolution
    gives the spatial part, which the ring model (the sum over a ring of pixels around each pixel, with per-pixel
    weights) turns into the factorized ring term, projected onto the spatial basis with U's local projector.

    The downsampled movie of U V is computed once. The QR and the SVDs of masknmf's lowrank_background_svd are
    computed from Gram matrices on the GPU and factorizations of (rank + 5) x (rank + 5) matrices on the CPU, as are
    those of the randomized SVD of the full residual for the multiunit factorization (``get_multiunit_factorization``).

    Parameters
    ----------
    compression: CompressionBuffers

    u: torch.Tensor
        sparse COO [n_pixels, rank], U, pixels in row-major order

    downsampling_factor, ring_radius, background_sketch:
        masknmf's parameters of the same names
    """

    def __init__(
        self,
        compression: CompressionBuffers,
        u: torch.Tensor,
        downsampling_factor: int = 20,
        ring_radius: int = 10,
        background_sketch: int = 300,
    ):
        super().__init__()
        device = pygfx.renderers.wgpu.get_shared().device
        self._compression = compression
        self._factor = downsampling_factor
        self._radius = ring_radius
        self._background_sketch = background_sketch

        height, width = compression.fov_shape
        self._n_bins = (height // downsampling_factor) * (width // downsampling_factor)
        # rows of the downsampled movie, a multiple of 4 so that they can be the reduction dimension of a GEMM
        self._n_bins4 = round_up(self._n_bins, 4)
        self._n4 = compression.n_frames_padded // 4

        # U downsampled as masknmf's downsample_sparse, CSR over the bins
        u = u.coalesce()
        pixels, cols = u.indices().cpu().numpy()
        values = u.values().cpu().numpy().astype(np.float64)
        bins = downsample_bins(pixels, compression.fov_shape, downsampling_factor)
        keep = bins >= 0
        keys, inverse = np.unique(bins[keep] * compression.rank + cols[keep], return_inverse=True)
        u_ds_values = np.zeros(keys.size)
        np.add.at(u_ds_values, inverse, values[keep] / downsampling_factor**2)
        u_ds_ptr = np.searchsorted(keys // compression.rank, np.arange(self._n_bins4 + 1))

        # the downsampled movie of U V, [n_bins4, n_frames_padded]
        self._movie_ds = create_empty_buffer(device, 16 * self._n_bins4 * self._n4)
        shader = load_shader("rows_combination.wgsl")
        set_constants_and_resources(
            shader,
            {"n4": self._n4},
            {
                0: create_buffer(device, u_ds_ptr.astype(np.uint32)),
                1: create_buffer(device, (keys % compression.rank).astype(np.uint32)),
                2: create_buffer(device, u_ds_values.astype(np.float32)),
                3: compression.temporal_compressed,
                4: create_empty_buffer(device, 16 * self._n_bins4 * self._n4),
                5: self._movie_ds,
            },
        )
        shader.dispatch(-(-self._n4 // 256), self._n_bins4)

        # the tile columns of each column of U, for col_reduce
        _, cell_cols = compression.get_cell_structure()
        col_tiles = np.argsort(cell_cols, kind="stable")
        col_tile_ptr = np.searchsorted(cell_cols[col_tiles], np.arange(compression.rank + 1))
        self._col_tiles = create_buffer(device, col_tiles.astype(np.uint32))
        self._col_tile_ptr = create_buffer(device, col_tile_ptr.astype(np.uint32))
        self._n_tile_cols = cell_cols.size

        taps = ring_taps(ring_radius)
        self._taps = create_buffer(device, taps)
        self._n_taps = taps.shape[0]

        self._ones = create_buffer(device, np.ones(height * width, dtype=np.float32))
        self._seed = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # the factorized ring term, reallocated when the padded background rank changes
        self._ring_rank_padded = None
        self._ring_term = None
        self._signals = None

    def set_signals(self, signals: SignalBuffers):
        """the (bin, signal) pairs of the downsampled footprints, recomputed when the support of a changes"""
        device = pygfx.renderers.wgpu.get_shared().device
        s = signals.structures
        a_pixels = s["a_pixels"].astype(np.int64)
        a_signals = np.repeat(np.arange(signals.n_signals), np.diff(s["a_ptr"]))
        bins = downsample_bins(a_pixels, self._compression.fov_shape, self._factor)
        entries = np.flatnonzero(bins >= 0)
        keys = bins[entries] * signals.n_signals + a_signals[entries]
        order = np.argsort(keys, kind="stable")
        pair_keys, entry_ptr = np.unique(keys[order], return_index=True)
        self._n_pairs = pair_keys.size
        self._pair_buffers = {
            "entry_ptr": create_buffer(device, np.append(entry_ptr, order.size).astype(np.uint32)),
            "entries": create_buffer(device, entries[order].astype(np.uint32)),
            "bin_ptr": create_buffer(
                device,
                np.searchsorted(pair_keys // signals.n_signals, np.arange(self._n_bins4 + 1)).astype(np.uint32),
            ),
            "pair_signals": create_buffer(device, (pair_keys % signals.n_signals).astype(np.uint32)),
            "pair_values": create_empty_buffer(device, 4 * self._n_pairs),
        }
        self._signals = signals

    def _encode_movie(self, encoder: wgpu.GPUCommandEncoder):
        """the downsampled residual M = ds(U V) - ds(a) c^T and its transpose"""
        signals = self._signals
        p = self._pair_buffers
        shader = self._shader("a_downsample", self._n_pairs)
        set_constants_and_resources(
            shader,
            {"n_pairs": self._n_pairs, "scale": -1.0 / self._factor**2},
            {0: p["entry_ptr"], 1: p["entries"], 2: signals.buffers["a_values"], 3: p["pair_values"]},
        )
        shader.encode(encoder, *dispatch_grid(max(-(-self._n_pairs // 256), 1)))

        movie = self._buffer("movie", 16 * self._n_bins4 * self._n4)
        shader = self._shader("rows_combination", "residual")
        set_constants_and_resources(
            shader,
            {"n4": self._n4},
            {
                0: p["bin_ptr"],
                1: p["pair_signals"],
                2: p["pair_values"],
                3: signals.buffers["temporal_demixed"],
                4: self._movie_ds,
                5: movie,
            },
        )
        shader.encode(encoder, -(-self._n4 // 256), self._n_bins4)

        movie_t = self._buffer("movie_t", 16 * self._n_bins4 * self._n4)
        self._encode_transpose(encoder, "movie", movie, self._n_bins4, 4 * self._n4, movie_t, self._n_bins4)

    def _gram_eig(
        self, name: str, x: wgpu.GPUBuffer, n_rows: int, n_cols: int, k: int, encoder: wgpu.GPUCommandEncoder
    ) -> tuple[np.ndarray, np.ndarray]:
        """eigenvalues (descending) and eigenvectors of the Gram matrix of the first k rows of x [n_rows, n_cols],
        submits the encoder"""
        gram = self._buffer(f"{name}_gram", 4 * n_rows * n_rows)
        g = self._gemm(f"{name}_gram")
        g.set_arguments(x, x, gram, n_rows, n_rows, n_cols)
        g.encode(encoder)
        self._submit(encoder)
        h = read_buffer(gram, np.float32, (n_rows, n_rows))[:k, :k].astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh((h + h.T) / 2)
        return eigenvalues[::-1], eigenvectors[:, ::-1]

    def _encode_omega(
        self, encoder: wgpu.GPUCommandEncoder, k: int, k8: int, omega: np.ndarray | None, seed: int
    ) -> wgpu.GPUBuffer:
        """the random matrix Omega^T [k8, n_frames_padded], zero beyond k and the frames: standard normal on the GPU
        from seed, or omega [n_frames, k]"""
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        n_frames_padded = comp.n_frames_padded
        omega_t = self._buffer("omega_t", 4 * k8 * n_frames_padded)
        if omega is None:
            device.queue.write_buffer(self._seed, 0, np.array([seed, 0, 0, 0], dtype=np.uint32))
            shader = self._shader("normal", k8)
            set_constants_and_resources(
                shader,
                {"n_rows": k8, "n_cols": n_frames_padded, "n_used": k, "n_cols_used": comp.n_frames},
                {0: omega_t, 1: self._seed},
            )
            shader.encode(encoder, *dispatch_grid(-(-k8 * n_frames_padded // 256)))
        else:
            padded = np.zeros((k8, n_frames_padded), dtype=np.float32)
            padded[:k, : comp.n_frames] = omega.T
            device.queue.write_buffer(omega_t, 0, padded)
        return omega_t

    def _svd(self, k: int, omega: np.ndarray | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        masknmf's lowrank_background_svd with k = rank + oversamples random columns, up to the SVD of the residual
        projected onto the temporal basis v_bkgd, L = (U V - a c^T) v_bkgd^T, whose transpose is left in the buffer
        "left_t", [k8 = k rounded up to 8, n_pixels].

        Returns the squared singular values of L (descending), its right singular vectors W [k, k] and G'
        [k8, n_bins4] with v_bkgd = G' M
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        signals = self._signals
        n_frames_padded = comp.n_frames_padded
        k8 = round_up(k, 8)

        encoder = self._new_encoder()
        omega_t = self._encode_omega(encoder, k, k8, omega, seed)

        # Y = M Omega and its QR on the CPU
        y = self._buffer("y", 4 * self._n_bins4 * k8)
        g = self._gemm("y")
        g.set_arguments(self._buffers["movie"], omega_t, y, self._n_bins4, k8, n_frames_padded)
        g.encode(encoder)
        self._submit(encoder)
        q, _ = np.linalg.qr(read_buffer(y, np.float32, (self._n_bins4, k8))[: self._n_bins, :k].astype(np.float64))
        k_used = q.shape[1]
        q_t = np.zeros((k8, self._n_bins4), dtype=np.float32)
        q_t[:k_used, : self._n_bins] = q.T
        q_t_buffer = self._buffer("q_t", q_t.nbytes)
        device.queue.write_buffer(q_t_buffer, 0, q_t)

        # the right term Q^T M and its Gram matrix: v_bkgd = S^-1 W^T Q^T M = G' M. Directions that a float32 Gram
        # matrix cannot resolve are dropped
        encoder = self._new_encoder()
        right = self._buffer("right", 4 * k8 * n_frames_padded)
        g = self._gemm("right")
        g.set_arguments(q_t_buffer, self._buffers["movie_t"], right, k8, n_frames_padded, self._n_bins4)
        g.encode(encoder)
        eigenvalues, w = self._gram_eig("right", right, k8, n_frames_padded, k_used, encoder)
        resolved = eigenvalues > eigenvalues[0] * np.finfo(np.float32).eps
        inv_s = np.where(resolved, 1.0 / np.sqrt(np.where(resolved, eigenvalues, 1.0)), 0.0)
        g_prime = np.zeros((k8, self._n_bins4), dtype=np.float32)
        g_prime[:k_used, : self._n_bins] = (inv_s[:, None] * w.T) @ q.T
        g_prime_buffer = self._buffer("g_prime", g_prime.nbytes)
        device.queue.write_buffer(g_prime_buffer, 0, g_prime)

        # v_bkgd, phi = V v_bkgd^T, psi = c v_bkgd^T, L^T = (U phi - a psi)^T and its Gram matrix
        encoder = self._new_encoder()
        v_bkgd = self._buffer("v_bkgd", 4 * k8 * n_frames_padded)
        g = self._gemm("v_bkgd")
        g.set_arguments(g_prime_buffer, self._buffers["movie_t"], v_bkgd, k8, n_frames_padded, self._n_bins4)
        g.encode(encoder)
        phi = self._buffer("phi", 4 * comp.rank * k8)
        g = self._gemm("phi")
        g.set_arguments(comp.temporal_compressed, v_bkgd, phi, comp.rank, k8, n_frames_padded)
        g.encode(encoder)
        psi = self._buffer("psi", 4 * signals.n_signals * k8)
        g = self._gemm("psi")
        g.set_arguments(signals.buffers["temporal_demixed"], v_bkgd, psi, signals.n_signals, k8, n_frames_padded)
        g.encode(encoder)
        n_pixels = comp.fov_shape[0] * comp.fov_shape[1]
        left_t = self._buffer("left_t", 4 * k8 * n_pixels)
        self._encode_cell_spmm(encoder, phi, psi, left_t, k8, with_signals=True)
        eigenvalues, w = self._gram_eig("left", left_t, k8, n_pixels, k_used, encoder)
        return eigenvalues, w, g_prime

    def _encode_cell_spmm(self, encoder, phi, psi, out, n_j: int, with_signals: bool):
        """out [n_j, n_pixels] = (U phi - a psi)^T, the a term only if with_signals"""
        comp = self._compression
        tiles = comp.spatial_compressed
        s = self._signals.buffers
        shader = self._shader("cell_spmm", (n_j, with_signals))
        set_constants_and_resources(
            shader,
            {
                "cell_size": comp.cell_size,
                "n_cells_x": comp.n_cells[1],
                "n_cells_y": comp.n_cells[0],
                "max_cell_cols": comp.max_cell_cols,
                "n_j": n_j,
                "with_signals": with_signals,
            },
            {
                0: tiles.cell_col_ptr,
                1: tiles.cell_cols,
                2: tiles.tiles,
                3: phi,
                4: s["pixel_ptr"],
                5: s["pixel_entries"],
                6: s["pixel_signals"],
                7: s["a_values"],
                8: psi,
                9: out,
            },
        )
        shader.encode(encoder, n_j // 8, comp.n_cells[0] * comp.n_cells[1])

    def _encode_projection(self, encoder, tiles, x, scale, n_j: int, n_out: int, out_stride: int, out, name: str):
        """out [rank, out_stride] = T^T (scale * x^T) in columns :n_out, 0 in the others, x is [n_j, n_pixels] and T
        U or its local projector, whose cell tiles are tiles"""
        comp = self._compression
        partial = self._buffer(f"{name}_partial", 4 * self._n_tile_cols * n_j)
        shader = self._shader("cell_spmm_t", n_j)
        set_constants_and_resources(
            shader,
            {"cell_size": comp.cell_size, "n_cells_x": comp.n_cells[1], "n_cells_y": comp.n_cells[0], "n_j": n_j},
            {0: tiles.cell_col_ptr, 1: tiles.tiles, 2: x, 3: scale, 4: partial},
        )
        shader.encode(encoder, -(-n_j // 32), comp.n_cells[0] * comp.n_cells[1])
        shader = self._shader("col_reduce", (n_j, n_out, out_stride))
        set_constants_and_resources(
            shader,
            {"rank": comp.rank, "n_j": n_j, "n_out": n_out, "out_stride": out_stride},
            {0: self._col_tile_ptr, 1: self._col_tiles, 2: partial, 3: out},
        )
        shader.encode(encoder, *dispatch_grid(-(-comp.rank * out_stride // 256)))

    def update(self, background_rank: int | None, seed: int = 0, omegas: list[np.ndarray] | None = None) -> int:
        """
        masknmf's fluctuating_baseline_update for the signals of ``set_signals``, sets their factorized ring term
        (see ``SignalBuffers.set_ring_term_buffers``): a HALS that uses them needs ``HALS.set_signals`` when their
        ``ring_rank_padded`` changed. The buffers are reused while it does not change.

        background_rank None estimates it first from background_sketch + 5 columns, as masknmf does at the start
        and after each support update. The random matrices are drawn on the GPU from seed, or given as omegas
        [n_frames, k], one per SVD, for comparisons with masknmf.

        Returns the background rank.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        omegas = None if omegas is None else list(omegas)

        encoder = self._new_encoder()
        self._encode_movie(encoder)
        self._submit(encoder)

        if background_rank is None:
            eigenvalues, _, _ = self._svd(self._background_sketch + 5, None if omegas is None else omegas.pop(0), seed)
            s2 = np.maximum(eigenvalues[: self._background_sketch], 0.0)
            background_rank = int(np.argmax(np.cumsum(s2) / np.sum(s2) >= 0.99)) + 1
            seed += 1

        r = background_rank
        eigenvalues, w, g_prime = self._svd(r + 5, None if omegas is None else omegas.pop(0), seed)
        k_used = w.shape[0]
        k8 = round_up(r + 5, 8)
        rrp = round_up(r, 4)
        r8 = round_up(r, 8)
        # B = W[:, :r]: u_bkgd s_bkgd = L B and q1 = B^T v_bkgd = (B^T G') M
        b_t = np.zeros((r8, k8), dtype=np.float32)
        b_t[:r, :k_used] = w[:, :r].T
        b_t_buffer = self._buffer("b_t", b_t.nbytes)
        device.queue.write_buffer(b_t_buffer, 0, b_t)
        bg = np.zeros((rrp, self._n_bins4), dtype=np.float32)
        bg[:r] = w[:, :r].T @ g_prime[:k_used].astype(np.float64)
        bg_buffer = self._buffer("bg", bg.nbytes)
        device.queue.write_buffer(bg_buffer, 0, bg)

        if rrp != self._ring_rank_padded:
            self._ring_term = (
                create_empty_buffer(device, 4 * comp.rank * rrp),
                create_empty_buffer(device, 4 * rrp * comp.n_frames_padded),
                create_empty_buffer(device, 4 * rrp * comp.n_frames_padded),
            )
            self._ring_rank_padded = rrp
        q0, q1, q1_t = self._ring_term

        encoder = self._new_encoder()
        n_pixels = comp.fov_shape[0] * comp.fov_shape[1]
        # Z = U_proj^T L B, masknmf's projection of u_bkgd * s_bkgd onto the spatial basis
        p1 = self._buffer("p1", 4 * comp.rank * k8)
        projector = comp.spatial_compressed_local_projector
        self._encode_projection(encoder, projector, self._buffers["left_t"], self._ones, k8, k8, k8, p1, "p1")
        z = self._buffer("z", 4 * comp.rank * r8)
        g = self._gemm("z")
        g.set_arguments(p1, b_t_buffer, z, comp.rank, r8, k8)
        g.encode(encoder)
        # X^T = (U Z)^T, the ring model and its weights, q0 = U_proj^T (weights * wx)
        x_t = self._buffer("x_t", 4 * r8 * n_pixels)
        self._encode_cell_spmm(encoder, z, z, x_t, r8, with_signals=False)
        wx_t = self._buffer("wx_t", 4 * r8 * n_pixels)
        weights = self._buffer("weights", 4 * n_pixels)
        height, width = comp.fov_shape
        shader = self._shader("ring", r8)
        set_constants_and_resources(
            shader,
            {"height": height, "width": width, "n_j": r8, "n_taps": self._n_taps, "radius": self._radius},
            {0: x_t, 1: wx_t, 2: weights, 3: self._taps},
        )
        shader.encode(encoder, -(-width // 16), -(-height // 16))
        self._encode_projection(encoder, projector, wx_t, weights, r8, r, rrp, q0, "q0")
        g = self._gemm("q1")
        g.set_arguments(bg_buffer, self._buffers["movie_t"], q1, rrp, comp.n_frames_padded, self._n_bins4)
        g.encode(encoder)
        g = self._gemm("q1_t")
        g.set_arguments(self._buffers["movie_t"], bg_buffer, q1_t, comp.n_frames_padded, rrp, self._n_bins4)
        g.encode(encoder)
        self._submit(encoder)

        self._signals.set_ring_term_buffers(q0, q1, q1_t, r)
        return r

    def get_multiunit_factorization(
        self, test_rank: int = 200, num_oversamples: int = 5, seed: int = 0, omega: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        masknmf's extract_multiunit_factorization at the end of a pass, for the signals of ``set_signals`` with their
        ring term: a randomized SVD of the residual R = U V - a c^T - U q0 q1 (b is 0 in masknmf's demix loop) with
        test_rank + num_oversamples random columns, each pixel's row of R Omega centered as masknmf does, truncated to
        the rank whose singular values explain 99% of the variance of the first test_rank. The QR and the SVD are
        computed from Gram matrices on the GPU and factorizations on the CPU. The random matrix is drawn on the GPU
        from seed, or given as omega [n_frames, test_rank + num_oversamples].

        Returns masknmf's multiunit_basis_term1 [rank, r], the left singular vectors projected onto the spatial basis
        with U's local projector, and multiunit_basis_term2 [r, n_frames], the right singular vectors times the
        singular values.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        b = self._signals.buffers
        n = self._signals.n_signals
        n_frames_padded = comp.n_frames_padded
        n_pixels = comp.fov_shape[0] * comp.fov_shape[1]
        rank4 = round_up(comp.rank, 4)
        n4 = round_up(n, 4)
        rrp = self._signals.ring_rank_padded
        k = test_rank + num_oversamples
        k8 = round_up(k, 8)

        # Y = R Omega: Y^T = (U phi - a psi)^T with phi = V Omega - q0 q1 Omega and psi = c Omega, and its Gram matrix
        encoder = self._new_encoder()
        omega_t = self._encode_omega(encoder, k, k8, omega, seed)
        phi = self._buffer("multiunit_phi", 4 * comp.rank * k8)
        g = self._gemm("multiunit_phi")
        g.set_arguments(comp.temporal_compressed, omega_t, phi, comp.rank, k8, n_frames_padded)
        g.encode(encoder)
        if rrp > 0:
            q1_omega_t = self._buffer("multiunit_q1_omega_t", 4 * k8 * rrp)
            g = self._gemm("multiunit_q1_omega_t")
            g.set_arguments(omega_t, b["ring_right"], q1_omega_t, k8, rrp, n_frames_padded)
            g.encode(encoder)
            ring_phi = self._buffer("multiunit_ring_phi", 4 * comp.rank * k8)
            g = self._gemm("multiunit_ring_phi")
            g.set_arguments(b["ring_left"], q1_omega_t, ring_phi, comp.rank, k8, rrp)
            g.encode(encoder)
            phi_prime = self._buffer("multiunit_phi_prime", 4 * comp.rank * k8)
            self._encode_axpy(encoder, "multiunit_phi", phi, ring_phi, phi_prime, comp.rank * k8, 1)
            phi = phi_prime
        psi = self._buffer("multiunit_psi", 4 * n * k8)
        g = self._gemm("multiunit_psi")
        g.set_arguments(b["temporal_demixed"], omega_t, psi, n, k8, n_frames_padded)
        g.encode(encoder)
        y_t = self._buffer("multiunit_y_t", 4 * k8 * n_pixels)
        self._encode_cell_spmm(encoder, phi, psi, y_t, k8, with_signals=True)
        y_gram = self._buffer("multiunit_y_gram", 4 * k8 * k8)
        g = self._gemm("multiunit_y_gram")
        g.set_arguments(y_t, y_t, y_gram, k8, k8, n_pixels)
        g.encode(encoder)

        # P = U^T Y, U_proj^T Y [rank, k8] and S = a^T Y [n, k8]
        p = self._buffer("multiunit_p", 4 * comp.rank * k8)
        self._encode_projection(encoder, comp.spatial_compressed, y_t, self._ones, k8, k8, k8, p, "multiunit_p")
        p_proj = self._buffer("multiunit_p_proj", 4 * comp.rank * k8)
        projector = comp.spatial_compressed_local_projector
        self._encode_projection(encoder, projector, y_t, self._ones, k8, k8, k8, p_proj, "multiunit_p_proj")
        s_y = self._buffer("multiunit_s", 4 * n * k8)
        shader = self._shader("a_spmm_t", k8)
        set_constants_and_resources(
            shader,
            {"n_pixels": n_pixels, "n_j": k8},
            {0: b["a_ptr"], 1: b["a_pixels"], 2: b["a_values"], 3: y_t, 4: s_y},
        )
        shader.set_uniform(5, np.array([n, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, -(-k8 // 256), n)

        # X = Y^T R = P^T V - (P^T q0) q1 - S^T c [k8, n_frames_padded] and its Gram matrix. The contractions over the
        # rows of V, q0 and c are GEMMs of their transposes, padded to multiples of 4
        p_t = self._buffer("multiunit_p_t", 4 * k8 * rank4)
        encoder.clear_buffer(p_t)
        self._encode_transpose(encoder, "multiunit_p", p, comp.rank, k8, p_t, rank4)
        v_t = create_empty_buffer(device, 4 * n_frames_padded * rank4)
        self._encode_transpose(encoder, "v", comp.temporal_compressed, comp.rank, n_frames_padded, v_t, rank4)
        x = self._buffer("multiunit_x", 4 * k8 * n_frames_padded)
        g = self._gemm("multiunit_x")
        g.set_arguments(p_t, v_t, x, k8, n_frames_padded, rank4)
        g.encode(encoder)
        if rrp > 0:
            q0_t = self._buffer("multiunit_q0_t", 4 * rrp * rank4)
            encoder.clear_buffer(q0_t)
            self._encode_transpose(encoder, "q0", b["ring_left"], comp.rank, rrp, q0_t, rank4)
            z_t = self._buffer("multiunit_z_t", 4 * k8 * rrp)
            g = self._gemm("multiunit_z_t")
            g.set_arguments(p_t, q0_t, z_t, k8, rrp, rank4)
            g.encode(encoder)
            ring_x = self._buffer("multiunit_ring_x", 4 * k8 * n_frames_padded)
            g = self._gemm("multiunit_ring_x")
            g.set_arguments(z_t, b["ring_right_t"], ring_x, k8, n_frames_padded, rrp)
            g.encode(encoder)
            x_ring = self._buffer("multiunit_x_ring", 4 * k8 * n_frames_padded)
            self._encode_axpy(encoder, "multiunit_x_ring", x, ring_x, x_ring, k8 * n_frames_padded, 1)
            x = x_ring
        s_t = self._buffer("multiunit_s_t", 4 * k8 * n4)
        encoder.clear_buffer(s_t)
        self._encode_transpose(encoder, "multiunit_s", s_y, n, k8, s_t, n4)
        c_t = self._buffer("multiunit_c_t", 4 * n_frames_padded * n4)
        encoder.clear_buffer(c_t)
        self._encode_transpose(encoder, "c", b["temporal_demixed"], n, n_frames_padded, c_t, n4)
        signal_x = self._buffer("multiunit_signal_x", 4 * k8 * n_frames_padded)
        g = self._gemm("multiunit_signal_x")
        g.set_arguments(s_t, c_t, signal_x, k8, n_frames_padded, n4)
        g.encode(encoder)
        x_signals = self._buffer("multiunit_x_signals", 4 * k8 * n_frames_padded)
        self._encode_axpy(encoder, "multiunit_x_signals", x, signal_x, x_signals, k8 * n_frames_padded, 1)
        x = x_signals
        x_gram = self._buffer("multiunit_x_gram", 4 * k8 * k8)
        g = self._gemm("multiunit_x_gram")
        g.set_arguments(x, x, x_gram, k8, k8, n_frames_padded)
        g.encode(encoder)
        self._submit(encoder)

        # Q = Y_c A_q with Y_c = Y C the centered rows and A_q = C W L^-1/2 from the Gram matrix of Y_c, masknmf's
        # orth_qr up to a rotation. Directions that a float32 Gram matrix cannot resolve are dropped, among them the
        # one of the centering
        y_gram = read_buffer(y_gram, np.float32, (k8, k8))[:k, :k].astype(np.float64)
        centering = np.eye(k) - 1.0 / k
        eigenvalues, w = np.linalg.eigh(centering @ ((y_gram + y_gram.T) / 2) @ centering)
        eigenvalues, w = eigenvalues[::-1], w[:, ::-1]
        resolved = eigenvalues > eigenvalues[0] * np.finfo(np.float32).eps
        a_q = centering @ w[:, resolved] / np.sqrt(eigenvalues[resolved])
        # the right term Q^T R = A_q^T X, its left singular vectors B and squared singular values from its Gram matrix
        x_gram = read_buffer(x_gram, np.float32, (k8, k8))[:k, :k].astype(np.float64)
        s2, left = np.linalg.eigh(a_q.T @ ((x_gram + x_gram.T) / 2) @ a_q)
        s2, left = np.maximum(s2[::-1][:test_rank], 0.0), left[:, ::-1]
        r = int(np.argmax(np.cumsum(s2) / np.sum(s2) >= 0.99)) + 1
        # term1 = U_proj^T Q B = U_proj^T Y A_q B and term2 = B^T Q^T R = (A_q B)^T X
        r4 = round_up(r, 4)
        coefficients_t = np.zeros((r4, k8), dtype=np.float32)
        coefficients_t[:r, :k] = (a_q @ left[:, :r]).T
        coefficients_t_buffer = self._buffer("multiunit_coefficients_t", coefficients_t.nbytes)
        device.queue.write_buffer(coefficients_t_buffer, 0, coefficients_t)

        encoder = self._new_encoder()
        term1 = self._buffer("multiunit_term1", 4 * comp.rank * r4)
        g = self._gemm("multiunit_term1")
        g.set_arguments(p_proj, coefficients_t_buffer, term1, comp.rank, r4, k8)
        g.encode(encoder)
        x_t = self._buffer("multiunit_x_t", 4 * n_frames_padded * k8)
        self._encode_transpose(encoder, "multiunit_x", x, k8, n_frames_padded, x_t, k8)
        term2 = self._buffer("multiunit_term2", 4 * r4 * n_frames_padded)
        g = self._gemm("multiunit_term2")
        g.set_arguments(coefficients_t_buffer, x_t, term2, r4, n_frames_padded, k8)
        g.encode(encoder)
        self._submit(encoder)
        return (
            read_buffer(term1, np.float32, (comp.rank, r4))[:, :r].copy(),
            read_buffer(term2, np.float32, (r4, n_frames_padded))[:r, : comp.n_frames].copy(),
        )
