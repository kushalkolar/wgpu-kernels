"""
masknmf's superpixel initialization and demixing loop on the GPU, see Demixer
"""

import itertools
from collections.abc import Iterator

import numpy as np
import pygfx
import wgpu
from masknmf.demixing import NoSignalsDetectedError
from masknmf.demixing.demixing_results import DemixingResults

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    HALS,
    SignalBuffers,
    set_constants_and_resources,
    create_buffer,
    create_empty_buffer,
    read_buffer,
    dispatch_grid,
)
from ._background import FluctuatingBaseline
from ._correlation import CorrelationImages, sample_noise_frames
from ._merge import SignalMerger
from ._local_correlation import LocalCorrelationImage
from ._results import get_demixing_results
from ._superpixels import get_patches, get_patch_pairs, select_pure_superpixels


class Demixer(GPUComputation):
    """
    masknmf's demixing of a dataset (InitializingState, DemixingState) on the GPU: ``initialize_signals`` finds new
    signals in the residual of the signals by superpixels, ``demix`` runs a pass from initial signals one iteration at a
    time, the signals of the current iteration in ``signals``, and computes masknmf's outputs at the end of the pass,
    which ``get_results`` exports.

    Parameters
    ----------
    compression: CompressionBuffers

    compression_results: dict
        the compression results, see ``load_compression``: U for the ring model and the tensors of the export

    frame_batch_size: int
        masknmf's frame_batch_size (SignalDemixer's default), the max number of signals of a group and the frames per
        batch of the robust noise term
    """

    def __init__(self, compression: CompressionBuffers, compression_results: dict, frame_batch_size: int = 5000):
        super().__init__()
        device = pygfx.renderers.wgpu.get_shared().device
        height, width = compression.fov_shape
        self._compression = compression
        self._compression_results = compression_results
        self._frame_batch_size = frame_batch_size
        self._correlation = CorrelationImages(compression)
        self._merger = SignalMerger(compression, frame_batch_size)
        self._local_correlation = LocalCorrelationImage(compression)
        # the ring model of the downsampling factor and ring radius of the last pass
        self._background = None
        self._background_parameters = None
        self._hals = None
        self._signals = None
        self._background_rank = None
        self._multiunit = None
        # (mad_threshold, carry_background) of the local correlation image of initialize_signals, None after a pass
        self._initialization_image = None
        # the local correlation image, superpixels and pure superpixels of the last initialize_signals
        self._superpixels = None
        # x of a_spmm_t for a^T 1
        self._ones = create_buffer(device, np.ones(height * width, dtype=np.float32))

    def initialize_signals(
        self,
        mad_threshold: int = 1,
        mad_correlation_threshold: float = 0.9,
        min_peak_distance: int = 3,
        residual_threshold: float = 0.3,
        patch_size: tuple[int, int] = (100, 100),
        sign: str = "unconstrained",
        carry_background: bool = False,
    ) -> SignalBuffers:
        """
        masknmf's superpixel initialization (InitializingState._initialize_signals_superpixels, superpixel_init) on the
        residual of ``signals``, the signals of the last pass of ``demix``, with their ring term if carry_background, or
        on U V before the first pass. The local correlation image of the residual with mad_threshold and sign is
        computed again after a pass or when mad_threshold or carry_background change, masknmf ignores a change of sign.
        Its peaks above mad_correlation_threshold, the max of their windows of min_peak_distance pixels around them,
        are one-pixel signals, updated with the signals by one temporal and one spatial HALS update without a ring term.
        Of them the pure superpixels of each patch of patch_size pixels are selected by successive projection with
        residual_threshold.

        Returns the signals, unchanged, followed by the pure superpixels, with the ring term of the signals if
        carry_background. The signals of the Demixer do not change. Raises NoSignalsDetectedError without peaks or
        pure superpixels.
        """
        comp = self._compression
        correlation = self._correlation
        # the robust noise term of the last pass, before the first pass drawn as masknmf's InitializingState draws it
        if correlation.robust_noise is None:
            correlation.set_robust_noise(sample_noise_frames(comp.n_frames), self._frame_batch_size)
        signals = self._signals
        if signals is not None and not carry_background and signals.ring_rank > 0:
            signals = self._without_ring_term(signals)
        if self._initialization_image != (mad_threshold, carry_background):
            self._local_correlation.compute(correlation.robust_noise, mad_threshold, sign, signals)
            self._initialization_image = (mad_threshold, carry_background)

        peaks = self._get_superpixels(mad_correlation_threshold, min_peak_distance)
        superpixels = self._update_superpixels(signals, peaks)
        n_old = 0 if signals is None else signals.n_signals
        pure = self._select_pure_superpixels(superpixels, n_old, peaks, patch_size, residual_threshold)
        self._superpixels = (self._local_correlation.get_image(), peaks, peaks[pure])

        # the signals, as they are, followed by the pure superpixels as superpixels has them
        initialized = self._extend_signals(signals, peaks[pure])
        b = initialized.buffers
        n_old_entries = 0 if signals is None else signals.structures["a_pixels"].size
        encoder = self._new_encoder()
        superpixels.encode_copy(encoder, n_old + pure, b["a_values"], b["temporal_demixed"], n_old_entries, n_old)
        self._submit(encoder)
        if signals is not None and signals.ring_rank > 0:
            ring = signals.buffers
            initialized.set_ring_term_buffers(
                ring["ring_left"], ring["ring_right"], ring["ring_right_t"], signals.ring_rank
            )
        return initialized

    def _without_ring_term(self, signals: SignalBuffers) -> SignalBuffers:
        """signals with the buffers of a, c and b of signals, without a ring term"""
        s = signals.structures
        b = signals.buffers
        return SignalBuffers.from_buffers(
            self._compression,
            s["a_pixels"],
            np.repeat(np.arange(signals.n_signals), np.diff(s["a_ptr"])),
            signals.n_signals,
            b["a_values"],
            b["temporal_demixed"],
            b["b"],
            frame_batch_size=self._frame_batch_size,
        )

    def _get_superpixels(self, mad_correlation_threshold: float, min_peak_distance: int) -> np.ndarray:
        """masknmf's peaks of the local correlation image (find_local_peaks_2d), the pixels of the superpixels in
        row-major order, see local_peaks.wgsl"""
        height, width = self._compression.fov_shape
        peaks = self._buffer("peaks", 4 * height * width)
        shader = self._shader("local_peaks")
        set_constants_and_resources(
            shader,
            {
                "height": height,
                "width": width,
                "radius": min_peak_distance,
                "threshold": float(mad_correlation_threshold),
            },
            {0: self._local_correlation.image, 1: peaks},
        )
        encoder = self._new_encoder()
        shader.encode(encoder, *dispatch_grid(-(-height * width // 256)))
        self._submit(encoder)
        pixels = np.flatnonzero(read_buffer(peaks, np.uint32, (height * width,)))
        if pixels.size == 0:
            raise NoSignalsDetectedError(
                "No Signals passed correlation threshold. Lower the threshold or verify input data quality."
            )
        return pixels

    def _extend_signals(self, signals: SignalBuffers | None, pixels: np.ndarray) -> SignalBuffers:
        """new signals: those of signals, their a and c copied, followed by signals of one entry each at pixels, their
        a and c 0, b 0, the groups of the support of a, without a ring term"""
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        height, width = comp.fov_shape
        n_old = 0 if signals is None else signals.n_signals
        n = n_old + pixels.size
        old_pixels = np.zeros(0, dtype=np.int64)
        old_signals = np.zeros(0, dtype=np.int64)
        if signals is not None:
            s = signals.structures
            old_pixels = s["a_pixels"].astype(np.int64)
            old_signals = np.repeat(np.arange(n_old), np.diff(s["a_ptr"]))
        a_values = create_empty_buffer(device, 4 * (old_pixels.size + pixels.size))
        temporal_demixed = create_empty_buffer(device, 4 * n * comp.n_frames_padded)
        if signals is not None:
            encoder = self._new_encoder()
            signals.encode_copy(encoder, np.arange(n_old), a_values, temporal_demixed)
            self._submit(encoder)
        return SignalBuffers.from_buffers(
            comp,
            np.concatenate([old_pixels, pixels]),
            np.concatenate([old_signals, n_old + np.arange(pixels.size)]),
            n,
            a_values,
            temporal_demixed,
            create_empty_buffer(device, 4 * height * width),
            frame_batch_size=self._frame_batch_size,
        )

    def _update_superpixels(self, signals: SignalBuffers | None, peaks: np.ndarray) -> SignalBuffers:
        """
        masknmf's spatial_temporal_ini_uv: signals followed by a one-pixel signal of value 1 and trace 0 at each peak,
        updated by a temporal and then a spatial HALS update without a ring term, the groups of the support of a, with
        masknmf's baseline b = x - a mean(c), x = U mean(V) - a mean(c) of the traces before the updates.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        height, width = comp.fov_shape
        superpixels = self._extend_signals(signals, peaks)
        n = superpixels.n_signals
        n_old_entries = 0 if signals is None else signals.structures["a_pixels"].size
        b = superpixels.buffers
        device.queue.write_buffer(b["a_values"], 4 * n_old_entries, np.ones(peaks.size, dtype=np.float32))
        if self._hals is None:
            self._hals = HALS(comp, superpixels)
        else:
            self._hals.set_signals(superpixels)

        c = b["temporal_demixed"]
        sums = self._buffer("superpixel_sums", 4 * n)
        x = self._buffer("superpixel_x", 4 * height * width)
        encoder = self._new_encoder()
        self._encode_row_sums(encoder, "superpixel_sums", c, comp.n_frames_padded, 0, comp.n_frames, sums, n)
        self._encode_a_spmv(encoder, superpixels, sums, self._correlation.std_corr_img_mean, x)
        self._encode_a_spmv(encoder, superpixels, sums, x, b["b"])
        self._hals.temporal_update(True, encoder)
        self._encode_row_sums(encoder, "superpixel_sums", c, comp.n_frames_padded, 0, comp.n_frames, sums, n)
        self._encode_a_spmv(encoder, superpixels, sums, x, b["b"])
        self._hals.spatial_update(encoder)
        self._submit(encoder)
        return superpixels

    def _encode_a_spmv(
        self,
        encoder: wgpu.GPUCommandEncoder,
        signals: SignalBuffers,
        sums: wgpu.GPUBuffer,
        x: wgpu.GPUBuffer,
        out: wgpu.GPUBuffer,
    ):
        """out = x - a (sums / n_frames) for the a of signals, see a_spmv.wgsl"""
        comp = self._compression
        height, width = comp.fov_shape
        b = signals.buffers
        shader = self._shader("a_spmv")
        set_constants_and_resources(
            shader,
            {"n_pixels": height * width, "n_frames": comp.n_frames},
            {
                0: b["pixel_ptr"],
                1: b["pixel_entries"],
                2: b["pixel_signals"],
                3: b["a_values"],
                4: sums,
                5: x,
                6: out,
            },
        )
        shader.encode(encoder, *dispatch_grid(-(-height * width // 256)))

    def _select_pure_superpixels(
        self,
        superpixels: SignalBuffers,
        n_old: int,
        peaks: np.ndarray,
        patch_size: tuple[int, int],
        residual_threshold: float,
    ) -> np.ndarray:
        """
        masknmf's pure superpixels, of the superpixels after the first n_old signals of superpixels: successive
        projection in each patch of patch_size pixels, from the products of their traces (trace_products.wgsl on the
        pairs of each patch) and their sums, the L1 norms of the nonnegative traces.

        Returns the pure superpixels, indices of the superpixels, sorted.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        n = superpixels.n_signals
        c = superpixels.buffers["temporal_demixed"]
        patches, patch_ptr, members = get_patches(peaks, comp.fov_shape, patch_size)
        neighbor_ptr, neighbors = get_patch_pairs(patches, patch_ptr, members)
        products = self._buffer("superpixel_products", 4 * neighbors.size)
        self_products = self._buffer("superpixel_self_products", 4 * n)
        shader = self._shader("trace_products", "superpixels", reductions=True)
        set_constants_and_resources(
            shader,
            {"n4": comp.n_frames_padded // 4},
            {
                0: c,
                1: c,
                # the graph of all signals, the first n_old without pairs
                2: create_buffer(device, np.concatenate([np.zeros(n_old), neighbor_ptr]).astype(np.uint32)),
                3: create_buffer(device, (n_old + neighbors).astype(np.uint32)),
                4: products,
                5: self_products,
            },
        )
        shader.set_uniform(6, np.array([n, 0, 0, 0], dtype=np.uint32))
        encoder = self._new_encoder()
        shader.encode(encoder, *dispatch_grid(n))
        self._submit(encoder)
        pure = select_pure_superpixels(
            patch_ptr,
            members,
            neighbor_ptr,
            read_buffer(products, np.float32, (neighbors.size,)),
            read_buffer(self_products, np.float32, (n,))[n_old:],
            read_buffer(self._buffer("superpixel_sums", 4 * n), np.float32, (n,))[n_old:],
            residual_threshold,
        )
        if pure.size == 0:
            raise NoSignalsDetectedError(
                "No pure superpixels passed the residual threshold. Lower the threshold or verify input data quality."
            )
        return pure

    def demix(
        self,
        signals: SignalBuffers,
        maxiter: int = 25,
        support_threshold: float | list | tuple = 0.9,
        deletion_threshold: float = 0.2,
        min_brightness: float | None = 1.0,
        ring_model_start_pt: int | None = 0,
        background_downsampling_factor: int = 20,
        ring_radius: int = 10,
        merge_threshold: float = 0.8,
        merge_overlap_threshold: float = 0.4,
        update_frequency: int = 4,
        c_nonneg: bool = True,
        sign: str = "unconstrained",
        seed: int = 0,
    ) -> Iterator[int]:
        """
        A pass of masknmf's demix from signals, with its parameters: a generator that yields each iteration after it
        and computes the outputs of the pass after the last one, the residual and background-to-signal correlation
        images, the global residual correlation image and the multiunit factorization. b of signals is set to 0, as
        masknmf's static baseline is in its loop. The random matrices of the ring model and of the multiunit
        factorization are drawn on the GPU from seed. Not ported: reassign_background, detrender and denoise, which is
        a no-op in masknmf.
        """
        comp = self._compression

        # masknmf's schedules
        if isinstance(support_threshold, (list, tuple)):
            if len(support_threshold) == 2:
                support_threshold = np.linspace(support_threshold[0], support_threshold[1], maxiter).tolist()
            elif len(support_threshold) != maxiter:
                raise ValueError(f"Length of list ``support_threshold`` is not equal to maxiter, which is {maxiter}")
        elif isinstance(support_threshold, float):
            support_threshold = [support_threshold] * maxiter
        else:
            raise ValueError(f"support_threshold has invalid type: {type(support_threshold)}")
        min_brightness_list = [None] * maxiter if min_brightness is None else np.linspace(0, min_brightness, maxiter)

        # the start of the pass, masknmf's DemixingState and the start of its demix
        self._multiunit = None
        # the end of the pass writes the global residual correlation image over the image of initialize_signals
        self._initialization_image = None
        self._correlation.set_robust_noise(sample_noise_frames(comp.n_frames), self._frame_batch_size)
        encoder = self._new_encoder()
        encoder.clear_buffer(signals.buffers["b"])
        self._submit(encoder)
        if self._background_parameters != (background_downsampling_factor, ring_radius):
            self._background = FluctuatingBaseline(
                comp, self._compression_results["u"], background_downsampling_factor, ring_radius
            )
            self._background_parameters = (background_downsampling_factor, ring_radius)
        self._set_signals(signals)
        self._background_rank = signals.ring_rank if signals.ring_rank > 0 else None
        self._correlation.set_uv_norms(signals)
        # update uses seed and seed + 1 when it estimates the background rank
        seeds = itertools.count(seed, 2)

        background_enabled = False
        for iteration in range(maxiter):
            if ring_model_start_pt is not None and iteration >= ring_model_start_pt:
                background_enabled = True
            if background_enabled:
                self._update_ring_term(next(seeds))
            self._update_signals(c_nonneg)
            if update_frequency and (iteration + 1) % update_frequency == 0:
                self._update_supports(
                    merge_threshold,
                    merge_overlap_threshold,
                    support_threshold[iteration],
                    deletion_threshold,
                    min_brightness_list[iteration],
                )
            yield iteration

        # the outputs of the pass, with W for the last c
        signals = self._signals
        self._correlation.update_residual(signals, np.zeros(0, dtype=np.int64))
        self._correlation.set_background_images(signals)
        self._local_correlation.compute(self._correlation.robust_noise, 1, sign, signals)
        self._multiunit = self._background.get_multiunit_factorization(seed=next(seeds))

    def _set_signals(self, signals: SignalBuffers):
        """the signals of the HALS updates and the ring model"""
        self._signals = signals
        if self._hals is None:
            self._hals = HALS(self._compression, signals)
        else:
            self._hals.set_signals(signals)
        self._background.set_signals(signals)

    def _update_ring_term(self, seed: int):
        """masknmf's fluctuating_baseline_update, which estimates the background rank when it is None"""
        signals = self._signals
        ring_left = signals.buffers["ring_left"]
        self._background_rank = self._background.update(self._background_rank, seed)
        # the ring term is in new buffers when its padded rank changed
        if signals.buffers["ring_left"] is not ring_left:
            self._hals.set_signals(signals)

    def _update_signals(self, c_nonneg: bool):
        """
        masknmf's spatial and temporal updates, each followed by the deletion of the signals whose a or c is 0. The
        temporal update is recorded while the spatial one runs, and again after a deletion.
        """
        encoder = self._new_encoder()
        self._hals.spatial_update(encoder)
        a_sums = self._encode_a_sums(encoder)
        self._submit(encoder)
        encoder = self._new_encoder()
        self._hals.temporal_update(c_nonneg, encoder)
        c_sums = self._encode_c_sums(encoder)
        if self._delete_zero_signals(a_sums):
            encoder = self._new_encoder()
            self._hals.temporal_update(c_nonneg, encoder)
            c_sums = self._encode_c_sums(encoder)
        self._submit(encoder)
        self._delete_zero_signals(c_sums)

    def _encode_a_sums(self, encoder: wgpu.GPUCommandEncoder) -> wgpu.GPUBuffer:
        """masknmf's a^T 1 of the signals, see a_spmm_t.wgsl"""
        comp = self._compression
        signals = self._signals
        b = signals.buffers
        height, width = comp.fov_shape
        sums = self._buffer("a_sums", 4 * signals.n_signals)
        shader = self._shader("a_spmm_t", "a_sums")
        set_constants_and_resources(
            shader,
            {"n_pixels": height * width, "n_j": 1},
            {0: b["a_ptr"], 1: b["a_pixels"], 2: b["a_values"], 3: self._ones, 4: sums},
        )
        shader.set_uniform(5, np.array([signals.n_signals, 0, 0, 0], dtype=np.uint32))
        shader.encode(encoder, 1, signals.n_signals)
        return sums

    def _encode_c_sums(self, encoder: wgpu.GPUCommandEncoder) -> wgpu.GPUBuffer:
        """masknmf's sums of the traces over the frames"""
        comp = self._compression
        signals = self._signals
        sums = self._buffer("c_sums", 4 * signals.n_signals)
        c = signals.buffers["temporal_demixed"]
        self._encode_row_sums(encoder, "c_sums", c, comp.n_frames_padded, 0, comp.n_frames, sums, signals.n_signals)
        return sums

    def _delete_zero_signals(self, sums: wgpu.GPUBuffer) -> bool:
        """
        masknmf's delete_comp of the signals whose sums are 0 and its update_hals_scheduler: index_select keeps their
        mask_ab. Returns whether signals were deleted.
        """
        signals = self._signals
        keep = np.flatnonzero(read_buffer(sums, np.float32, (signals.n_signals,)) != 0)
        if keep.size == signals.n_signals:
            return False
        if keep.size == 0:
            raise ValueError("All Components are slated to be deleted")
        self._set_signals(signals.index_select(keep, self._frame_batch_size))
        return True

    def _update_supports(
        self,
        merge_threshold: float,
        merge_overlap_threshold: float,
        support_threshold: float,
        deletion_threshold: float,
        min_brightness: float | None,
    ):
        """masknmf's merge_signals and support_update_routine"""
        correlation = self._correlation
        pairs = correlation.get_merge_pairs(self._signals, merge_threshold, merge_overlap_threshold)
        signals, preserved = self._merger.merge(self._signals, pairs)
        correlation.update_residual(signals, preserved)
        keep = correlation.get_signals_to_keep(deletion_threshold, min_brightness)
        # masknmf estimates the background rank again after a deletion test with min_brightness
        if min_brightness is not None:
            self._background_rank = None
        if keep.size < signals.n_signals:
            signals = signals.index_select(keep, self._frame_batch_size)
            correlation.update_residual(signals, keep)
        self._set_signals(correlation.expand_masks(support_threshold, self._frame_batch_size))

    @property
    def signals(self) -> SignalBuffers | None:
        """the signals of the current iteration of ``demix``, new ones after each change of the signals"""
        return self._signals

    @property
    def background_rank(self) -> int | None:
        """masknmf's background rank, None until the ring model estimates it"""
        return self._background_rank

    def get_superpixels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        The local correlation image [height, width] of the last ``initialize_signals``, and the pixels of its superpixels
        and of its pure superpixels, row-major. None before the first initialization.
        """
        return self._superpixels

    def get_results(self) -> DemixingResults:
        """masknmf's DemixingResults of the last pass of ``demix``, see ``get_demixing_results``"""
        if self._multiunit is None:
            raise RuntimeError("get_results needs a finished pass of demix")
        return get_demixing_results(
            self._compression_results, self._signals, self._correlation, self._local_correlation, self._multiunit
        )
