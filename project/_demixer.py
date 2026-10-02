"""
masknmf's demixing loop (DemixingState.demix) on the GPU, see Demixer
"""

import itertools
from collections.abc import Iterator

import numpy as np
import pygfx
import wgpu
from masknmf.demixing.demixing_results import DemixingResults

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    HALS,
    SignalBuffers,
    set_constants_and_resources,
    create_buffer,
    read_buffer,
)
from ._background import FluctuatingBaseline
from ._correlation import CorrelationImages, sample_noise_frames
from ._merge import SignalMerger
from ._local_correlation import LocalCorrelationImage
from ._results import get_demixing_results


class Demixer(GPUComputation):
    """
    masknmf's demixing of a dataset (DemixingState.demix) on the GPU: ``demix`` runs a pass from initial signals one
    iteration at a time, the signals of the current iteration in ``signals``, and computes masknmf's outputs at the end
    of the pass, which ``get_results`` exports.

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
        # x of a_spmm_t for a^T 1
        self._ones = create_buffer(device, np.ones(height * width, dtype=np.float32))

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

    def get_results(self) -> DemixingResults:
        """masknmf's DemixingResults of the last pass of ``demix``, see ``get_demixing_results``"""
        if self._multiunit is None:
            raise RuntimeError("get_results needs a finished pass of demix")
        return get_demixing_results(
            self._compression_results, self._signals, self._correlation, self._local_correlation, self._multiunit
        )
