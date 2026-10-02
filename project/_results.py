"""
masknmf's DemixingResults from the GPU buffers at the end of a pass, see get_demixing_results
"""

import numpy as np
import torch
from masknmf.demixing.demixing_results import DemixingResults

from ._hals import SignalBuffers
from ._correlation import CorrelationImages
from ._local_correlation import LocalCorrelationImage


def get_demixing_results(
    compression_results: dict,
    signals: SignalBuffers,
    correlation: CorrelationImages,
    local_correlation: LocalCorrelationImage,
    multiunit: tuple[np.ndarray, np.ndarray],
) -> DemixingResults:
    """
    masknmf's DemixingResults at the end of a pass (the end of signal_demixer.py's demix), read back from the GPU
    after ``correlation.update_residual(signals)``, ``correlation.set_background_images(signals)``,
    ``local_correlation.compute`` with a mad_threshold of 1 and the signals, and their multiunit factorization. Without
    a ring term the factorized ring term is the zeros masknmf's DemixingState starts from.

    Parameters
    ----------
    compression_results: dict
        the compression results, see ``load_compression``

    signals: SignalBuffers
        the signals at the end of the pass

    correlation: CorrelationImages

    local_correlation: LocalCorrelationImage

    multiunit: tuple[np.ndarray, np.ndarray]
        (term1 [rank, r], term2 [r, n_frames]), see ``FluctuatingBaseline.get_multiunit_factorization``
    """
    n_frames, height, width = compression_results["shape"]
    rank = compression_results["v"].shape[0]
    ring_term = signals.get_factorized_ring_term()
    if ring_term is None:
        ring_term = (np.zeros((rank, 1), dtype=np.float32), np.zeros((1, n_frames), dtype=np.float32))

    # the support values in the order of the entries of a
    s = signals.structures
    entry_signals = np.repeat(np.arange(signals.n_signals), np.diff(s["a_ptr"]))
    support_values = torch.sparse_coo_tensor(
        np.stack([s["a_pixels"].astype(np.int64), entry_signals]),
        correlation.get_resid_corr_img_support_values(),
        (height * width, signals.n_signals),
    ).coalesce()

    term1, term2 = multiunit
    return DemixingResults(
        (n_frames, height, width),
        compression_results["u"],
        compression_results["v"],
        signals.get_a(),
        torch.from_numpy(signals.get_c()),
        mean_img=compression_results["mean_img"],
        var_img=compression_results["var_img"],
        u_local_projector=compression_results["u_local_projector"],
        factorized_bkgd_term1=torch.from_numpy(ring_term[0]),
        factorized_bkgd_term2=torch.from_numpy(ring_term[1]),
        b=torch.from_numpy(signals.get_b()),
        std_corr_img_mean=torch.from_numpy(correlation.get_std_corr_img_mean()),
        std_corr_img_normalizer=torch.from_numpy(correlation.get_std_corr_img_normalizer()),
        resid_corr_img_support_values=support_values,
        resid_corr_img_mean=torch.from_numpy(correlation.get_resid_corr_img_mean()),
        resid_corr_img_normalizer=torch.from_numpy(correlation.get_resid_corr_img_normalizer()),
        bkgd_corr_img_mean=torch.from_numpy(correlation.get_bkgd_corr_img_mean()),
        bkgd_corr_img_normalizer=torch.from_numpy(correlation.get_bkgd_corr_img_normalizer()),
        global_residual_correlation_image=torch.from_numpy(local_correlation.get_image()),
        multiunit_basis_term1=torch.from_numpy(term1),
        multiunit_basis_term2=torch.from_numpy(term2),
        device="cpu",
    )
