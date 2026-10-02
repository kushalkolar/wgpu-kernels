"""
masknmf's merge of signals (merge_components) on the GPU, see SignalMerger
"""

import networkx as nx
import numpy as np
import pygfx

from ._compression import CompressionBuffers
from ._hals import (
    GPUComputation,
    SignalBuffers,
    set_constants_and_resources,
    create_buffer,
    create_empty_buffer,
    read_buffer,
    next_power_of_two,
)


class SignalMerger(GPUComputation):
    """
    masknmf's merge_components after its merge test (``CorrelationImages.get_merge_pairs``): the connected components
    of the pairs to merge, each replaced by the rank-1 nonnegative fit of its members' a_j c_j^T (rank_1_NMF_fit, see
    rank_1_nmf_fit.wgsl). The signals in no pair are kept in their order and the merged ones follow in the order of the
    components, as networkx gives them to masknmf.

    Parameters
    ----------
    compression: CompressionBuffers

    frame_batch_size: int, optional
        masknmf's ``frame_batch_size`` for the groups of the new signals, see ``SignalBuffers``
    """

    def __init__(self, compression: CompressionBuffers, frame_batch_size: int | None = None):
        super().__init__()
        self._compression = compression
        self._frame_batch_size = frame_batch_size

    def merge(self, signals: SignalBuffers, pairs: np.ndarray) -> tuple[SignalBuffers, np.ndarray]:
        """
        Merge the signals of the pairs [n_pairs, 2]. The new signals keep the ring term and b of signals, their groups
        are computed from the new a, as masknmf does after a merge (mask_ab = a.bool()). Without pairs, signals and all
        their indices are returned, signals with masknmf's mask_ab = a.bool() and their groups (see
        ``SignalBuffers.reset_mask``).

        Returns the new signals and the old index of each of the first ``preserved.size`` new signals.
        """
        device = pygfx.renderers.wgpu.get_shared().device
        comp = self._compression
        n = signals.n_signals
        if len(pairs) == 0:
            signals.reset_mask()
            return signals, np.arange(n)
        s = signals.structures
        a_ptr = s["a_ptr"].astype(np.int64)
        a_pixels = s["a_pixels"].astype(np.int64)
        n4 = comp.n_frames_padded // 4

        # masknmf's components and the order of their members
        pairs = np.asarray(pairs, dtype=np.int64)
        graph = nx.Graph()
        graph.add_edges_from(list(zip(pairs[:, 0], pairs[:, 1])))
        components = [np.array(list(component), dtype=np.int64) for component in nx.connected_components(graph)]
        preserved = np.setdiff1d(np.arange(n), np.unique(pairs))

        # the union of the members' pixels and the index in a_values of each member's entry at each of them
        member_ptr = np.cumsum([0] + [members.size for members in components])
        unions = []
        entries = []
        for members in components:
            union = np.unique(np.concatenate([a_pixels[a_ptr[j] : a_ptr[j + 1]] for j in members]))
            member_entries = np.full((union.size, members.size), 0xFFFFFFFF, dtype=np.uint32)
            for k, j in enumerate(members):
                member_entries[np.searchsorted(union, a_pixels[a_ptr[j] : a_ptr[j + 1]]), k] = np.arange(
                    a_ptr[j], a_ptr[j + 1]
                )
            unions.append(union)
            entries.append(member_entries.ravel())
        union_ptr = np.cumsum([0] + [union.size for union in unions])
        entry_ptr = np.cumsum([0] + [e.size for e in entries])

        n_components = len(components)
        spatial_fits = create_empty_buffer(device, 4 * union_ptr[-1])
        temporal_fits = create_empty_buffer(device, 16 * n_components * n4)
        shader = self._shader("rank_1_nmf_fit", reductions=True)
        # a power of two, so that the pipeline is recreated only when the largest component doubles
        max_members = max(next_power_of_two(int(np.diff(member_ptr).max())), 16)
        set_constants_and_resources(
            shader,
            {"n_frames": comp.n_frames, "n4": n4, "max_members": max_members},
            {
                0: create_buffer(device, member_ptr.astype(np.uint32)),
                1: create_buffer(device, np.concatenate(components).astype(np.uint32)),
                2: create_buffer(device, union_ptr.astype(np.uint32)),
                3: create_buffer(device, entry_ptr.astype(np.uint32)),
                4: create_buffer(device, np.concatenate(entries)),
                5: signals.buffers["a_values"],
                6: signals.buffers["temporal_demixed"],
                7: spatial_fits,
                8: create_empty_buffer(device, 4 * union_ptr[-1]),
                9: temporal_fits,
            },
        )
        encoder = self._new_encoder()
        shader.encode(encoder, n_components)
        self._submit(encoder)

        # the nonzero entries of each fit are the support of its merged signal
        spatial = read_buffer(spatial_fits, np.float32, (union_ptr[-1],))
        merged_pixels = []
        merged_values = []
        for g, union in enumerate(unions):
            x = spatial[union_ptr[g] : union_ptr[g + 1]]
            nonzero = np.flatnonzero(x)
            merged_pixels.append(union[nonzero])
            merged_values.append(x[nonzero])

        # new a and c: the preserved signals' entries and traces, then those of the merged signals
        n_new = preserved.size + n_components
        counts = np.concatenate([np.diff(a_ptr)[preserved], [p.size for p in merged_pixels]]).astype(np.int64)
        new_ptr = np.concatenate([[0], np.cumsum(counts)])
        pixels = np.concatenate([a_pixels[a_ptr[j] : a_ptr[j + 1]] for j in preserved] + merged_pixels)
        entry_signals = np.repeat(np.arange(n_new), np.diff(new_ptr))

        a_values = create_empty_buffer(device, 4 * new_ptr[-1])
        temporal_demixed = create_empty_buffer(device, 16 * n_new * n4)
        encoder = self._new_encoder()
        signals.encode_copy(encoder, preserved, a_values, temporal_demixed)
        encoder.copy_buffer_to_buffer(
            temporal_fits, 0, temporal_demixed, 16 * n4 * preserved.size, 16 * n4 * n_components
        )
        self._submit(encoder)
        if new_ptr[-1] > new_ptr[preserved.size]:
            device.queue.write_buffer(
                a_values, 4 * new_ptr[preserved.size], np.concatenate(merged_values).astype(np.float32)
            )

        merged = SignalBuffers.from_buffers(
            comp,
            pixels,
            entry_signals,
            n_new,
            a_values,
            temporal_demixed,
            signals.buffers["b"],
            frame_batch_size=self._frame_batch_size,
        )
        merged.set_ring_term_buffers(
            signals.buffers["ring_left"],
            signals.buffers["ring_right"],
            signals.buffers["ring_right_t"],
            signals.ring_rank,
        )
        return merged, preserved
