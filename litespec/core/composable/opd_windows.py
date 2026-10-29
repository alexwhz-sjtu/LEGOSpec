"""RNG-preserving detached OPD region metadata for multi-forward windows."""

from dataclasses import fields
from pathlib import Path

import torch

from litespec.training.checkpoint import capture_rng_state, restore_rng_state
from .batch import (BlockLayout, OPDReplayMetadata, PreparedLogicalBatch,
                    cpu_copy, prepare_layout, tensor_bytes)
from .component_reduction import global_component_counts


def prepare_opd_window(adapter, session, first, iterator, count):
    # Ordinary accumulation=1 is exactly one complete FSDP forward. Its three
    # counts can be reduced after all objective chunks without a metadata pass.
    if session.args.accumulation_steps == 1 and session.topology is None:
        return None
    from litespec.training.runner import iter_microbatch_slices

    entry_rng = capture_rng_state()
    prepared, used, local_counts, sample_count = [], 0, None, 0
    try:
        with torch.no_grad():
            for index in range(count):
                raw = cpu_copy(first if index == 0 else next(iterator))
                full = adapter._step_values(session, raw)
                layouts, replay = [], {}
                for start, end in iter_microbatch_slices(session):
                    values = {key: (None if value is None else value[start:end])
                              for key, value in full.items()}
                    layout = prepare_layout(values['input_ids'], values['loss_mask'],
                        session.draft.block_size, session.args.num_anchors, complete_blocks=True)
                    rng = capture_rng_state()
                    meta = session.fsdp_model(**values, layout=layout, metadata_only=True)
                    replay[start, end] = OPDReplayMetadata(
                        cpu_copy(meta['accepted_lengths']), rng)
                    layouts.append(cpu_copy(layout))
                    if local_counts is None:
                        local_counts = {branch: {key: den.detach().clone()
                            for key, den in terms.items()}
                            for branch, terms in meta['component_denominators'].items()}
                    else:
                        for branch, terms in meta['component_denominators'].items():
                            for key, den in terms.items():
                                local_counts[branch][key] += den.detach()
                    sample_count += meta['sample_count']
                joined = BlockLayout(**{field.name: torch.cat(
                    [getattr(layout, field.name) for layout in layouts], dim=0)
                    for field in fields(BlockLayout)})
                data = cpu_copy(full)
                item = PreparedLogicalBatch(data, joined, None,
                    opd_metadata=replay, opd_inputs_prepared=True)
                size = tensor_bytes(data)
                if used + size > session.args.window_buffer_mb * 1024**2:
                    item.spill(Path(session.args.cache_dir) / 'composable_windows')
                else:
                    used += size
                prepared.append(item)
                del full, values, meta, layout
        totals, reference = global_component_counts(local_counts, sample_count, session.process_group)
        for branch, terms in totals.items():
            for key, den in terms.items():
                if key in ('survival', 'token') and den <= 0:
                    raise ValueError('OPD optimizer window has no complete supervised blocks')
        for item in prepared:
            item.opd_component_denominators = cpu_copy(totals)
            item.opd_reference_samples = cpu_copy(reference)
        return prepared
    except BaseException:
        for item in prepared:
            item.release()
        raise
    finally:
        # Formal forwards replay their individual post-layout RNG snapshots.
        # At the end of the window the stochastic stream advances only once.
        restore_rng_state(entry_rng)
