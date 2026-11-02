"""Collect once per optimizer window, replay fixed blocks without target reruns."""

from pathlib import Path

from .batch import PreparedLogicalBatch, cpu_copy, tensor_bytes
from .component_reduction import global_component_counts


def prepare_anchor_opd_window(adapter, session, first, iterator, count):
    if session.args.accumulation_steps == 1:
        return None
    import torch

    prepared, used, counts, samples = [], 0, None, 0
    try:
        for index in range(count):
            raw = first if index == 0 else next(iterator)
            with torch.no_grad():
                metadata = session.fsdp_model(**adapter._anchor_values(session, raw),
                    opd_scorer=session.anchor_scorer, metadata_only=True)
            local = metadata['component_denominators']
            if counts is None:
                counts = {b: {k: v.detach().clone() for k, v in terms.items()} for b, terms in local.items()}
            else:
                for branch, terms in local.items():
                    for key, value in terms.items():
                        counts[branch][key] += value
            samples += metadata['sample_count']
            record = cpu_copy(metadata['sampled_opd_batch'])
            data = {'input_ids': cpu_copy(raw['input_ids']), 'loss_mask': cpu_copy(raw['loss_mask']),
                    'sampled_opd_batch': record}
            item = PreparedLogicalBatch(data, record.layout, None)
            prepared.append(item)
            size = tensor_bytes(data)
            if used + size > session.args.window_buffer_mb * 1024**2:
                item.spill(Path(session.args.cache_dir) / 'anchor_opd_windows')
            else:
                used += size
            del metadata, record, data
        totals, reference = global_component_counts(counts, samples, session.process_group)
        if totals['final']['survival'] <= 0:
            raise ValueError('Sampled OPD optimizer window contains no complete blocks')
        for item in prepared:
            item.opd_component_denominators = totals
            item.opd_reference_samples = reference
        return prepared
    except BaseException:
        for item in prepared:
            item.release()
        raise
