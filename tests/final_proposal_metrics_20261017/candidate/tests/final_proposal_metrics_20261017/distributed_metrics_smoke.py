"""Eight-GPU FSDP/BF16 smoke test of the final-proposal logging contract."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from test_implementation import config, fixture
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.training.logging import build_training_log_payload, reduce_training_metrics


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.set_num_threads(2)
    mesh = init_device_mesh('cuda', (dist.get_world_size(),))
    rows = []
    run = None
    if rank == 0:
        import wandb
        run = wandb.init(project='spec-decoding', entity='alex-wang_sjtu-sjtu',
                         name='validation_final_proposal_metrics_20261017',
                         tags=['validation', 'metrics-only-smoke'], mode='online',
                         config={'validation_only': True, 'world_size': 8,
                                 'precision': 'BF16', 'synthetic_inputs': True})
    for proposal in ['parallel', 'selector', 'markov', 'rnn', 'sequential_rnn']:
        torch.manual_seed(741)
        model, embedding, head, ids, hidden, teacher, mask = fixture(
            config('dflash', proposal), batch_size=1 + rank % 2)
        with torch.no_grad():
            if hasattr(model.proposal, 'candidate_selector'):
                model.proposal.candidate_selector.successor_codebook.normal_(std=.4)
            if hasattr(model.proposal, 'markov_head'):
                model.proposal.markov_head.markov_w2.weight.normal_(std=.4)
        wrapper = ComposableTrainingModel(model, head, embedding, 0,
                                          num_anchors=4, objective_chunk_blocks=1).cuda()
        fully_shard(wrapper, mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                                   reduce_dtype=torch.float32))
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=.001)
        for step in range(2):
            torch.manual_seed(2000 + rank + step * 100)
            output = wrapper(ids.cuda(), hidden.cuda(), mask.cuda(), teacher.cuda(),
                             collect_detailed_metrics=True)
            assert torch.isfinite(output.loss)
            output.loss.backward()
            for param in wrapper.parameters():
                if param.grad is not None:
                    grad = param.grad.to_local() if hasattr(param.grad, 'to_local') else param.grad
                    assert torch.isfinite(grad).all()
            keys = ['hard_label/token_accuracy', 'hard_label/block_accepted_length',
                    'expected_accept_len']
            for key in keys:
                terms = output.ratio_metrics[key]
                pair = torch.stack([terms.numerator.float(), terms.denominator.float()])
                dist.all_reduce(pair)
                expected = pair[0] / pair[1].clamp_min(1e-12)
                # The public reducer must sum counts before division on ragged ranks.
                if key == keys[0]:
                    reduced = reduce_training_metrics(output)
                torch.testing.assert_close(reduced[key], expected)
            payload = build_training_log_payload(reduced, .001, detail_prefix='composable')
            assert 0 <= payload['detail/composable/hard_label/token_accuracy'] <= 1
            assert 1 <= payload['train/walk_accepted_length'] <= 4
            assert 1 <= payload['train/expected_accept_len'] <= 4
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                payload['validation/proposal'] = proposal
                run.log(payload)
                rows.append({'proposal': proposal, 'step': step, 'payload': payload})
        del wrapper, optimizer, model
        torch.cuda.empty_cache()
    if rank == 0:
        report = {'world_size': 8, 'wandb_id': run.id, 'wandb_url': run.url, 'rows': rows}
        Path(os.environ['METRICS_REPORT']).write_text(json.dumps(report, indent=2))
        run.finish()
        print(json.dumps({'world_size': 8, 'checks': len(rows), 'wandb_url': report['wandb_url']}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
