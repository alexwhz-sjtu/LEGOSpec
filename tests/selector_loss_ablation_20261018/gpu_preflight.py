"""Eight GPU FSDP/BF16 validation; runs only after the preceding queue releases GPUs."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from test_losses import RECIPES, config, fixture
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.training.logging import reduce_training_metrics


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    torch.set_num_threads(2)
    mesh = init_device_mesh('cuda', (dist.get_world_size(),))
    assert dist.get_world_size() == 8
    rows = []
    for recipe in RECIPES:
        torch.manual_seed(1818)
        cfg = config(proposal='selector')
        cfg.composable_config['objective'] = recipe
        model, embedding, head, ids, hidden, teacher, mask = fixture(cfg, batch_size=1+rank%2)
        wrapper = ComposableTrainingModel(model, head, embedding, 0, num_anchors=4,
                                          objective_chunk_blocks=1).cuda()
        fully_shard(wrapper, mesh=mesh, mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        optimizer = torch.optim.SGD(wrapper.parameters(), lr=.001)
        for step in range(2):
            torch.manual_seed(1800 + rank + 100*step)
            out = wrapper(ids.cuda(), hidden.cuda(), mask.cuda(), teacher.cuda(),
                          collect_detailed_metrics=True)
            assert torch.isfinite(out.loss)
            out.loss.backward()
            for param in wrapper.parameters():
                if param.grad is not None:
                    grad = param.grad.to_local() if hasattr(param.grad, 'to_local') else param.grad
                    assert torch.isfinite(grad).all()
            reduced = reduce_training_metrics(out)
            for key in ['objective/main','objective/selector_ce','hard_label/token_accuracy',
                        'hard_label/block_accepted_length','expected_accept_len']:
                assert torch.isfinite(reduced[key]), (recipe, key)
                terms = out.ratio_metrics[key]
                pair = torch.stack([terms.numerator.float(), terms.denominator.float()])
                dist.all_reduce(pair)
                torch.testing.assert_close(reduced[key], pair[0]/pair[1].clamp_min(1e-12))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                rows.append({'objective':recipe, 'step':step,
                             'metrics':{k:float(v) for k,v in reduced.items()}})
        del optimizer, wrapper, model
        torch.cuda.empty_cache()
    if rank == 0:
        report = {'passed':True,'world_size':8,'precision':'BF16','rows':rows}
        Path(os.environ['PREFLIGHT_REPORT']).write_text(json.dumps(report, indent=2))
        print(json.dumps({'passed':True,'checks':len(rows)}), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == '__main__':
    main()
