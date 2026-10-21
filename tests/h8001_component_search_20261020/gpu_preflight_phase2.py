"""Eight GPU FSDP/BF16 validation; runs only after the preceding queue releases GPUs."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from copy import deepcopy
from torch import nn
from litespec.modeling.draft.composable import ComposableDraftConfig,ComposableDraftModel
RECIPES=[{'type':'hard_ce'},{'type':'dpace','alpha':.5},{'type':'nla','loss_decay_gamma':4.0},{'type':'weighted_sum','components':[{'type':'hard_ce','weight':.1},{'type':'tv','weight':.9}]}]
def config(proposal='rnn'):
 return ComposableDraftConfig(hidden_size=32,intermediate_size=48,num_attention_heads=4,
  num_key_value_heads=2,head_dim=8,num_hidden_layers=2,num_target_layers=8,vocab_size=32,
  block_size=8,composable_config={'context':{'type':'dflash','target_layer_ids':[1,5]},
   'backbone':{'type':'transformer','layers':[{'attention':{'representation':'mha','access':'dense','backend':'sdpa'},'local_operator':{'type':'none'}} for _ in range(2)]},
   'proposal':{'type':proposal,'rank':8,'loss_decay_gamma':4.},'objective':{'type':'hard_ce'}})
def fixture(cfg,batch_size):
 model=ComposableDraftModel(cfg);embedding=nn.Embedding(32,32);head=nn.Linear(32,32,bias=False)
 embedding.requires_grad_(False);head.requires_grad_(False)
 return model,embedding,head,torch.randint(32,(batch_size,24)),torch.randn(batch_size,24,64),torch.randn(batch_size,24,32),torch.ones(batch_size,24)

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
    for scenario, recipe in [(p,r) for p in ['dflare','conv'] for r in RECIPES]:
        proposal='rnn'
        torch.manual_seed(1818)
        cfg = config(proposal=proposal)
        if scenario=='dflare':cfg.composable_config['context']['type']='dflare'
        else:
            for layer in cfg.composable_config['backbone']['layers']:
                layer['local_operator']={'type':'dflash2_conv','kernel_size':2,'group_size':16}
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
            for key in ['objective/main','hard_label/token_accuracy',
                        'hard_label/block_accepted_length','expected_accept_len']:
                assert torch.isfinite(reduced[key]), (recipe, key)
                terms = out.ratio_metrics[key]
                pair = torch.stack([terms.numerator.float(), terms.denominator.float()])
                dist.all_reduce(pair)
                torch.testing.assert_close(reduced[key], pair[0]/pair[1].clamp_min(1e-12))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                rows.append({'proposal':proposal,'objective':recipe, 'step':step,
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
