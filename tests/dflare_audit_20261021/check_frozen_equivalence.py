"""CPU-only numerical audit of official DFlare and the exact frozen experiment."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import PretrainedConfig

HERE = Path(__file__).resolve().parent
PROJECT = Path('/share/dai-sys/wanghanzhen/projects/DAC27/LiteSpec_exp')
SNAPSHOT = PROJECT / 'tests/h8001_component_search_20261020/snapshot'
sys.path.insert(0, str(SNAPSHOT))
# Skip only eager parent-package initialization (which imports distributed
# training/yunchang and queries CUDA even for this CPU-only model audit).
# All actual model modules below still load unchanged from the frozen source.
for package in ('litespec','litespec.modeling','litespec.modeling.draft'):
    module=ModuleType(package)
    module.__path__=[str(SNAPSHOT.joinpath(*package.split('.')))]
    sys.modules[package]=module
torch.set_num_threads(2)
torch.manual_seed(20261021)
assert not torch.cuda.is_initialized()
ns = dict(torch=torch, nn=nn, F=F, List=List, Optional=Optional, Tuple=Tuple, PretrainedConfig=PretrainedConfig)
for file, names in [('pinned_dflash.py', {'DFlashRMSNorm','DFlashRotaryEmbedding','DFlashMLP','_repeat_kv','_rotate_half'}), ('pinned_dflare.py', {'DFlareAttention','DFlareDecoderLayer'})]:
    for node in ast.parse((HERE / file).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
tree=ast.parse((HERE/'pinned_dflare.py').read_text())
for node in tree.body:
    if isinstance(node,ast.ClassDef) and node.name=='DFlareDraftModel':
        for m in node.body:
            if isinstance(m,ast.FunctionDef) and m.name in ('_init_fusion_weights','forward'):
                exec(compile(ast.Module(body=[m],type_ignores=[]),'official_methods.py','exec'),ns)

from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel, DraftState
from litespec.modeling.draft.composable.contracts import ModelInputs

cfg=ComposableDraftConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
    num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=40000,
    rope_theta=1000000.,rms_norm_eps=1e-6,attention_dropout=0.,attention_bias=False,
    vocab_size=64,num_target_layers=36,block_size=8,
    composable_config=dict(context=dict(type='dflare',target_layer_ids=[1,9,17,25,33]),
      backbone=dict(type='transformer',layers=[dict(attention=dict(representation='mha',access='dense',backend='sdpa'),local_operator=dict(type='none')) for _ in range(5)]),
      proposal=dict(type='parallel'),objective=dict(type='hard_ce')))
draft=ComposableDraftModel(cfg).eval()
ref=SimpleNamespace(num_layers=5,num_target_layers=5,layer_fusion_weights=nn.Parameter(torch.empty(5,5)),
  context_norm=ns['DFlashRMSNorm'](32),layers=nn.ModuleList([ns['DFlareDecoderLayer'](cfg) for _ in range(5)]),final_norm=ns['DFlashRMSNorm'](32))
ns['_init_fusion_weights'](ref)
torch.testing.assert_close(ref.layer_fusion_weights,draft.context.layer_fusion_weights,rtol=0,atol=0)
results={'fusion_prior_exact':True,'cpu_only':True,'official_pinned_equals_main':(HERE/'pinned_dflare.py').read_bytes()==(HERE/'current_dflare.py').read_bytes()}
checkpoint=Path('/share/dai-sys/wanghanzhen/projects/DAC27/runs/component_search_h8001_20261020/09_dflare_context_ce01_tv09/checkpoints/final/model.safetensors')
with safe_open(checkpoint,framework='pt',device='cpu') as f:
    learned=f.get_tensor('context.layer_fusion_weights')
    weights=learned.float().softmax(-1)
    results['trained_fusion']={'logits':learned.float().tolist(),'probabilities':weights.tolist(),'entropy':(-(weights*weights.log()).sum(-1)).tolist()}
    results['context_norm_range']=[float(f.get_tensor('context.norm.weight').min()),float(f.get_tensor('context.norm.weight').max())]

with torch.no_grad():
    # Use the actual learned fusion logits, and identical synthetic backbone weights.
    draft.context.layer_fusion_weights.copy_(learned)
    ref.layer_fusion_weights.copy_(draft.context.layer_fusion_weights)
    ref.context_norm.weight.copy_(draft.context.norm.weight)
    ref.final_norm.weight.copy_(draft.backbone.norm.weight)
    remap={'k_proj_target':'context_key_projection','v_proj_target':'context_value_projection'}
    for layer,official in zip(draft.backbone.layers,ref.layers):
        official.input_layernorm.weight.copy_(layer.input_norm.weight)
        official.post_attention_layernorm.weight.copy_(layer.post_attention_norm.weight)
        official.mlp.load_state_dict(layer.mlp.state_dict())
        actual=dict(layer.self_attn.named_parameters())
        for name,p in official.self_attn.named_parameters():
            root,tail=name.split('.',1);p.copy_(actual[remap.get(root,root)+'.'+tail])

def inputs(context,noise,anchor,state=None,positions=None):
    return ModelInputs(context,noise,torch.tensor([[anchor]]),torch.ones(1,1,dtype=torch.bool),
        context_position_ids=(torch.arange(context.size(1))[None] if positions is None else positions),
        draft_state=state,mode='features')

def local_forward(context,noise,anchor,state=None,positions=None):
    return draft.backbone(draft.context(inputs(context,noise,anchor,state,positions))).draft_hidden_states

def official_forward(context,noise,anchor):
    return ns['forward'](ref,None,context.reshape(1,-1,5,32),torch.arange(anchor,anchor+8)[None],torch.arange(context.size(1))[None],noise_embedding=noise)

# Full five-layer forward and backward at a short sequence.
context=torch.randn(1,37,160,requires_grad=True)
noise=torch.randn(1,8,32,requires_grad=True)
cref=context.detach().clone().requires_grad_();nref=noise.detach().clone().requires_grad_()
got=local_forward(context,noise,37);want=official_forward(cref,nref,37)
results['full_short_fp32_forward_max_abs']=float((got-want).detach().abs().max())
torch.testing.assert_close(got,want,rtol=3e-5,atol=3e-6)
scale=torch.randn_like(got);(got*scale).sum().backward();(want*scale).sum().backward()
grads=[(context.grad,cref.grad),(noise.grad,nref.grad),(draft.context.layer_fusion_weights.grad,ref.layer_fusion_weights.grad)]
for a,b in grads:torch.testing.assert_close(a,b,rtol=3e-4,atol=1e-5)
results['full_short_gradient_max_abs']=max(float((a-b).abs().max()) for a,b in grads)
for anchor in (4096,32768):
    with torch.inference_mode():
        c=torch.randn(1,anchor,160);n=torch.randn(1,8,32)
        got=local_forward(c,n,anchor);want=official_forward(c,n,anchor)
        err=float((got-want).abs().max());results[f'full_{anchor}_fp32_forward_max_abs']=err
        torch.testing.assert_close(got,want,rtol=2e-4,atol=2e-5)
        state=DraftState()
        first=local_forward(c,n,anchor,state)
        torch.testing.assert_close(first,got,rtol=0,atol=0)
        appended=torch.randn(1,4,160);next_noise=torch.randn(1,8,32)
        cached=local_forward(appended,next_noise,anchor+4,state,torch.arange(anchor,anchor+4)[None])
        full=local_forward(torch.cat([c,appended],1),next_noise,anchor+4)
        results[f'cached_vs_full_{anchor}_fp32_max_abs']=float((cached-full).abs().max())
        torch.testing.assert_close(cached,full,rtol=2e-4,atol=2e-5)
        assert all(s.positions.shape[-1]==anchor+4 for s in state.layers.values())
        results[f'cache_positions_{anchor}']=True
draft=copy.deepcopy(draft).to(torch.bfloat16)
ref=SimpleNamespace(layer_fusion_weights=nn.Parameter(ref.layer_fusion_weights.detach().to(torch.bfloat16)),
    context_norm=copy.deepcopy(ref.context_norm).to(torch.bfloat16),
    layers=copy.deepcopy(ref.layers).to(torch.bfloat16), final_norm=copy.deepcopy(ref.final_norm).to(torch.bfloat16))
for anchor in (37,32768):
    with torch.inference_mode():
        c=torch.randn(1,anchor,160,dtype=torch.bfloat16);n=torch.randn(1,8,32,dtype=torch.bfloat16)
        got=local_forward(c,n,anchor);want=official_forward(c,n,anchor)
        results[f'full_{anchor}_bf16_forward_max_abs']=float((got-want).abs().max())
        torch.testing.assert_close(got,want,rtol=.02,atol=.02)
assert not torch.cuda.is_initialized()
results['status']='passed'
results['scope']='CPU tiny hidden32, full5layer, actual learned fusion25 scalars; no production-sized GPU/kernel equivalence claim'
(HERE/'numerical_equivalence.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
