"""CPU-only comparison against unmodified, pinned AngelSpec method bodies."""
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/data/temp/LiteSpec_exp')
sys.path.insert(0, str(ROOT))
torch.set_num_threads(2)
torch.manual_seed(20261017)
HERE = Path(__file__).parent
ns = dict(torch=torch, nn=nn, F=F, List=List, Optional=Optional, Tuple=Tuple)
common = ast.parse((HERE / 'dflash.py').read_text())
for node in common.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in {
        'DFlashRMSNorm', 'DFlashRotaryEmbedding', '_repeat_kv', '_rotate_half'
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'official_dflash.py', 'exec'), ns)
official = ast.parse((HERE / 'dflare.py').read_text())
for node in official.body:
    if isinstance(node, ast.ClassDef) and node.name == 'DFlareAttention':
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'official_dflare.py', 'exec'), ns)
    if isinstance(node, ast.ClassDef) and node.name == 'DFlareDraftModel':
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name in {'forward', '_init_fusion_weights'}:
                exec(compile(ast.Module(body=[method], type_ignores=[]), 'official_dflare.py', 'exec'), ns)

from transformers import Qwen3Config
from litespec.modeling.draft.composable.context.hidden_gated import HiddenGatedContext
from litespec.modeling.draft.composable.backbone.attention.mha import MHAAttention
from litespec.modeling.draft.composable.contracts import ModelInputs
from litespec.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS

results = {}
cfg = SimpleNamespace(hidden_size=32, num_hidden_layers=5, rms_norm_eps=1e-6,
                      composable_config={'context': {'target_layer_ids': [1, 9, 17, 25, 33]}})
current = HiddenGatedContext(cfg)
reference = SimpleNamespace(num_layers=5, num_target_layers=5,
                            layer_fusion_weights=nn.Parameter(torch.empty(5, 5)))
ns['_init_fusion_weights'](reference)
torch.testing.assert_close(current.layer_fusion_weights, reference.layer_fusion_weights, rtol=0, atol=0)
results['fusion_prior_exact'] = True

class Capture(nn.Module):
    def forward(self, draft_hidden, context_hidden, **kwargs):
        self.context = context_hidden
        return draft_hidden

for dtype in [torch.float32, torch.bfloat16]:
    local = copy.deepcopy(current).to(dtype)
    with torch.no_grad():
        local.layer_fusion_weights.add_(torch.randn_like(local.layer_fusion_weights) * .2)
        local.norm.weight.copy_(torch.randn_like(local.norm.weight) * .1 + 1)
    ref = SimpleNamespace(
        layer_fusion_weights=nn.Parameter(local.layer_fusion_weights.detach().clone()),
        context_norm=ns['DFlashRMSNorm'](32).to(dtype),
        layers=nn.ModuleList([Capture() for _ in range(5)]), final_norm=nn.Identity())
    ref.context_norm.weight.data.copy_(local.norm.weight)
    x = torch.randn(2, 6, 5, 32, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    got = local(SimpleNamespace(target_hidden_states=x.flatten(2))).context_hidden_states
    ns['forward'](ref, None, y, None, None, noise_embedding=torch.zeros(2, 8, 32, dtype=dtype))
    want = [layer.context for layer in ref.layers]
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    scales = [torch.randn_like(a) for a in got]
    sum((a * scale).float().sum() for a, scale in zip(got, scales)).backward()
    sum((a * scale).float().sum() for a, scale in zip(want, scales)).backward()
    for a, b in [(x.grad, y.grad), (local.layer_fusion_weights.grad, ref.layer_fusion_weights.grad),
                 (local.norm.weight.grad, ref.context_norm.weight.grad)]:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    results[f'fusion_forward_backward_exact_{dtype}'] = True

cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=5,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                  max_position_embeddings=64, rms_norm_eps=1e-6,
                  attention_bias=False, attention_dropout=0., rope_theta=10000.)
cfg._attn_implementation = 'eager'
local = MHAAttention(cfg, 0, DEFAULT_DFLASH_KERNELS,
                     {'backend': 'sdpa', 'access': 'dense'}, separate_context=True)
ref = ns['DFlareAttention'](cfg)
mapping = {'k_proj_target': 'context_key_projection', 'v_proj_target': 'context_value_projection'}
for name, param in ref.named_parameters():
    root, tail = name.split('.', 1)
    param.data.copy_(dict(local.named_parameters())[mapping.get(root, root) + '.' + tail])
ctx_pos = torch.arange(6)[None].expand(2, -1)
draft_pos = torch.arange(4, 12)[None].expand(2, -1)
inputs = ModelInputs(target_hidden_states=torch.zeros(2, 6, 32),
                     draft_embeddings=torch.zeros(2, 8, 32), anchor_positions=torch.full((2, 1), 4),
                     block_keep_mask=torch.ones(2, 1, dtype=torch.bool), context_position_ids=ctx_pos)
positions = torch.cat([ctx_pos, draft_pos], 1)
cos, sin = ref.rotary_emb(torch.zeros(2, 4, 8, 8), seq_len=14)
embeddings = (cos[0, 0][positions], sin[0, 0][positions])
visibility = local.access(inputs, ctx_pos, torch.ones_like(ctx_pos, dtype=torch.bool),
                          torch.zeros_like(ctx_pos), 8)
real_sdpa = F.scaled_dot_product_attention

def masked_sdpa(q, k, v, **kwargs):
    return real_sdpa(q, k, v, attn_mask=visibility, **kwargs)

ns['F'] = SimpleNamespace(scaled_dot_product_attention=masked_sdpa)
draft = torch.randn(2, 8, 32, requires_grad=True)
context = torch.randn(2, 6, 32, requires_grad=True)
draft_ref = draft.detach().clone().requires_grad_()
context_ref = context.detach().clone().requires_grad_()
got = local(draft, context, embeddings, inputs)
want = ref(draft_ref, context_ref, draft_pos, ctx_pos)
torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
scale = torch.randn_like(got)
(got * scale).sum().backward()
(want * scale).sum().backward()
max_grad_error = 0.
for a, b in [(draft.grad, draft_ref.grad), (context.grad, context_ref.grad)]:
    torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-6)
    max_grad_error = max(max_grad_error, (a - b).abs().max().item())
for name, p in ref.named_parameters():
    root, tail = name.split('.', 1)
    local_p = dict(local.named_parameters())[mapping.get(root, root) + '.' + tail]
    torch.testing.assert_close(local_p.grad, p.grad, rtol=1e-4, atol=3e-6)
    max_grad_error = max(max_grad_error, (local_p.grad - p.grad).abs().max().item())
results['attention_fp32_max_forward_error'] = (got - want).abs().max().item()
results['attention_fp32_max_gradient_error'] = max_grad_error
results['context_kv_storage_independent'] = local.k_proj.weight.data_ptr() != local.context_key_projection.weight.data_ptr()
(HERE / 'numerical_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
