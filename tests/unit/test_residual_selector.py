"""Final-only CE/L1 supervision keeps residual support and final gradients."""
from pathlib import Path
from copy import deepcopy
import sys,torch
from torch import nn
from transformers import AutoModel
from litespec.modeling.draft.composable import ComposableDraftConfig,ComposableDraftModel
from litespec.modeling.draft.composable.contracts import DraftState,DraftOutput
from litespec.modeling.draft.composable.export import export_composable
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.objectives.selector import LossBranches
from litespec.core.composable.objectives.base import Supervision
from litespec.core.composable.batch import position_weights
from _draft_fixtures import cfg,inputs
def config():
 d=cfg().to_dict();d['num_hidden_layers']=5;d['layer_types']=['full_attention']*5
 r=d['composable_config'];r['backbone']['layers']=[deepcopy(r['backbone']['layers'][0]) for _ in range(5)]
 r['proposal']=dict(type='residual_selector',rank=8,top_k=4,stop_gradient=False)
 r['objective']=dict(type='selector_losses',loss_decay_gamma=4.,base=dict(type='hard_ce',weight=0.),final=dict(type='weighted_sum',weight=1.,components=[dict(type='hard_ce',weight=.1),dict(type='l1',weight=.9)]))
 return ComposableDraftConfig(**d)
def test_l1_value_and_gradients_use_full_l1_with_no_base_branch():
 torch.manual_seed(40);b=LossBranches(config().composable_config)
 assert list(b.objectives)==['final'] and b.weights==dict(final=1.,base=0.)
 z=torch.randn(1,1,3,37,requires_grad=True);teacher=torch.randn_like(z,requires_grad=True);labels=torch.tensor([[[1,2,3]]]);mask=torch.ones_like(labels,dtype=torch.float)
 prediction=DraftOutput(torch.randn(1,1,3,32),draft_logits=z,base_logits=z+5.)
 out=b.branch_output('final',prediction,Supervision(labels,mask,target_logits=teacher))
 weights=position_weights(mask,4.);p=teacher.detach().softmax(-1);q=z.softmax(-1)
 ce=torch.nn.functional.cross_entropy(z.reshape(-1,37),labels.reshape(-1),reduction='none').reshape_as(mask)
 oracle=((.1*ce+.9*(p-q).abs().sum(-1))*weights).sum()/weights.sum()
 torch.testing.assert_close(out.loss,oracle)
 actual=torch.autograd.grad(out.loss,z,retain_graph=True)[0];expected=torch.autograd.grad(oracle,z)[0];torch.testing.assert_close(actual,expected)
 assert teacher.grad is None
def test_final_only_training_backpropagates_into_backbone_and_selector():
 torch.manual_seed(40);torch.set_num_threads(1);m=ComposableDraftModel(config())
 head=nn.Linear(32,37,bias=False).requires_grad_(False);emb=nn.Embedding(37,32).requires_grad_(False)
 wrapper=ComposableTrainingModel(m,head,emb,36,num_anchors=3,objective_chunk_blocks=2,complete_blocks=True,supervision_mode='all')
 assert list(wrapper.loss_branches.objectives)==['final']
 out=wrapper(input_ids=torch.randint(37,(1,9)),loss_mask=torch.ones(1,9),context_valid_mask=torch.ones(1,9,dtype=torch.bool),target_hidden_states=torch.randn(1,9,64),target_last_hidden_states=torch.randn(1,9,32))
 assert torch.isfinite(out.loss);out.loss.backward()
 for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
 assert any(p.grad.abs().sum()>0 for p in m.backbone.parameters())
 assert all(p.grad is None for p in head.parameters())
def test_residual_distribution_keeps_non_topk_support():
 m=ComposableDraftModel(config());z=torch.randn(1,3,37);h=torch.randn(1,3,32);prev=torch.tensor([[1,2,3]])
 logits,ids=m.proposal.full_vocab_logits(z,h,prev)
 outside=torch.ones_like(z,dtype=torch.bool).scatter(-1,ids,False)
 assert (logits.softmax(-1)[outside]>0).all()
 torch.testing.assert_close(logits[outside],z[outside])
def test_cache_replay_and_cold_export(tmp_path):
 torch.manual_seed(40);torch.set_num_threads(1);m=ComposableDraftModel(config()).eval()
 f=torch.randn(1,7,64);emb=torch.randn(1,4,32);state=DraftState()
 m(inputs(f[:,:3],emb,3,state=state));chunk=m(inputs(f[:,3:],emb,7,3,state)).draft_hidden_states
 full=m(inputs(f,emb,7)).draft_hidden_states;torch.testing.assert_close(chunk,full,atol=2e-6,rtol=2e-5)
 export_composable(m,tmp_path/'final');restored=AutoModel.from_pretrained(tmp_path/'final',trust_remote_code=True).eval()
 torch.testing.assert_close(restored(inputs(f,emb,7)).draft_hidden_states,full,rtol=0,atol=0)
