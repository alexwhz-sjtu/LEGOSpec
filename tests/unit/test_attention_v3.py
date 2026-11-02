import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel, ModelInputs
from litespec.modeling.draft.composable.contracts import DraftState, SharedOps
from litespec.modeling.draft.composable.schema_v3 import FIELDS
from litespec.training.recipes.composable import ADAPTER


def recipe(types=('gqa',), accesses=None, **updates):
    accesses=accesses or ['linear' if t=='gdn' else 'dense' for t in types]
    profiles={t:dict({'gdn':dict(linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8),
                     'mla':dict(num_attention_heads=2,num_key_value_heads=2,q_lora_rank=None,kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=8)}.get(t,{})) for t in types}
    r=dict(context=dict(type='dflash',target_layer_ids=[0,2]),
           backbone=dict(type='layer_stack',layers=[dict(attention=t,access=a) for t,a in zip(types,accesses)]),
           proposal=dict(type='parallel'), objective=dict(type='hard_ce',loss_decay_gamma=4.),
           access_configs={a:({'window_size':3} if a=='sliding_window' else {}) for a in set(accesses)},attention_configs=profiles)
    r.update(updates);return r


def cfg(types=('gqa',), accesses=None, **kwargs):
    r=kwargs.pop('composable_config',None) or recipe(types,accesses)
    return ComposableDraftConfig(hidden_size=32,intermediate_size=48,vocab_size=37,head_dim=8,
        num_attention_heads=4,num_key_value_heads=2,num_hidden_layers=len(r['backbone']['layers']),
        num_target_layers=4,block_size=4,max_position_embeddings=128,
        execution_config=kwargs.pop('execution_config',dict(attention_backend='sdpa',gdn_backend='reference')),
        composable_config=r,**kwargs)


def data(anchor=3, features=None, embeddings=None, start=0, state=None, doc=0):
    features=torch.randn(1,anchor-start,64) if features is None else features
    embeddings=torch.randn(1,4,32) if embeddings is None else embeddings
    return ModelInputs(features,embeddings,torch.tensor([[anchor]]),torch.ones(1,1,dtype=torch.bool),
        context_position_ids=torch.arange(start,anchor)[None],document_ids=torch.full((1,anchor-start),doc),
        anchor_document_ids=torch.tensor([[doc]]),draft_state=state,mode='features')


PATTERNS=[('mha',),('gqa',),('mla',),('gdn',),('gdn','gqa','mla','mha'),('mla','gdn','gqa'),('gqa','gdn','gdn'),('gdn','mla','gdn')]

@pytest.mark.parametrize('types',PATTERNS)
def test_arbitrary_order_forward_backward_and_independent_parameters(types):
    torch.manual_seed(4);m=ComposableDraftModel(cfg(types))
    x=data();out=m(x).draft_hidden_states
    assert out.shape==(1,1,3,32) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert len({p.data_ptr() for p in m.parameters()})==len(list(m.parameters()))
    for t in set(types):
        assert len(m.config.composable_config['attention_configs'][t])>0
    for layer in m.config.composable_config['backbone']['layers']:
        assert set(layer)=={'attention','access'}


@pytest.mark.parametrize('types',PATTERNS)
def test_incremental_document_reset_and_read_only_slots(types):
    torch.manual_seed(8);m=ComposableDraftModel(cfg(types)).eval()
    features=torch.randn(1,5,64);emb=torch.randn(1,4,32);state=DraftState()
    m(data(3,features[:,:3],emb,state=state))
    got=m(data(5,features[:,3:],emb,start=3,state=state)).draft_hidden_states
    full=m(data(5,features,emb)).draft_hidden_states
    torch.testing.assert_close(got,full,atol=2e-6,rtol=2e-5)
    for item in state.layers.values():assert item.key_states.shape[2]==5
    # Rejected draft content is absent from state; repeated current block agrees.
    again=m(data(5,features[:,:0],emb,start=5,state=state)).draft_hidden_states
    torch.testing.assert_close(again,full,atol=2e-6,rtol=2e-5)
    new=torch.randn(1,2,64)
    reset=m(data(2,new,emb,state=state,doc=1)).draft_hidden_states
    oracle=m(data(2,new,emb,doc=1)).draft_hidden_states
    torch.testing.assert_close(reset,oracle,atol=2e-6,rtol=2e-5)
    if 'gdn' in types:
        with pytest.raises(ValueError,match='cannot crop'):state.rollback(1)


@pytest.mark.parametrize('types',[('gdn','gqa'),('mla','gdn'),('gqa','mha')])
def test_cache_failure_is_atomic(types):
    m=ComposableDraftModel(cfg(types)).eval();state=DraftState();features=torch.randn(1,4,64)
    m(data(2,features[:,:2],state=state))
    original_layers=state.layers;original_gdn=state.backbone_state
    saved={i:s.key_states.clone() for i,s in state.layers.items()}
    with mock.patch.object(m.backbone.layers[-1],'forward',side_effect=RuntimeError('injected')):
        with pytest.raises(RuntimeError,match='injected'):m(data(4,features[:,2:],start=2,state=state))
    assert state.layers is original_layers and state.backbone_state is original_gdn
    for i,value in saved.items():torch.testing.assert_close(state.layers[i].key_states,value,rtol=0,atol=0)


@pytest.mark.parametrize('types',[('gqa','mla'),('gdn','gqa'),('mla','gdn','mha')])
def test_future_and_anchor_isolation_and_padding_gradients(types):
    torch.manual_seed(5);m=ComposableDraftModel(cfg(types))
    f=torch.randn(1,7,64);e=torch.randn(1,8,32)
    x=ModelInputs(f,e,torch.tensor([[2,5]]),torch.tensor([[True,False]]),mode='features')
    first=m(x).draft_hidden_states[:,:1]
    x.target_hidden_states=f.clone();x.target_hidden_states[:,2:]+=100
    x.draft_embeddings=e.clone();x.draft_embeddings[:,4:]+=100
    changed=m(x).draft_hidden_states[:,:1]
    torch.testing.assert_close(first,changed,atol=2e-6,rtol=2e-5)
    changed.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_defaults_head_semantics_and_override():
    c=cfg(('mha','gqa','gdn','mla'))
    assert c.composable_config['attention_configs']['mha']['num_key_value_heads']==4
    assert c.composable_config['attention_configs']['gqa']['num_key_value_heads']==2
    # Explicit defaults test without reduced test profiles.
    r=recipe(('gdn','mla'));r['attention_configs']={'gdn':{},'mla':{}}
    m=ComposableDraftModel(cfg(composable_config=r));g=m.backbone.layers[0].mixer;a=m.backbone.layers[1].self_attn
    assert (g.hk,g.hv,g.dk,g.dv,g.q_kernel,g.context_kernel)==(16,32,128,128,2,4)
    assert (a.num_heads,a.qk_head_dim,a.v_head_dim,a.kv_lora_rank,a.q_lora_rank)==(16,192,128,512,None)
    out=m(data()).draft_hidden_states;out.square().mean().backward()
    assert torch.isfinite(out).all()
    r=recipe(('mla',));r['attention_configs']['mla'].update(q_lora_rank=6,num_attention_heads=3,num_key_value_heads=3)
    a=ComposableDraftModel(cfg(composable_config=r)).backbone.layers[0].self_attn
    assert a.q_b_proj.weight.shape==(24,6)


def test_swa_incremental_and_shared_window():
    m=ComposableDraftModel(cfg(('gqa','mla'),['sliding_window']*2)).eval();state=DraftState()
    f=torch.randn(1,6,64);e=torch.randn(1,4,32)
    m(data(4,f[:,:4],e,state=state))
    got=m(data(6,f[:,4:],e,start=4,state=state)).draft_hidden_states
    full=m(data(6,f,e)).draft_hidden_states
    torch.testing.assert_close(got,full,atol=2e-6,rtol=2e-5)
    assert all(s.key_states.shape[2]<=3 for s in state.layers.values())


def test_dsa_single_index_shared_selection_and_indexer_only_gradients():
    r=recipe(('mha','gqa','gqa'),['dsa']*3);r['access_configs']['dsa']={'index_topk':2,'num_query_heads':2,'head_dim':8,'rope_head_dim':4}
    m=ComposableDraftModel(cfg(composable_config=r));x=data();x.prediction_mask=torch.ones(1,1,3);x.collect_indexer_teacher=True
    with mock.patch.object(m.backbone.indexer,'forward',wraps=m.backbone.indexer.forward) as run:
        out=m(x)
    assert run.call_count==1 and torch.isfinite(out.indexer_aux.numerator)
    out.indexer_aux.numerator.backward()
    assert any(p.grad is not None for p in m.backbone.indexer.parameters())
    assert all(p.grad is None or p.grad.count_nonzero()==0 for n,p in m.named_parameters() if not n.startswith('backbone.indexer.'))


def test_dsa_full_selection_matches_dense():
    torch.manual_seed(9)
    d=ComposableDraftModel(cfg(('gqa','mha'))).eval()
    r=recipe(('gqa','mha'),['dsa']*2);s=ComposableDraftModel(cfg(composable_config=r)).eval()
    info=s.load_state_dict(d.state_dict(),strict=False)
    assert all(k.startswith('backbone.indexer.') for k in info.missing_keys) and not info.unexpected_keys
    x=data();a=d(x).draft_hidden_states;b=s(x).draft_hidden_states
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('case',list(range(22)))
def test_strict_rejection(case):
    r=recipe();execution=dict(attention_backend='sdpa',gdn_backend='reference')
    if case==0:r['backbone']['layers']=[{}]
    elif case==1:r['backbone']['layers'][0]['head_dim']=16
    elif case==2:r['backbone']['layers'][0]['attention']='gdn'
    elif case==3:r['backbone']['layers'][0]['access']='linear'
    elif case==4:r['attention_configs']['gqa']['num_attention_heads']=2
    elif case==5:r['attention_configs']['gqa']['num_key_value_heads']=3
    elif case==6:r['attention_configs']['gqa']['head_dim']=True
    elif case==7:r['attention_configs']['gqa']['backend']='eager'
    elif case==8:r['extra']=1
    elif case==9:r['proposal']['rank']=4
    elif case==10:r['objective']['extra']=1
    elif case==11:r=recipe(('mla',),['dsa'])
    elif case==12:r=recipe(('gqa','mha'),['dense','dsa'])
    elif case==13:r=recipe(('gqa',),['dsa']);r['proposal']={'type':'selector','top_k':3};r['objective']={'type':'selector_losses'}
    elif case==14:r=recipe(('gqa',),['dsa']);execution['attention_backend']='eager'
    elif case==15:r=recipe(('gdn',));r['attention_configs']['gdn']['linear_num_value_heads']=3
    elif case==16:r=recipe(('gdn',));r['attention_configs']['gdn']['query_conv_mode']='confirmed_history'
    elif case==17:r=recipe(('mha',));r['attention_configs']['mha']['num_key_value_heads']=2
    elif case==18:r=recipe(('mla',));r['attention_configs']['mla']['q_lora_rank']=0
    elif case==19:r=recipe(('mla',));r['attention_configs']['mla']['num_key_value_heads']=1
    elif case==20:r['proposal']={'type':'selector','selector_mode':'candidate_only'};r['objective']={'type':'selector_losses'}
    elif case==21:r['context']['target_layer_ids']=[True]
    with pytest.raises(ValueError):cfg(composable_config=r,execution_config=execution)


def test_old_schema_rejected_explicitly():
    with pytest.raises(ValueError,match='Only composable_v3'):cfg(schema_version='composable_v2')


def test_all_active_recipes_build_real_target_shapes_without_allocation():
    import litespec
    root=Path(litespec.__file__).resolve().parents[1]
    paths=sorted((root/'configs').glob('*.json'))+sorted((root/'configs/qwen35_9b').glob('[0-9][0-9].json'))
    assert paths
    for p in paths:
        native=p.parent.name=='qwen35_9b'
        target=Qwen3Config(hidden_size=4096 if native else 2560,intermediate_size=12288 if native else 9728,
            num_hidden_layers=32 if native else 36,num_attention_heads=16 if native else 32,
            num_key_value_heads=4 if native else 8,head_dim=256 if native else 128,vocab_size=248320 if native else 151936)
        args=SimpleNamespace(draft_config=p,attention_backend='sdpa',override_attention_backend=None,flex_attention_backend='TRITON',gdn_backend='auto',block_size=8,target_layer_ids=None)
        c=ADAPTER.build_config(args,source_config=target)
        with torch.device('meta'):m=ComposableDraftModel(c)
        assert len(m.backbone.layers)==len(c.composable_config['backbone']['layers'])
        assert 'indexer' not in c.to_dict()['composable_config']

@pytest.mark.parametrize('anchor',[0,1,2,4])
def test_confirmed_history_conv4_short_prefix_repeat_and_increment(anchor):
    r=recipe(('gdn','gqa','gdn'))
    r['attention_configs']['gdn'].update(query_conv_kernel_size=4,query_conv_mode='confirmed_history')
    m=ComposableDraftModel(cfg(composable_config=r)).eval()
    features=torch.randn(1,anchor+2,64);emb=torch.randn(1,4,32);state=DraftState()
    m(data(anchor,features[:,:anchor],emb,state=state))
    out=m(data(anchor+2,features[:,anchor:],emb,start=anchor,state=state)).draft_hidden_states
    full=m(data(anchor+2,features,emb)).draft_hidden_states
    torch.testing.assert_close(out,full,rtol=2e-5,atol=2e-6)
    repeated=ModelInputs(features,emb.repeat(1,2,1),torch.tensor([[anchor,anchor]]),torch.ones(1,2,dtype=torch.bool),mode='features')
    out=m(repeated).draft_hidden_states
    torch.testing.assert_close(out[:,0],out[:,1],rtol=0,atol=0)
    for layer in state.backbone_state.layers.values():assert layer.q_conv_state.shape[-1]==3


def test_shared_config_roundtrip_and_independent_export(tmp_path):
    import os,subprocess,sys
    from litespec.modeling.draft.composable.export import export_composable
    m=ComposableDraftModel(cfg(('gdn','mla','mha'))).eval();x=data()
    m.save_pretrained(tmp_path/'saved')
    saved=json.loads((tmp_path/'saved/config.json').read_text())
    assert all(set(layer)=={'attention','access'} for layer in saved['composable_config']['backbone']['layers'])
    assert saved['target_attention']['num_key_value_heads']==2
    cold,info=ComposableDraftModel.from_pretrained(tmp_path/'saved',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys'))
    torch.testing.assert_close(m(x).draft_hidden_states,cold(x).draft_hidden_states,atol=0,rtol=0)
    export_composable(m,tmp_path/'export')
    code='''import sys,torch
from transformers import AutoModel
assert not any(k.startswith('litespec') for k in sys.modules)
m,info=AutoModel.from_pretrained(sys.argv[1],trust_remote_code=True,output_loading_info=True)
assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys'))
assert m.config.schema_version=='composable_v3'
assert len(m.backbone.layers)==3
T=type(__import__(m.__module__,fromlist=['ModelInputs']))
# The exported dataclass is obtained through the generated contracts module.
from pathlib import Path
import importlib
p=Path(sys.argv[1]); module=next(p.glob('cs_composable_contracts.py')).stem
contracts=importlib.import_module(m.__module__.rsplit('.',1)[0]+'.'+module)
x=contracts.ModelInputs(torch.randn(1,3,64),torch.randn(1,4,32),torch.tensor([[3]]),torch.ones(1,1,dtype=torch.bool),mode='features')
assert torch.isfinite(m(x).draft_hidden_states).all()
assert not any(k.startswith('litespec') for k in sys.modules)
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path/'export')],cwd=tmp_path,
        env=dict(os.environ,PYTHONPATH='',HF_MODULES_CACHE=str(tmp_path/'cache')),check=True,capture_output=True,text=True)


def test_manual_covers_validator_fields_and_html_is_self_contained():
    import litespec
    root=Path(litespec.__file__).resolve().parents[1]
    task=Path(__file__).parent
    coverage=json.loads((task/'manual_field_coverage.json').read_text())
    for name,allowed in FIELDS.items():assert set(coverage[name])==allowed,name
    md=(root/'docs/COMPOSABLE_CONFIG.md').read_text();html=(root/'docs/COMPOSABLE_CONFIG.html').read_text()
    assert '"access": "context_memory"' not in md and '<script' not in html and 'http://' not in html and 'https://' not in html
    assert all(x in html for x in ['linear','num_query_heads','loss_decay_gamma','full_vocab_residual','survival_weight'])

@pytest.mark.parametrize('stage',['warmup','sparse'])
def test_dsa_rnn_training_loss_and_chunk512_alignment(stage):
    from litespec.core.composable.training_model import ComposableTrainingModel
    from litespec.core.composable.batch import prepare_layout
    torch.manual_seed(12)
    r=recipe(('gqa','mha'),['dsa']*2,proposal={'type':'rnn','rank':8},
        objective={'type':'weighted_sum','loss_decay_gamma':4.,'components':[{'type':'hard_ce','weight':.1},{'type':'l1','weight':.9}]})
    r['access_configs']['dsa']={'stage':stage,'num_query_heads':2,'head_dim':8,'rope_head_dim':4,'index_topk':2}
    m=ComposableDraftModel(cfg(composable_config=r))
    if stage=='warmup':m.requires_grad_(False);m.backbone.indexer.requires_grad_(True)
    embed=nn.Embedding(37,32).requires_grad_(False);head=nn.Linear(32,37).requires_grad_(False)
    w=ComposableTrainingModel(m,head,embed,0,num_anchors=3,objective_chunk_blocks=512)
    ids=torch.randint(37,(1,10));mask=torch.ones(1,10);mask[:,-2:]=0
    layout=prepare_layout(ids,mask,4,3)
    values=dict(input_ids=ids,target_hidden_states=torch.randn(1,10,64),loss_mask=mask,
                target_last_hidden_states=torch.randn(1,10,32),layout=layout)
    full=w(**values).loss;full.backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    assert torch.isfinite(full) and grads
    if stage=='warmup':assert all(n.startswith('backbone.indexer.') for n in grads)
    else:assert any(n.startswith('proposal.markov_head.') and g.abs().sum()>0 for n,g in grads.items())
    m.zero_grad(set_to_none=True);w.objective_chunk_blocks=1
    chunk=w(**values).loss;chunk.backward()
    torch.testing.assert_close(full,chunk,rtol=2e-5,atol=2e-6)
    for n,p in m.named_parameters():
        if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=2e-4,atol=2e-6)


@pytest.mark.parametrize('proposal',['parallel','markov','rnn','sequential_rnn','selector'])
def test_all_proposals_real_conditional_sampling(proposal):
    torch.manual_seed(23)
    p={'type':proposal}
    if proposal!='parallel':p['rank']=8
    if proposal=='selector':p['top_k']=3
    objective={'type':'selector_losses'} if proposal=='selector' else {'type':'hard_ce'}
    m=ComposableDraftModel(cfg(composable_config=recipe(('gdn','mla','gqa'),proposal=p,objective=objective))).eval()
    x=data();x.mode='sample';x.temperature=1.;x.anchor_token_ids=torch.tensor([[1]])
    ops=SharedOps(nn.Embedding(37,32),nn.Linear(32,37))
    out=m(x,shared_ops=ops)
    assert out.draft_token_ids.shape==(1,1,3)
    assert out.draft_probabilities.shape==(1,1,3,37)
    torch.testing.assert_close(out.draft_probabilities.sum(-1),torch.ones(1,1,3),rtol=2e-6,atol=2e-6)
