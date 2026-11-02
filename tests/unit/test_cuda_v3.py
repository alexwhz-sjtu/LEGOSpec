import copy,json
from pathlib import Path
import pytest,torch
from test_attention_v3 import cfg,recipe,data
from litespec.modeling.draft.composable import ComposableDraftModel

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('hv',[16,32])
@pytest.mark.parametrize('mode',['block','confirmed_history'])
def test_gdn_grouped_fla_matches_reference(hv,mode):
    torch.manual_seed(22)
    r=recipe(('gdn',));r['attention_configs']['gdn']=dict(linear_num_key_heads=16,linear_num_value_heads=hv,
        query_conv_kernel_size=2 if mode=='block' else 4,query_conv_mode=mode)
    c=cfg(composable_config=r,execution_config={'gdn_backend':'reference','attention_backend':'sdpa'})
    reference=ComposableDraftModel(c).cuda().bfloat16()
    c=copy.deepcopy(c);c.execution_config['gdn_backend']='fla';c.validate_recipe()
    fast=ComposableDraftModel(c).cuda().bfloat16();fast.load_state_dict(reference.state_dict())
    x=data(anchor=5)
    x.target_hidden_states=x.target_hidden_states.cuda().bfloat16();x.draft_embeddings=x.draft_embeddings.cuda().bfloat16()
    x.anchor_positions=x.anchor_positions.cuda();x.block_keep_mask=x.block_keep_mask.cuda()
    x.context_position_ids=x.context_position_ids.cuda();x.document_ids=x.document_ids.cuda();x.anchor_document_ids=x.anchor_document_ids.cuda()
    a=reference(x).draft_hidden_states;b=fast(x).draft_hidden_states
    torch.testing.assert_close(a.float(),b.float(),atol=.02,rtol=.03)
    a.float().square().mean().backward();b.float().square().mean().backward()
    worst=0.
    for (name,p),(other,q) in zip(reference.named_parameters(),fast.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:
            assert torch.isfinite(q.grad).all()
            norm=p.grad.float().norm().item();error=(p.grad.float()-q.grad.float()).norm().item()
            if norm>1e-5:worst=max(worst,error/norm)
    assert worst<.08,worst
    print(json.dumps(dict(test='gdn_fla',v_heads=hv,mode=mode,forward_max_error=(a.float()-b.float()).abs().max().item(),worst_parameter_relative_gradient_error=worst)),flush=True)

@pytest.mark.parametrize('kind',['mha','gqa','mla'])
@pytest.mark.parametrize('backend',['eager','flex_attention'])
def test_softmax_cuda_backend_forward_backward(kind,backend):
    torch.manual_seed(31)
    r=recipe((kind,))
    if kind in ('mha','gqa'):r['attention_configs'][kind]={'head_dim':16}
    else:r['attention_configs'][kind]={} # real default QK192 / V128
    a=ComposableDraftModel(cfg(composable_config=r)).cuda().bfloat16()
    c=cfg(composable_config=r,execution_config={'attention_backend':backend,'gdn_backend':'reference'})
    b=ComposableDraftModel(c).cuda().bfloat16();b.load_state_dict(a.state_dict())
    x=data(anchor=5)
    for name in ('target_hidden_states','draft_embeddings'):setattr(x,name,getattr(x,name).cuda().bfloat16())
    for name in ('anchor_positions','block_keep_mask','context_position_ids','document_ids','anchor_document_ids'):setattr(x,name,getattr(x,name).cuda())
    y=a(x).draft_hidden_states;z=b(x).draft_hidden_states
    torch.testing.assert_close(y.float(),z.float(),atol=.02,rtol=.03)
    z.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in b.parameters() if p.grad is not None)
    print(json.dumps(dict(test='softmax_backend',attention=kind,backend=backend,max_error=(y.float()-z.float()).abs().max().item())),flush=True)

@pytest.mark.parametrize('kind',['gqa','mla','gdn','dsa'])
def test_cuda_padding_rows_have_finite_gradients(kind):
    from litespec.modeling.draft.composable import ModelInputs
    if kind=='dsa':r=recipe(('gqa','mha'),['dsa','dsa'])
    elif kind=='gdn':
        r=recipe(('gdn','gqa'));r['attention_configs']['gdn']={}
    else:r=recipe((kind,))
    m=ComposableDraftModel(cfg(composable_config=r,execution_config={'gdn_backend':'fla','attention_backend':'sdpa'})).cuda().bfloat16()
    x=ModelInputs(torch.randn(1,6,64,device='cuda',dtype=torch.bfloat16),
        torch.randn(1,8,32,device='cuda',dtype=torch.bfloat16),torch.tensor([[2,5]],device='cuda'),
        torch.tensor([[True,False]],device='cuda'),mode='features')
    y=m(x).draft_hidden_states
    assert torch.isfinite(y).all()
    y[:,:1].float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
