import datetime,importlib.metadata,json,os,sys
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
assert sys.version_info[:2]==(3,11),sys.version
assert torch.__version__.startswith('2.9.1+cu128'),torch.__version__
assert torch.cuda.device_count()==8
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=180))
x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
loss=(x@x.T).float().square().mean()
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
value=torch.tensor(float(rank+1),device='cuda')
dist.all_reduce(value)
assert value.item()==36,value
print('GPU_NCCL_PASS',json.dumps({'rank':rank,'gpu':torch.cuda.get_device_name(rank),'bf16_loss':loss.item(),'all_reduce':value.item()}),flush=True)
if rank==0:
 from causal_conv1d import causal_conv1d_fn
 import torch.nn.functional as F
 inputs=torch.randn(2,64,128,device='cuda',dtype=torch.float16,requires_grad=True)
 weight=torch.randn(64,4,device='cuda',dtype=torch.float16,requires_grad=True)
 bias=torch.randn(64,device='cuda',dtype=torch.float16,requires_grad=True)
 actual=causal_conv1d_fn(inputs,weight,bias,activation='silu')
 expected=F.silu(F.conv1d(inputs,weight[:,None],bias,padding=3,groups=64)[...,:128])
 torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
 actual.float().square().mean().backward()
 assert all(torch.isfinite(t.grad).all() for t in [inputs,weight,bias])
 print('CAUSAL_CONV1D_FORWARD_BACKWARD_PASS',flush=True)
 import sgl_kernel
 h=torch.randn(8,128,device='cuda',dtype=torch.bfloat16)
 w=torch.ones(128,device='cuda',dtype=torch.bfloat16)
 actual=sgl_kernel.rmsnorm(h,w,1e-6)
 expected=(h.float()*torch.rsqrt(h.float().square().mean(-1,keepdim=True)+1e-6)).to(h.dtype)
 torch.testing.assert_close(actual,expected,atol=.03,rtol=.03)
 print('SGL_KERNEL_RMSNORM_PASS',flush=True)
 import fla,sglang,transformers
 from torch.nn.attention.flex_attention import flex_attention
 q=torch.randn(1,4,64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
 k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
 actual=torch.compile(flex_attention,dynamic=False)(q,k,v)
 expected=F.scaled_dot_product_attention(q,k,v)
 torch.testing.assert_close(actual,expected,atol=.03,rtol=.03)
 actual.float().square().mean().backward()
 assert all(torch.isfinite(t.grad).all() for t in [q,k,v])
 print('FLEX_ATTENTION_TRITON_FORWARD_BACKWARD_PASS',flush=True)
 print('VERSIONS',json.dumps({p:importlib.metadata.version(p) for p in ['torch','transformers','sglang','sgl-kernel','causal-conv1d','flash-linear-attention']}),flush=True)
dist.barrier()
dist.destroy_process_group()
if rank==0:print('ALL_8GPU_VALIDATIONS_PASSED',flush=True)
