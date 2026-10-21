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
dist.barrier()
dist.destroy_process_group()
if rank==0:print("ALL_8GPU_COMPUTE_AND_NCCL_PASSED",flush=True)
