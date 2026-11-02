"""Native Qwen3.5 teacher with exact recurrent-state rollback per verified token."""
from types import SimpleNamespace
import torch
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
_CONFIG=None

def load_target(path,device):
 target,info=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
 if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('Native teacher load mismatch: '+str(info))
 return target.to(device).eval()

class RollbackCache(Qwen3_5DynamicCache):
 def __init__(self):
  assert _CONFIG is not None
  super().__init__(_CONFIG);self.snapshots={}
 def remember(self):
  self.snapshots[self.get_seq_length()]=tuple([None if t is None else t.clone() for t in states] for states in [self.conv_states,self.recurrent_states])
 def crop(self,length):
  assert length in self.snapshots,('Missing recurrent rollback state',length,list(self.snapshots))
  conv,recurrent=self.snapshots[length]
  self.conv_states=[None if t is None else t.clone() for t in conv];self.recurrent_states=[None if t is None else t.clone() for t in recurrent]
  for i in self.transformer_layers:
   self.key_cache[i]=self.key_cache[i][...,:length,:];self.value_cache[i]=self.value_cache[i][...,:length,:]
  self.snapshots={length:(conv,recurrent)}
  assert self.get_seq_length()==length

class TargetFeatures:
 def __init__(self,target,layer_ids):
  self.target,self.ids,self.values=target,list(layer_ids),{};self.handles=[]
  layers=target.model.language_model.layers
  assert self.ids==[3,11,19,23,27]
  for i in self.ids:
   def capture(module,args,output,index=i):self.values[index]=output[0] if isinstance(output,tuple) else output
   self.handles.append(layers[i].register_forward_hook(capture))
 def one(self,ids,cache,last_only):
  self.values.clear();out=self.target(input_ids=ids,past_key_values=cache,use_cache=True,output_hidden_states=False,logits_to_keep=1 if last_only else 0)
  context=torch.cat([self.values[i] for i in self.ids],-1);cache.remember();return out,context
 def forward(self,ids,cache,last_only=False):
  # HF5.2's GDN continuation kernel consumes previous state only for seq_len=1.
  if cache.get_seq_length()==0 or ids.shape[1]==1:return self.one(ids,cache,last_only)
  logits=[];features=[]
  for j in range(ids.shape[1]):
   out,context=self.one(ids[:,j:j+1],cache,False);logits.append(out.logits);features.append(context)
  return SimpleNamespace(logits=torch.cat(logits,1)),torch.cat(features,1)
 def close(self):
  for handle in self.handles:handle.remove()

def configure(target):
 global _CONFIG
 _CONFIG=target.config.text_config
 import engine
 engine.DynamicCache=RollbackCache;engine.TargetFeatures=TargetFeatures
 import litespec.modeling.draft.composable.eagle3_inference as eagle
 eagle.DynamicCache=RollbackCache;eagle.TargetFeatures=TargetFeatures
