"""Both native capture layouts round-trip; corrupt/mixed profiles remain invalid."""
import io
from types import SimpleNamespace
import pytest
import torch
from litespec.data.capture_profiles import PROFILES,resolve_capture_profile
from litespec.data.hidden_tar_raw import RawSample,load_raw
from litespec.data.capture_engine import Pipeline,parse_args,build_training_dataset
from litespec.data import build_training_dataset as original_builder
from litespec.data.parse import GeneralParser
from litespec.data.template import TEMPLATE_REGISTRY

@pytest.mark.parametrize('name',list(PROFILES))
def test_exact_profile_dimensions_and_cli_layers(name):
 p=PROFILES[name];cfg=SimpleNamespace(num_hidden_layers=p.num_layers,hidden_size=p.hidden_size,vocab_size=p.vocab_size)
 assert resolve_capture_profile(cfg).captured_layers==p.captured_layers
 assert resolve_capture_profile(cfg,name,list(p.context_layers))==p
 with pytest.raises(ValueError):resolve_capture_profile(cfg,name,[0,1,2,3,4])
 with pytest.raises(ValueError):resolve_capture_profile(SimpleNamespace(num_hidden_layers=1,hidden_size=4,vocab_size=8),name)
 assert build_training_dataset is original_builder

@pytest.mark.parametrize('name',list(PROFILES))
@pytest.mark.parametrize('separate',[False,True])
def test_native_raw_roundtrip_and_pipeline_serialization(name,separate):
 p=PROFILES[name];layers={i:torch.full((2,3),float(i),dtype=torch.bfloat16) for i in p.captured_layers}
 teacher=torch.full((2,3),-1.,dtype=torch.bfloat16)
 hidden={i:t for i,t in layers.items() if separate or i!=p.final_layer}
 payload={'sample_index':3,'input_ids':torch.tensor([1,2]),'attention_mask':torch.ones(2,dtype=torch.long),'loss_mask':torch.ones(2),'hidden_states':hidden,'final_hidden_state':teacher}
 raw=RawSample(payload);data=raw.header+b''.join(bytes(x) for x in raw.parts);got=load_raw(io.BytesIO(data),0,len(data));assert set(got['hidden_states'])==set(hidden);assert torch.equal(got['final_hidden_state'],teacher)
 pipe=Pipeline.__new__(Pipeline);pipe.args=SimpleNamespace(cache_format='tar-raw-v1',capture_final_norm_layer_id=p.final_layer)
 class Metrics:
  def time(self,*a):
   from contextlib import nullcontext
   return nullcontext()
 pipe.metrics=Metrics();cpu={i:t[None] for i,t in layers.items()}
 if separate:cpu['teacher_final']=teacher[None]
 record={k:payload[k] for k in ['input_ids','attention_mask','loss_mask']};record['source_index']=3
 _,raw,_=pipe.serialize(cpu,record,0);data=raw.header+b''.join(bytes(x) for x in raw.parts);got=load_raw(io.BytesIO(data),0,len(data));assert torch.equal(got['final_hidden_state'],teacher if separate else layers[p.final_layer])

@pytest.mark.parametrize('bad',[{3,11,19,23},{1,11,19,23,27},{3,11,19,23,27,35}])
def test_missing_or_mixed_hidden_streams_are_rejected(bad):
 x=torch.ones(2,3,dtype=torch.bfloat16);v={'sample_index':0,'input_ids':torch.ones(2,dtype=torch.long),'attention_mask':torch.ones(2,dtype=torch.long),'loss_mask':torch.ones(2),'hidden_states':{i:x for i in bad},'final_hidden_state':x};r=RawSample(v);data=r.header+b''.join(bytes(x) for x in r.parts)
 with pytest.raises(ValueError,match='Missing or mixed'):load_raw(io.BytesIO(data),0,len(data))

def test_native_template_does_not_change_legacy_qwen():
 assert TEMPLATE_REGISTRY.get('qwen').system_prompt=='You are a helpful assistant.'
 template=TEMPLATE_REGISTRY.get('qwen35');assert template.system_prompt is None
 class Tokenizer:
  def apply_chat_template(self,messages,**kwargs):self.kwargs=kwargs;return 'rendered'
 t=Tokenizer();p=GeneralParser(t,template);p.apply_chat_template([]);assert t.kwargs['enable_thinking'] is False
 p.apply_chat_template([],enable_thinking=True);assert t.kwargs['enable_thinking'] is True
