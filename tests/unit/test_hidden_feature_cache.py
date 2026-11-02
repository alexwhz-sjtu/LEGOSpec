"""Round-trip both context/teacher streams and retain legacy cache loading."""
from contextlib import nullcontext
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from litespec.data.capture_engine import Pipeline
from litespec.data.hidden_cache import RegenFullDataset, RegenFullCollator
from litespec.data.hidden_tar_raw import RawSample, load_raw
from litespec.data.hidden_v3 import BlockArchive, BlockSink, StreamingBlockSink
from litespec.training.recipes import _dflash_common as common
from scripts.data.prepare_hidden_states import sample_payload


class Metrics:
    def time(self,*args):return nullcontext()


def payload(separate=True,length=3):
    raw=torch.full((length,4),2.,dtype=torch.bfloat16)
    teacher=torch.full_like(raw,7.)
    hidden={i:torch.full_like(raw,float(i)) for i in [1,9,17,25,33]}
    if separate:hidden[35]=raw
    return dict(schema_version=2,sample_index=0,input_ids=torch.arange(length),
        attention_mask=torch.ones(length,dtype=torch.long),loss_mask=torch.ones(length),
        hidden_states=hidden,final_hidden_state=teacher)


class HiddenFeatureCacheTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_legacy_and_new_raw_tar_roundtrip(self):
        for separate in [False,True]:
            p=payload(separate);encoded=RawSample(p)
            data=encoded.header+b''.join(bytes(part) for part in encoded.parts)
            got=load_raw(io.BytesIO(data),0,len(data))
            self.assertEqual(set(got["hidden_states"]),set(p["hidden_states"]))
            for k,v in p["hidden_states"].items():self.assertTrue(torch.equal(v,got["hidden_states"][k]))
            self.assertTrue(torch.equal(got["final_hidden_state"],p["final_hidden_state"]))

    def test_legacy_sample_serializer_preserves_separate_features(self):
        for separate in [False,True]:
            p=payload(separate)
            hidden=dict(p["hidden_states"])
            hidden.setdefault(35,p["final_hidden_state"])
            output=SimpleNamespace(hidden_states={k:v.unsqueeze(0) for k,v in hidden.items()},
                input_ids=p["input_ids"][None],attention_mask=p["attention_mask"][None],
                loss_mask=p["loss_mask"][None],
                final_hidden_state=p["final_hidden_state"][None] if separate else None)
            encoded=sample_payload(output,0,0,[1,9,17,25,33,35],35)
            got=torch.load(io.BytesIO(encoded),weights_only=False)
            self.assertEqual(35 in got["hidden_states"],separate)
            self.assertTrue(torch.equal(got["final_hidden_state"],p["final_hidden_state"]))

    def test_async_serializer_keeps_both_streams(self):
        for separate in [False,True]:
            for fmt in ["tar-v2","tar-indexed-v2","tar-raw-v1","safetensors-v3"]:
                with self.subTest(separate=separate,fmt=fmt):
                    p=payload(separate);cpu={k:v[None] for k,v in p["hidden_states"].items()}
                    if separate:cpu["teacher_final"]=p["final_hidden_state"][None]
                    else:cpu[35]=p["final_hidden_state"][None]
                    pipeline=Pipeline.__new__(Pipeline)
                    pipeline.args=SimpleNamespace(cache_format=fmt,direct_block_pack=False)
                    pipeline.metrics=Metrics()
                    record={k:p[k] for k in ["input_ids","attention_mask","loss_mask"]}
                    record["source_index"]=0
                    _,got,_=pipeline.serialize(cpu,record,0)
                    if isinstance(got,bytes):got=torch.load(io.BytesIO(got),weights_only=False)
                    elif isinstance(got,RawSample):
                        data=got.header+b''.join(bytes(x) for x in got.parts)
                        got=load_raw(io.BytesIO(data),0,len(data))
                    self.assertEqual(35 in got["hidden_states"],separate)
                    self.assertTrue(torch.equal(got["final_hidden_state"],p["final_hidden_state"]))

    def test_both_safetensor_writers_roundtrip_separate_and_legacy(self):
        for separate in [False,True]:
            for cls in [BlockSink,StreamingBlockSink]:
                with self.subTest(separate=separate,writer=cls.__name__), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                    args=SimpleNamespace(resume=False,block_bytes=65536)
                    sink=cls(args,Path(folder),{"test":True},0,Metrics())
                    p=payload(separate)
                    sink.add(0,p,{"sequence_length":3});sink.finish()
                    path=next((Path(folder)/"shards").glob("*.safetensors"))
                    reader=BlockArchive(path)
                    try:got=reader.payload(0,3,0)
                    finally:reader.close()
                    self.assertEqual(35 in got["hidden_states"],separate)
                    self.assertTrue(torch.equal(got["final_hidden_state"],p["final_hidden_state"]))
                    if separate:self.assertTrue(torch.equal(got["hidden_states"][35],p["hidden_states"][35]))

    def test_dataset_collator_and_teacher_routing(self):
        for separate in [False,True]:
            p=payload(separate)
            dataset=RegenFullDataset.__new__(RegenFullDataset)
            dataset.max_length=3;dataset.hidden_size=4;dataset.final_layer_id=35
            dataset.required_layer_ids={1,35};dataset._load_payload=lambda index:p
            item=dataset[0]
            expected=p["hidden_states"].get(35,p["final_hidden_state"])
            self.assertTrue(torch.equal(item["hidden_states"][35],expected))
            batch=RegenFullCollator(0,pad_to_length=5)([item])
            self.assertEqual(tuple(batch["final_hidden_state"].shape),(1,5,4))
            draft=SimpleNamespace(target_layer_ids=[35],config=SimpleNamespace(num_target_layers=36))
            with patch.object(common,"hidden_states_to_cuda",lambda x:x):
                context,teacher=common.prepare_dflash_family_features(batch,None,draft,True)
            self.assertTrue(torch.equal(context[0,:3],expected))
            self.assertTrue(torch.equal(teacher[0,:3],p["final_hidden_state"]))
            self.assertTrue(torch.equal(teacher[0,3:],torch.zeros(2,4,dtype=teacher.dtype)))


if __name__=="__main__":unittest.main()
