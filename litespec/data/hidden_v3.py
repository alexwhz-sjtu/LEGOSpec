"""Packed safetensors blocks and a backwards-compatible dataset payload reader."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.data import prepare_hidden_states as legacy


def durable_json(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,sort_keys=True)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class BlockSink:
    def __init__(self,args,root,identity,rank,metrics):
        self.args,self.root,self.identity,self.rank,self.metrics=args,root,identity,rank,metrics
        self.directory=root/'ranks'/f'rank_{rank:05d}'
        self.directory.mkdir(parents=True,exist_ok=True)
        (root/'shards').mkdir(exist_ok=True)
        self.records=[];self.nbytes=0;self.done=set();self.sequence=0
        for path in sorted(self.directory.glob('block_*.json')):
            meta=json.loads(path.read_text())
            assert args.resume and meta['input_identity']==identity,'Block belongs to another input'
            assert legacy.sha256_file(root/meta['path'])==meta['sha256'],'Committed block checksum mismatch'
            assert not self.done.intersection(meta['source_indices']),'Duplicate committed samples'
            self.done.update(meta['source_indices']);self.sequence=max(self.sequence,meta['sequence']+1)
        if args.resume:
            for path in (root/'shards').glob(f'.rank_{rank:05d}_*.tmp'):path.unlink()

    def add(self,index,payload,metadata):
        if index in self.done:raise ValueError(f'Duplicate committed source {index}')
        size=sum(v.numel()*v.element_size() for v in payload['hidden_states'].values())
        size+=payload['final_hidden_state'].numel()*payload['final_hidden_state'].element_size()
        if self.records and self.nbytes+size>1024**3:self.flush()
        self.records.append((index,payload,metadata));self.nbytes+=size

    def flush(self):
        if not self.records:return
        records=self.records
        name=f'rank_{self.rank:05d}_block_{self.sequence:06d}.safetensors'
        final=self.root/'shards'/name
        temporary=final.with_name('.'+name+'.tmp')
        with self.metrics.time('block_pack'):
            tensors={key:torch.cat([r[1][key] for r in records])
                     for key in ('input_ids','attention_mask','loss_mask')}
            for layer in (1,9,17,25,33,35):
                tensors[f'hidden_{layer}']=torch.cat([
                    r[1]['final_hidden_state'] if layer==35 else r[1]['hidden_states'][layer]
                    for r in records])
            offsets=[0]
            for _,payload,_ in records:offsets.append(offsets[-1]+payload['input_ids'].numel())
            tensors['offsets']=torch.tensor(offsets,dtype=torch.int64)
            tensors['source_indices']=torch.tensor([r[0] for r in records],dtype=torch.int64)
        self.publish(tensors,records,offsets,name,final,temporary)

    def publish(self,tensors,records,offsets,name,final,temporary):
        meta=self.write_block(self.sequence,tensors,records,offsets,final,temporary)
        self.done.update(meta['source_indices'])
        self.sequence+=1;self.records=[];self.nbytes=0

    def write_block(self,sequence,tensors,records,offsets,final,temporary):
        with self.metrics.time('block_write'):
            save_file(tensors,str(temporary),metadata={'schema_version':'3'})
        with self.metrics.time('fsync'):
            with temporary.open('rb') as handle:os.fsync(handle.fileno())
        digest=legacy.sha256_file(temporary)
        os.replace(temporary,final)
        fd=os.open(final.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        meta={'sequence':sequence,'input_identity':self.identity,
              'path':str(final.relative_to(self.root)),'sample_count':len(records),
              'source_indices':[r[0] for r in records],
              'sample_metadata':[r[2] for r in records], 'token_offsets':offsets,
              'tensor_dtypes':{key:str(value.dtype) for key,value in tensors.items()},'sha256':digest}
        metadata_path=self.directory/f'block_{sequence:06d}.json'
        durable_json(metadata_path,meta)
        return meta

    def finish(self):self.flush()


class ParallelBlockSink(BlockSink):
    """Independent atomic block commits with bounded IO work and byte quotas."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.executor=ThreadPoolExecutor(max_workers=self.args.writer_workers)
        self.pending=deque();self.pending_bytes=0

    def reap(self,wait=False):
        # Check all completed work before possibly waiting on the oldest block.
        for future,_ in self.pending:
            if future.done():future.result()
        while self.pending and (wait or self.pending[0][0].done()):
            future,size=self.pending.popleft()
            meta=future.result();self.done.update(meta['source_indices']);self.pending_bytes-=size
            if wait:break

    def add(self,*args):
        self.reap()
        return super().add(*args)

    def publish(self,tensors,records,offsets,name,final,temporary):
        size=sum(value.numel()*value.element_size() for value in tensors.values())
        # Reserve room for the active serialized batch, unfinished block and
        # freshly packed block, including token/mask and Python overhead.
        allowance=self.args.max_pending_bytes-int(4.25*1024**3)
        if size>allowance:raise MemoryError('Block exceeds pending commit byte budget')
        with self.metrics.time('block_queue_wait'):
            while self.pending and (len(self.pending)>=self.args.pipeline_depth or self.pending_bytes+size>allowance):
                self.reap(wait=True)
        compact_records=[(index,None,meta) for index,_,meta in records]
        future=self.executor.submit(self.write_block,self.sequence,tensors,compact_records,offsets,final,temporary)
        self.pending.append((future,size));self.pending_bytes+=size
        self.sequence+=1;self.records=[];self.nbytes=0

    def finish(self):
        try:
            self.flush()
            while self.pending:self.reap(wait=True)
        finally:self.executor.shutdown(wait=True,cancel_futures=True)


class StreamingBlockSink(BlockSink):
    """Copy directly into reusable per-layer block buffers; no per-sample clone/cat."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.buffers=None;self.used=0

    def add(self,index,payload,metadata):
        if index in self.done:raise ValueError(f'Duplicate committed source {index}')
        values={key:payload[key] for key in ('input_ids','attention_mask','loss_mask')}
        values.update({f'hidden_{layer}':value for layer,value in payload['hidden_states'].items()})
        values['hidden_35']=payload['final_hidden_state']
        length=payload['input_ids'].numel()
        if self.buffers is None:
            per_token=sum(value[0].numel()*value.element_size() for value in values.values())
            self.capacity=getattr(self.args,'block_bytes',1024**3)//per_token
            if self.capacity<length:raise MemoryError('Block budget smaller than one sample')
            self.buffers={key:torch.empty((self.capacity,*value.shape[1:]),dtype=value.dtype)
                          for key,value in values.items()}
        if self.used+length>self.capacity:self.flush()
        with self.metrics.time('block_copy'):
            for key,value in values.items():
                if self.buffers[key].dtype!=value.dtype:raise ValueError('Inconsistent block dtype')
                self.buffers[key][self.used:self.used+length].copy_(value)
        self.records.append((index,None,metadata));self.used+=length

    def flush(self):
        if not self.records:return
        with self.metrics.time('block_pack'):
            offsets=[0]
            for _,_,meta in self.records:offsets.append(offsets[-1]+meta['sequence_length'])
            tensors={key:value[:self.used] for key,value in self.buffers.items()}
            tensors['offsets']=torch.tensor(offsets,dtype=torch.int64)
            tensors['source_indices']=torch.tensor([r[0] for r in self.records],dtype=torch.int64)
        name=f'rank_{self.rank:05d}_block_{self.sequence:06d}.safetensors'
        final=self.root/'shards'/name;temporary=final.with_name('.'+name+'.tmp')
        self.publish(tensors,self.records,offsets,name,final,temporary)
        self.used=0


def finalize(root,identity,metadata,count):
    shards=[];indices=[];ordinal=0
    for path in sorted((root/'ranks').glob('rank_*/block_*.json')):
        block=json.loads(path.read_text())
        assert block['input_identity']==identity
        size=block['sample_count'];block['success_start']=ordinal;block['success_end']=ordinal+size-1
        ordinal+=size;indices.extend(block['source_indices']);shards.append(block)
    assert len(indices)==count and sorted(indices)==list(range(count)),'Incomplete or duplicate source coverage'
    manifest={**metadata,'schema_version':3,'cache_format':'safetensors-v3','complete':True,
              'input_identity':identity,'successful_samples':count,'failed_samples':0,
              'shard_size':None,'shards':shards}
    durable_json(root/'manifest.json',manifest)
    return manifest


class BlockArchive:
    def __init__(self,path):
        self.handle=safe_open(str(path),framework='pt',device='cpu')
        self.handle.__enter__()

    def payload(self,offset,length,index):
        end=offset+length
        value={key:self.handle.get_slice(key)[offset:end]
               for key in ('input_ids','attention_mask','loss_mask')}
        value.update(schema_version=2,sample_index=index,
                     hidden_states={layer:self.handle.get_slice(f'hidden_{layer}')[offset:end]
                                    for layer in (1,9,17,25,33)},
                     final_hidden_state=self.handle.get_slice('hidden_35')[offset:end])
        return value

    def close(self):self.handle.__exit__(None,None,None)
