"""Bounded asynchronous hidden capture, with legacy-compatible tar output."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import torch
import torch.distributed as dist

from scripts.data import prepare_hidden_states as legacy


def parse_args(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--target-model-backend',choices=['hf','sglang'],default='sglang')
    p.add_argument('--capture-mode',choices=['sync','async'],default='async')
    p.add_argument('--cache-format',choices=['tar-v2','tar-indexed-v2','tar-raw-v1','safetensors-v3'],default='tar-raw-v1')
    p.add_argument('--pipeline-depth',type=int,default=2)
    p.add_argument('--writer-workers',type=int,default=2)
    p.add_argument('--max-pending-bytes',type=int,default=8*1024**3)
    p.add_argument('--max-batch-tokens',type=int,default=0)
    p.add_argument('--packed-requests',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--batch-order',choices=['original','bucket'],default='original')
    p.add_argument('--tar-shard-bytes',type=int,default=2*1024**3)
    p.add_argument('--tar-stream-sha',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--tar-async-commit',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--tar-commit-workers',type=int,default=1)
    p.add_argument('--tar-pending-commits',type=int,default=2)
    p.add_argument('--tar-buffer-bytes',type=int,default=4*1024**2)
    p.add_argument('--clear-cache',action='store_true')
    p.add_argument('--no-buffer-reuse',action='store_true')
    p.add_argument('--bulk-packed-transfer',action='store_true')
    p.add_argument('--direct-block-pack',action='store_true')
    p.add_argument('--parallel-block-writes',action='store_true')
    p.add_argument('--preallocate-pinned',action=argparse.BooleanOptionalAction,default=True)
    if argv is not None and "--help" in argv:
        p.print_help()
    own,rest=p.parse_known_args(argv)
    # Forward the resolved backend before legacy validation (including TP).
    args=legacy.parse_args(['--target-model-backend',own.target_model_backend,*rest])
    for key,value in vars(own).items():setattr(args,key,value)
    if min(args.pipeline_depth,args.writer_workers,args.max_pending_bytes,args.tar_buffer_bytes,args.tar_commit_workers,args.tar_pending_commits)<=0:
        p.error('pipeline and writer limits must be positive')
    if (args.packed_requests or args.max_batch_tokens) and args.target_model_backend!='sglang':
        p.error('Packed capture requires SGLang; select --target-model-backend sglang or disable packed capture with --no-packed-requests and --max-batch-tokens 0')
    if args.max_batch_tokens<0:p.error('token budget must be nonnegative')
    if args.tar_shard_bytes<0:p.error('Tar shard byte limit must be nonnegative')
    if (args.max_batch_tokens or args.packed_requests or args.batch_order=='bucket') and args.cache_format not in ('safetensors-v3','tar-indexed-v2','tar-raw-v1'):
        p.error('Variable-length capture requires an indexed cache')
    if args.direct_block_pack and args.cache_format!='safetensors-v3':
        p.error('Direct block packing requires v3')
    if args.parallel_block_writes and (args.cache_format!='safetensors-v3' or args.direct_block_pack):
        p.error('Parallel block writes require v3 and cannot reuse the direct packing buffer')
    return args


class Slot:
    def __init__(self):
        self.layers={}
        self.valid=torch.empty((),dtype=torch.bool,pin_memory=True)

    def preallocate(self,tokens,hidden_size,bulk):
        count=tokens*hidden_size
        self.flat=torch.empty(count*6,dtype=torch.bfloat16,pin_memory=True)
        if bulk:
            self.layers={'aux':self.flat[:count*5],'last':self.flat[count*5:]}
        else:self.layers={layer:self.flat[i*count:(i+1)*count] for i,layer in enumerate((1,9,17,25,33,35))}

    @property
    def allocated_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.layers.values())

    def allocate(self,hidden,limit):
        required=sum(t.numel()*t.element_size() for t in hidden.values())
        if required>limit:raise MemoryError(f'batch needs {required} pinned bytes; budget {limit}')
        # Release unused capacities before growing; never let a slot grow unbounded.
        for key,tensor in hidden.items():
            existing=self.layers.get(key)
            if existing is None or existing.numel()<tensor.numel() or existing.dtype!=tensor.dtype:
                self.layers.pop(key,None)
                self.layers[key]=torch.empty(tensor.numel(),dtype=tensor.dtype,pin_memory=True)
        return {key:self.layers[key][:value.numel()].view(value.shape) for key,value in hidden.items()}


def metadata(record):
    active=record['loss_mask'].reshape(-1)>0.5
    padded=torch.cat((torch.zeros(1,dtype=torch.int8),active.to(torch.int8),torch.zeros(1,dtype=torch.int8)))
    changes=padded[1:]-padded[:-1]
    starts=torch.nonzero(changes==1).reshape(-1).tolist()
    ends=torch.nonzero(changes==-1).reshape(-1).tolist()
    return {'sequence_length':record['input_ids'].numel(),'supervised_token_count':int(active.sum()),
            'loss_mask_runs':[[int(a),int(b)] for a,b in zip(starts,ends)]}


class TarSink:
    def __init__(self,args,root,identity,rank,metrics):
        self.args,self.root,self.identity,self.rank,self.metrics=args,root,identity,rank,metrics
        existing=legacy.discover_staging(root,identity,rank) if args.resume else []
        self.done={i for shard in existing for i in shard.metadata['source_indices']}
        if args.resume:
            finals=legacy.valid_final_shards(root,args.shard_size,identity)
            self.done.update(i for shard in finals for i in shard.metadata['source_indices'])
        self.sequence=max((s.metadata['sequence'] for s in existing),default=-1)+1
        self.writer=None

    def add(self,index,value,meta):
        if self.writer is None:
            self.writer=legacy.TarShardWriter(self.root/'staging'/f'rank_{self.rank:05d}',
                'stage',self.sequence,self.args.compress,'staging',self.identity)
            self.writer.archive.copybufsize=self.args.tar_buffer_bytes
        self.writer.add_sample(index,value,sample_metadata=meta)
        if self.writer.sample_count==self.args.shard_size:
            self.commit({'rank':self.rank})

    def commit(self,extra):
        committed=self.writer.commit(extra)
        with self.metrics.time('fsync'):
            with committed.path.open('rb') as handle:os.fsync(handle.fileno())
            fd=os.open(committed.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        self.sequence+=1;self.writer=None

    def finish(self):
        if self.writer is not None:
            self.commit({'rank':self.rank,'final_local':True})
        legacy.write_rank_errors(self.root,self.rank,legacy.discover_staging(self.root,self.identity,self.rank))


class Pipeline:
    def __init__(self,args,sink,metrics):
        self.args,self.sink,self.metrics=args,sink,metrics
        self.stream=torch.cuda.Stream()
        self.free=queue.Queue()
        self.slots=[Slot() for _ in range(args.pipeline_depth)]
        if getattr(args,'preallocate_pinned',False) and not getattr(args,'no_buffer_reuse',False):
            budget=args.max_batch_tokens or args.batch_size*args.max_length
            tokens=min(budget,args.batch_size*args.max_length)
            hidden_size=getattr(args,'capture_hidden_size',2560)
            size=tokens*hidden_size*6*2
            rounded=1<<(size-1).bit_length()
            reserved=torch.cuda.memory.host_memory_stats().get('reserved_bytes.current',0)
            if rounded*args.pipeline_depth+reserved>args.max_pending_bytes:
                raise MemoryError('Rounded pinned pool allocation exceeds byte quota')
            for slot in self.slots:slot.preallocate(tokens,hidden_size,args.bulk_packed_transfer and bool(args.max_batch_tokens))
        for slot in self.slots:self.free.put(slot)
        self.work=queue.Queue(maxsize=args.pipeline_depth)
        self.error=None
        self.pool=ThreadPoolExecutor(max_workers=args.writer_workers)
        self.thread=threading.Thread(target=self.consume,name='hidden-writer',daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:raise RuntimeError('Background hidden writer failed') from self.error

    def submit(self,output,records):
        self.check()
        with self.metrics.time('queue_wait'):
            while True:
                self.check()
                try:slot=self.free.get(timeout=0.1);break
                except queue.Empty:pass
        hidden=legacy.normalize_hidden_states(output.hidden_states,None)
        if set(hidden)!={1,9,17,25,33,35}:
            raise legacy.HiddenStateError(f'unexpected capture layers: {sorted(hidden)}')
        bulk=bool(getattr(self.args,'bulk_packed_transfer',False) and getattr(output,'packed',False))
        transport={'aux':output.packed_aux,'last':output.packed_last} if bulk else hidden
        try:
            buffers=slot.allocate(transport,self.args.max_pending_bytes//self.args.pipeline_depth)
            if bulk:
                viewed=buffers['aux'].view(-1,5,output.packed_last.shape[-1])
                cpu={layer:viewed[:,i,:] for i,layer in enumerate((1,9,17,25,33))}
                cpu[35]=buffers['last']
            else:cpu=buffers
            allocated=sum(s.allocated_bytes for s in self.slots)
            self.metrics.peak_pinned_bytes=max(getattr(self.metrics,'peak_pinned_bytes',0),allocated)
        except BaseException:
            self.free.put(slot)
            raise
        with self.metrics.gpu('finite_gpu'):
            good=torch.ones((),device='cuda',dtype=torch.bool)
            if getattr(output,'packed',False):
                for value in transport.values():good.logical_and_(torch.isfinite(value).all())
            else:
                valid=output.attention_mask.bool().unsqueeze(-1)
                for value in hidden.values():good.logical_and_((torch.isfinite(value)|~valid).all())
        produced=torch.cuda.Event();produced.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(produced)
            with self.metrics.gpu('d2h_async',self.stream):
                for key,value in transport.items():
                    buffers[key].copy_(value.detach(),non_blocking=True)
                    value.record_stream(self.stream)
                slot.valid.copy_(good,non_blocking=True)
            ready=torch.cuda.Event();ready.record(self.stream)
        offsets=output.token_offsets if getattr(output,'packed',False) else None
        packet=(slot,cpu,ready,records,hidden,good,offsets)
        self.work.put(packet)
        if self.args.capture_mode=='sync':
            with self.metrics.time('sync_wait'):
                self.work.join()
            self.check()

    def serialize(self,cpu,record,index,offsets=None):
        length=record['input_ids'].numel()
        with self.metrics.time('serialize'):
            # Compact clones are necessary: torch.save otherwise serializes the
            # entire shared batch storage behind a seemingly small tensor view.
            compact=self.args.cache_format!='tar-raw-v1' and not getattr(self.args,'direct_block_pack',False)
            hidden={key:(value[index,:length] if offsets is None else value[offsets[index]:offsets[index+1]])
                    for key,value in cpu.items()}
            if compact:hidden={key:value.clone() for key,value in hidden.items()}
            payload={'schema_version':2,'sample_index':record['source_index'],
                     'input_ids':record['input_ids'].reshape(-1).clone(),
                     'attention_mask':record['attention_mask'].reshape(-1).clone(),
                     'loss_mask':record['loss_mask'].reshape(-1).clone(),
                     'hidden_states':{key:value for key,value in hidden.items() if key!=35},
                     'final_hidden_state':hidden[35]}
            if self.args.cache_format=='tar-raw-v1':
                from .hidden_tar_raw import RawSample
                value=RawSample(payload)
            elif self.args.cache_format in ('tar-v2','tar-indexed-v2'):
                buffer=io.BytesIO();torch.save(payload,buffer);value=buffer.getvalue()
            else:value=payload
            return record['source_index'],value,metadata(record)

    def consume(self):
        while True:
            packet=self.work.get()
            if packet is None:
                self.work.task_done();break
            slot,cpu,ready,records,gpu_refs,good,offsets=packet
            futures=[]
            try:
                with self.metrics.time('d2h_wait'):ready.synchronize()
                if self.error is not None:continue
                if not bool(slot.valid):raise legacy.HiddenStateError('captured NaN or Inf')
                futures=[self.pool.submit(self.serialize,cpu,record,i,offsets) for i,record in enumerate(records)]
                for future in futures:
                    index,value,meta=future.result()
                    self.sink.add(index,value,meta)
            except BaseException as error:
                self.error=error
            finally:
                # GPU storage is released only after its D2H event completed;
                # pinned storage is released only after all serializers finish.
                for future in futures:
                    try:future.result()
                    except BaseException:pass
                if getattr(self.args,'no_buffer_reuse',False):slot.layers.clear()
                self.free.put(slot)
                self.work.task_done()

    def finish(self):
        self.work.join();self.work.put(None);self.thread.join();self.pool.shutdown()
        self.check();self.sink.finish()


def run_generation(args,metrics):
    from datasets import load_dataset
    from transformers import AutoConfig,AutoTokenizer
    from litespec.args import SGLangBackendArgs
    from litespec.distributed import init_distributed,destroy_distributed,get_dp_group
    from litespec.modeling.target import get_training_target_model

    init_distributed(timeout=args.dist_timeout,tp_size=args.tp_size)
    rank=dist.get_rank(get_dp_group());world=dist.get_world_size(get_dp_group())
    root=Path(args.output_path);root.mkdir(parents=True,exist_ok=True)
    cpu_group=dist.new_group(backend='gloo')
    writer_init=legacy.TarShardWriter.__init__
    def buffered_init(writer,*values,**kwargs):
        writer_init(writer,*values,**kwargs)
        writer.archive.copybufsize=args.tar_buffer_bytes
    legacy.TarShardWriter.__init__=buffered_init
    identity=legacy.input_fingerprint(Path(args.data_path))
    try:
        header={'input_identity':identity,'model':args.target_model_path,'layers':[1,9,17,25,33,35],
                'max_length':args.max_length,'format':args.cache_format,'world_size':world}
        header_path=root/'capture_identity.json'
        if rank==0:
            if header_path.exists():
                assert args.resume and json.loads(header_path.read_text())==header,'Resume identity mismatch'
            elif any(root.iterdir()):raise ValueError('Nonempty output directory')
            else:legacy.atomic_write_json(header_path,header)
        dist.barrier()
        tokenizer=AutoTokenizer.from_pretrained(args.target_model_path)
        if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id or 0
        config=AutoConfig.from_pretrained(args.target_model_path)
        args.capture_hidden_size=config.hidden_size
        kwargs=SGLangBackendArgs.from_args(args).to_kwargs() if args.target_model_backend=='sglang' else {}
        target=get_training_target_model(args.target_model_path,backend=args.target_model_backend,
            torch_dtype=torch.bfloat16,device='cuda' if args.target_model_backend=='hf' else None,**kwargs)
        target.set_capture_layers([1,9,17,25,33,35])
        raw=load_dataset('json',data_files=args.data_path)['train']
        if args.num_samples is not None:raw=raw.select(range(min(len(raw),args.num_samples)))
        key=hashlib.sha256(json.dumps({'identity':identity,'max_length':args.max_length,
            'chat_template':args.chat_template,'model':args.target_model_path,
            'preformatted':args.is_preformatted,'num_samples':args.num_samples},sort_keys=True).encode()).hexdigest()
        def build():
            return legacy.build_training_dataset(raw,tokenizer,args.chat_template,max_length=args.max_length,
                shuffle_seed=None,num_proc=args.build_dataset_num_proc,
                cache_dir=str(Path(args.cache_dir)/'processed_dataset'),cache_key=key,
                is_preformatted=args.is_preformatted,capture_errors=True)
        processed=build() if dist.get_rank()==0 else None
        dist.barrier()
        if processed is None:processed=build()
        start,end=legacy.contiguous_range(len(processed),rank,world)
        if args.cache_format=='tar-v2':sink=TarSink(args,root,identity,rank,metrics)
        elif args.cache_format in ('tar-indexed-v2','tar-raw-v1'):
            from litespec.data.hidden_tar_stream import StreamTarSink
            sink=StreamTarSink(args,root,identity,rank,metrics)
        else:
            from litespec.data.hidden_v3 import BlockSink,StreamingBlockSink,ParallelBlockSink
            factory=ParallelBlockSink if args.parallel_block_writes else StreamingBlockSink if args.direct_block_pack else BlockSink
            sink=factory(args,root,identity,rank,metrics)
        pipeline=Pipeline(args,sink,metrics)
        pending=[]
        captured=0
        def capture(records):
            nonlocal captured
            if not records:return
            if args.packed_requests or args.max_batch_tokens:
                total=sum(record['input_ids'].numel() for record in records)
                metrics.batch_counts(total,total)
                with metrics.time('target_host_total'):
                    output=target.generate_capture_data(records)
            else:
                batch=legacy.collate_records(records,tokenizer.pad_token_id)
                with metrics.time('h2d_host'),metrics.gpu('h2d'):
                    gpu={key:value.cuda(non_blocking=True) for key,value in batch.items()}
                with metrics.time('target_host_total'):
                    output=target.generate_training_data(**gpu,return_logits=False)
            pipeline.submit(output,records)
            captured+=len(records)
            if captured%64<len(records):
                print(f'[rank {rank}] captured {captured}/{len(indices)}; queued={pipeline.work.qsize()}',flush=True)
            if args.clear_cache:torch.cuda.empty_cache()
        indices=list(range(start,end))
        if args.batch_order=='bucket':
            with metrics.time('batch_plan'):
                lengths=[processed[i]['input_ids'].numel() for i in range(len(processed))]
                # Greedy token-balanced assignment, followed by length ordering.
                bins=[[] for _ in range(world)];loads=[0]*world
                for i in sorted(range(len(lengths)),key=lambda i:(-lengths[i],i)):
                    owner=min(range(world),key=lambda r:(loads[r],r))
                    bins[owner].append(i);loads[owner]+=lengths[i]
                indices=sorted(bins[rank],key=lambda i:(lengths[i],i))
        pending_tokens=0
        for index in indices:
            if index in sink.done:continue
            with metrics.time('dataset_read'):
                row=processed[index]
            if row.get('preprocessing_error'):raise legacy.HiddenStateError(row['preprocessing_error'])
            record={key:row[key].reshape(-1) for key in ('input_ids','attention_mask','loss_mask')}
            record['source_index']=index
            length=record['input_ids'].numel()
            if args.max_batch_tokens and pending and pending_tokens+length>args.max_batch_tokens:
                capture(pending);pending=[];pending_tokens=0
            pending.append(record)
            pending_tokens+=length
            if len(pending)>=args.batch_size:capture(pending);pending=[];pending_tokens=0
        capture(pending)
        with metrics.time('writer_drain'):pipeline.finish()
        metrics.capture_end=time.perf_counter()
        print(f'[rank {rank}] capture drained; waiting for finalization',flush=True)
        dist.barrier(group=cpu_group)
        run_metadata={'save_mode':'regen_full','sample_index_version':1,'target_model':args.target_model_path,
            'target_backend':args.target_model_backend,'chat_template':args.chat_template,'max_length':args.max_length,
            'requested_hidden_layer_ids':[1,9,17,25,33],'stored_hidden_layer_ids':[1,9,17,25,33,35],
            'final_norm_layer_id':35,'num_hidden_layers':config.num_hidden_layers,'hidden_size':config.hidden_size,
            'hidden_dtype':'torch.bfloat16','compressed':False}
        if rank==0:
            print('[rank 0] finalization started',flush=True)
            if args.cache_format=='tar-v2':
                legacy.finalize_staging(root,identity,args.shard_size,args.compress,run_metadata)
            elif args.cache_format in ('tar-indexed-v2','tar-raw-v1'):
                from litespec.data.hidden_tar_stream import finalize
                with metrics.time('finalize'):finalize(root,identity,run_metadata,len(processed),metrics,raw=args.cache_format=='tar-raw-v1')
            else:
                from litespec.data.hidden_v3 import finalize
                with metrics.time('finalize'):finalize(root,identity,run_metadata,len(processed))
            from .capture_metrics import persist_cache
            persist_cache(root,metrics)
            print('[rank 0] manifest and data persisted',flush=True)
        dist.barrier(group=cpu_group)
    finally:
        destroy_distributed()
