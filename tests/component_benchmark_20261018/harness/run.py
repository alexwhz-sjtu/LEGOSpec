"""Resume-safe acceptance evaluation, one GPU per process; no timing claims."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from jsonl_io import read_jsonl


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for key in ['target','draft','manifest','output','source']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--temperature',type=float,choices=[0,1],required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--device',default='cuda:0');p.add_argument('--smoke-limit',type=int)
    p.add_argument('--max-new-tokens',type=int,default=512)
    args=p.parse_args()
    if args.smoke_limit is None and args.max_new_tokens!=512:
        p.error('Non-standard output budget is allowed only for labeled smoke runs')
    sys.path.insert(0,args.source)
    import torch
    torch.cuda.set_device(torch.device(args.device))
    from transformers import AutoModelForCausalLM,AutoTokenizer,Qwen3Config
    from litespec.modeling.draft.registry import resolve_draft,method_config
    from litespec.modeling.draft.composable import ComposableDraftConfig
    from engine import generate
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    with (root/f'shard_{args.shard}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model_dir=Path(args.draft)
        config_raw=json.loads((model_dir/'config.json').read_text())
        cls=resolve_draft(config_raw['architectures'][0])
        cfg=(ComposableDraftConfig if cls.method_config_key=='composable_config' else Qwen3Config).from_dict(config_raw)
        if cls.method_config_key=='composable_config':
            for layer in cfg.composable_config['backbone']['layers']:
                layer['attention']['backend']='sdpa'
        target=AutoModelForCausalLM.from_pretrained(args.target,torch_dtype=torch.bfloat16,
                      attn_implementation='sdpa',local_files_only=True).to(args.device).eval()
        draft,loading=cls.from_pretrained(args.draft,config=cfg,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
            raise ValueError('Checkpoint loading mismatch: '+str(loading))
        draft=draft.to(args.device).eval()
        tokenizer=AutoTokenizer.from_pretrained(args.target,local_files_only=True)
        mask=method_config(draft.config).get('mask_token_id',tokenizer.mask_token_id)
        if mask is None:raise ValueError('No trained mask token ID')
        draft.mask_token_id=int(mask)
        eos=target.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos)
        identity={'protocol':'component-acceptance-v1','manifest_sha256':digest(args.manifest),
            'draft':args.draft,'draft_hashes':{f.name:digest(f) for f in sorted(model_dir.glob('*.safetensors'))},
            'config_sha256':digest(model_dir/'config.json'),'target':args.target,
            'target_hashes':{f.name:digest(f) for f in sorted(Path(args.target).glob('*'))
                if f.suffix=='.safetensors' or f.name in ['config.json','generation_config.json']},
            'harness_hashes':{f.name:digest(f) for f in Path(__file__).parent.glob('*.py')},
            'temperature':args.temperature,'top_p':.95 if args.temperature else 1.,
            'max_new_tokens':args.max_new_tokens,'shards':args.shards,'smoke':args.smoke_limit is not None,
            'torch':torch.__version__,'python':platform.python_version(),'target_attention':'sdpa',
            'draft_attention':'sdpa','gpu':torch.cuda.get_device_name(torch.device(args.device))}
        metadata_path=Path(args.manifest).parent/'manifest_metadata.json'
        if args.smoke_limit is None:
            expected=json.loads(metadata_path.read_text())
            if expected['target_hashes']!=identity['target_hashes']:
                raise ValueError('Target files differ from the target used to freeze this panel')
            for name,want in expected['tokenizer_hashes'].items():
                if digest(Path(args.target)/name)!=want:
                    raise ValueError('Tokenizer/template differs: '+name)
            if expected['hashes'][Path(args.manifest).name]!=identity['manifest_sha256']:
                raise ValueError('Manifest changed after freeze')
        meta=root/f'metadata_{args.shard}.json'
        if meta.exists() and json.loads(meta.read_text())!=identity:
            raise ValueError('Existing result identity differs; use a new output directory')
        meta.write_text(json.dumps(identity,indent=2)+'\n')
        rows=list(read_jsonl(args.manifest))
        rows=[r for i,r in enumerate(rows) if i%args.shards==args.shard]
        if args.smoke_limit is not None:rows=rows[:args.smoke_limit]
        result=root/f'requests_{args.shard}.jsonl'; done=set()
        if result.exists():
            done={r['id'] for r in read_jsonl(result)}
        with result.open('a',buffering=1) as out:
            for row in rows:
                if row['id'] in done:continue
                print('REQUEST',row['id'],row['input_tokens'],flush=True)
                try:
                    ids=torch.tensor([row['input_ids']],device=args.device)
                    measured=generate(target,draft,ids,row['id'],args.temperature,eos,
                        max_new_tokens=args.max_new_tokens,top_p=.95)
                    record={k:row[k] for k in ['id','dataset','category','source_id','turn','input_tokens']}
                    record.update(measured,status='ok',temperature=args.temperature)
                    out.write(json.dumps(record,separators=(',',':'))+'\n');out.flush();os.fsync(out.fileno())
                    print('RESULT',record['id'],'tau',record['tau'],'cycles',len(record['cycles']),flush=True)
                except Exception:
                    error={'id':row['id'],'status':'failed','traceback':traceback.format_exc()}
                    (root/f'failure_{args.shard}.json').write_text(json.dumps(error,indent=2))
                    raise
        (root/f'complete_{args.shard}.json').write_text(json.dumps({'requests':len(rows),'status':'completed'}))


if __name__=='__main__':main()
