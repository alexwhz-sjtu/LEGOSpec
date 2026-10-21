"""Download immutable sources and freeze seed42 test / disjoint seed43 selection."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
from jsonl_io import read_jsonl


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def rank(seed, dataset, category, identity):
    return sha(f'{seed}|{dataset}|{category}|{identity}'.encode())


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def load_source(name, info, directory):
    from huggingface_hub import hf_hub_download
    local = directory/'raw'/(name+('.parquet' if info.get('file','').endswith('.parquet') else '.jsonl' if info.get('file','').endswith('.jsonl') or 'url' in info else '.json'))
    if local.exists():
        path = local
    elif 'files' in info:
        raise ValueError('Pinned multi-file projection must be relayed with its source receipt')
    elif 'url' in info:
        path = directory/'question.jsonl'
        if not path.exists():
            urllib.request.urlretrieve(info['url'], path)
    else:
        path = Path(hf_hub_download(info['repo'], info['file'], repo_type='dataset',
                                   revision=info['revision'], cache_dir=directory/'hf'))
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    elif path.suffix == '.jsonl':
        rows = list(read_jsonl(path))
    else:
        rows = json.loads(path.read_text())
    if info.get('sha256') and file_sha(path)!=info['sha256']:
        raise ValueError(f'{name}: immutable source hash mismatch')
    if info.get('bytes') and path.stat().st_size!=info['bytes']:
        raise ValueError(f'{name}: source file is truncated')
    if not isinstance(rows, list):
        raise ValueError(f'{name}: source is not a row list')
    return rows, sha(path.read_bytes())


def prompt(name, row):
    if name == 'gsm8k':
        return row['question']+'\nPlease reason step by step and give the final answer.'
    if name == 'alpaca':
        return row['instruction']+ ('\n\n'+row['input'] if row.get('input','').strip() else '')
    if name == 'mbpp':
        return 'Write a Python function for the following task.\n'+row['text']+'\nExamples:\n'+'\n'.join(row['test_list'][:3])
    if name == 'math500':
        return row['problem']+'\nPlease reason step by step and put the final answer in \\boxed{}.'
    if name == 'livecodebench':
        text = row['question_content']
        if row.get('starter_code'):
            text += '\n\nStarter code:\n'+row['starter_code']
        return text+'\nWrite a Python 3 solution.'
    if name == 'mtbench':
        return row['turns'][0]
    raise ValueError(name)


def long_prompt(row, context):
    choices = '\n'.join(f'{label}. {row["choice_"+label]}' for label in 'ABCD')
    return ('Read the following text and answer the question. Explain your reasoning, '
            'then end with the letter of the correct answer.\n\n'+context+
            '\n\nQuestion: '+row['question']+'\n'+choices)


def long_input(row, tokenizer, encode, user):
    """Tokenize bounded head/tail windows, not 2M-token context seventeen times."""
    context=row['context'];overhead=len(encode(user(long_prompt(row,''))))
    budget=32752-overhead
    if budget<=0:raise ValueError('Question/options alone exceed the 32k track')
    window=65536
    while True:
        complete=len(context)<=2*window
        if complete:
            all_ids=tokenizer.encode(context,add_special_tokens=False)
            original_ids=encode(user(long_prompt(row,context)))
            if len(original_ids)<32704:return None
            if len(original_ids)<=32768:
                return long_prompt(row,context),original_ids,len(original_ids),len(original_ids)
            head,tail=all_ids,all_ids
            break
        head=tokenizer.encode(context[:window],add_special_tokens=False)
        tail=tokenizer.encode(context[-window:],add_special_tokens=False)
        if min(len(head),len(tail))>= (budget+1)//2+64:break
        window*=2
    for _ in range(6):
        left=(budget+1)//2;right=budget//2
        chosen=head[:left]+(tail[-right:] if right else [])
        text=long_prompt(row,tokenizer.decode(chosen))
        ids=encode(user(text))
        if 32704<=len(ids)<=32768:
            original=len(original_ids) if complete else None
            # Non-overlapping tokenized windows establish eligibility, while
            # avoiding a false exact original token count for huge documents.
            lower=original if complete else overhead+len(head)+len(tail)-64
            return text,ids,original,lower
        budget+=32752-len(ids)
    raise ValueError('Failed context-only 32k truncation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    if (root/'manifest_metadata.json').exists():
        raise SystemExit('A frozen manifest already exists; refusing to replace it')
    sources = json.loads(Path(__file__).with_name('sources.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    tokenizer.model_max_length=10**12  # Preparation only; final model input is capped below.
    def encode(messages):
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False)
    def user(text):
        return [{'role':'user','content':text}]
    canonical_path=root/'canonical_mtbench_histories.json'
    selected, selection, metadata = [], [], {}
    canonical=json.loads(canonical_path.read_text()) if canonical_path.exists() else {}
    target = None
    for name, info in sources.items():
        print('SOURCE',name,flush=True)
        rows, raw_hash = load_source(name, info, root)
        # Inspect five records before adapter use; retain only schemas and a short
        # excerpt in this audit, never execute data or include answers in prompts.
        dump(root/f'{name}_inspection.json', [
            {'keys':sorted(row), 'excerpt':str(row)[:800]} for row in rows[:5]])
        buckets = {}
        for i, row in enumerate(rows):
            identity = str(next((row[k] for k in ['_id','question_id','task_id','problem_id'] if k in row),i))
            if name=='livecodebench':identity=str(row['platform'])+'/'+identity
            category = str(row['domain']) if name == 'longbenchv2' else 'all'
            buckets.setdefault(category,[]).append((identity,row))
        counts = {}
        for category, items in sorted(buckets.items()):
            # Stable hashing gives a reproducible random sample without relying
            # on source order or a random.shuffle implementation.
            ordered = sorted(items,key=lambda pair:rank(42,name,category,pair[0]))
            test_limit, select_limit = (50,10) if name=='longbenchv2' else (128,32)
            def make(identity,row):
                if name == 'longbenchv2':
                    prepared=long_input(row,tokenizer,encode,user)
                    if prepared is None:return None
                    text,ids,original_length,original_lower_bound=prepared
                else:
                    text=prompt(name,row); ids=encode(user(text)); original=ids
                    ids=ids[-8192:]
                    original_length=len(original);original_lower_bound=original_length
                record={'id':f'{name}/{identity}/turn1', 'dataset':name,'category':category,
                        'source_id':identity,'turn':1,'prompt':text,'input_ids':ids,
                        'original_input_tokens':original_length,'input_tokens':len(ids),
                        'original_input_tokens_lower_bound':original_lower_bound,
                        'truncated':original_length is None or len(ids)<original_length}
                if name=='longbenchv2':record['original_context_characters']=len(row['context'])
                if name=='mtbench':
                    record['second_question']=row['turns'][1]
                return record
            eligible={}
            if name=='longbenchv2':
                for identity,row in items:
                    record=make(identity,row)
                    if record is not None:eligible[identity]=record
                ordered=[pair for pair in ordered if pair[0] in eligible]
            # Long context tokenization is deferred until a sampled candidate
            # is considered. Exhaust the stream only if the category is small.
            tests=[]; excluded=set(); considered=0
            for identity,row in ordered:
                considered+=1
                r=eligible[identity] if name=='longbenchv2' else make(identity,row)
                if r is not None:
                    tests.append(r);excluded.add(identity)
                if len(tests)>=test_limit:
                    break
            choices=sorted((p for p in items if p[0] not in excluded and
                (name!='longbenchv2' or p[0] in eligible)),key=lambda pair:rank(43,name,category,pair[0]))
            choices_selected=[]
            for identity,row in choices:
                r=eligible[identity] if name=='longbenchv2' else make(identity,row)
                if r is not None:choices_selected.append(r)
                if len(choices_selected)>=select_limit:break
            selected.extend(tests); selection.extend(choices_selected)
            counts[category]={'source':len(items),'test':len(tests),'selection':len(choices_selected),
                              'candidates_considered_for_test':considered,'test_cap':test_limit}
            if name=='longbenchv2':counts[category]['eligible_32k']=len(eligible)
        metadata[name]={'source':info,'raw_sha256':raw_hash,'counts':counts}
        if name=='livecodebench' and 'files' in info:
            receipt=root/'raw/livecodebench_source_receipt.json'
            metadata[name]['source_parts']=json.loads(receipt.read_text())
            assert metadata[name]['source_parts']['sha256']==raw_hash
        if name=='mtbench':
            needed=[r for panel in [selected,selection] for r in panel
                    if r['dataset']=='mtbench' and r['id'] not in canonical]
            if needed:
                target=AutoModelForCausalLM.from_pretrained(args.target,local_files_only=True,
                      torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(args.device).eval()
            for panel in [selected,selection]:
                turns=[]
                for record in [r for r in panel if r['dataset']=='mtbench']:
                    if record['id'] not in canonical:
                        ids=torch.tensor([record['input_ids']],device=args.device)
                        with torch.inference_mode():
                            result=target.generate(ids,do_sample=False,max_new_tokens=512,
                                     pad_token_id=tokenizer.eos_token_id)
                        answer_ids=result[0,ids.shape[1]:].tolist()
                        canonical[record['id']]={'ids':answer_ids,
                            'text':tokenizer.decode(answer_ids,skip_special_tokens=True),
                            'sha256':sha(json.dumps(answer_ids).encode())}
                    answer=canonical[record['id']]['text']
                    messages=[{'role':'user','content':record['prompt']},
                              {'role':'assistant','content':answer},
                              {'role':'user','content':record.pop('second_question')}]
                    input_ids=encode(messages)
                    turns.append({'id':record['id'].replace('/turn1','/turn2'),'dataset':name,
                        'category':record['category'],'source_id':record['source_id'],'turn':2,
                        'messages':messages,'input_ids':input_ids[-8192:],
                        'original_input_tokens':len(input_ids),'input_tokens':min(len(input_ids),8192),
                        'truncated':len(input_ids)>8192,'canonical_history_sha256':canonical[record['id']]['sha256']})
                panel.extend(turns)
            if target is not None:del target;target=None;torch.cuda.empty_cache()
        print('PREPARED',name,counts,flush=True)
    dump(root/'canonical_mtbench_histories.json',canonical)
    assert len({r['id'] for r in selected})==len(selected), 'Duplicate test request IDs'
    assert len({r['id'] for r in selection})==len(selection), 'Duplicate selection request IDs'
    test_sources={(r['dataset'],r['source_id']) for r in selected}
    selection_sources={(r['dataset'],r['source_id']) for r in selection}
    assert not test_sources.intersection(selection_sources), 'Test/selection source leakage'
    manifest_hashes={}
    for filename,panel in [('manifest.jsonl',selected),('selection_manifest.jsonl',selection)]:
        raw=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in sorted(panel,key=lambda r:r['id']))
        (root/filename).write_text(raw);manifest_hashes[filename]=sha(raw.encode())
    dump(root/'manifest_metadata.json',{'protocol':'component-acceptance-v1','seed':42,'selection_seed':43,
        'target':args.target,'sources':metadata,'hashes':manifest_hashes,
        'tokenizer_hashes':{p.name:sha(p.read_bytes()) for p in Path(args.target).glob('*')
                            if p.name.startswith(('tokenizer','special_tokens','chat_template'))},
        'target_hashes':{p.name:file_sha(p) for p in Path(args.target).glob('*')
            if p.suffix=='.safetensors' or p.name in ['config.json','generation_config.json']},
        'canonical_histories_sha256':sha((root/'canonical_mtbench_histories.json').read_bytes())})
    print('MANIFESTS_FROZEN',manifest_hashes,flush=True)


if __name__=='__main__':main()
