"""Create fixed MT-Bench second-turn histories in batches before full training."""
import argparse,json,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--data',required=True)
args=p.parse_args();root=Path(args.data);out=root/'canonical_mtbench_histories.json'
if out.exists():raise SystemExit('Canonical histories already exist')
rows=[json.loads(l) for l in (root/'raw/mtbench.jsonl').read_text().splitlines()]
tok=AutoTokenizer.from_pretrained(args.target,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(args.target,torch_dtype=torch.bfloat16,
           attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
result={};eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
for start in range(0,len(rows),8):
    batch=rows[start:start+8]
    sequences=[tok.apply_chat_template([{'role':'user','content':r['turns'][0]}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False)[-8192:] for r in batch]
    length=max(map(len,sequences));pad=tok.eos_token_id
    ids=torch.tensor([[pad]*(length-len(s))+s for s in sequences],device='cuda:0')
    mask=torch.tensor([[0]*(length-len(s))+[1]*len(s) for s in sequences],device='cuda:0')
    with torch.inference_mode():
        generated=model.generate(ids,attention_mask=mask,max_new_tokens=512,
                                  do_sample=False,pad_token_id=pad)
    for row,tokens in zip(batch,generated[:,length:].tolist()):
        stop=next((i+1 for i,t in enumerate(tokens) if t in eos),len(tokens));tokens=tokens[:stop]
        key=f'mtbench/{row["question_id"]}/turn1'
        result[key]={'ids':tokens,'text':tok.decode(tokens,skip_special_tokens=True),
                     'sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
    print('CANONICAL',len(result),'/',len(rows),flush=True)
temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');temp.replace(out)
print('CANONICAL_COMPLETE',len(result),flush=True)
