"""Paired DSpark speculative-decoding benchmark on one GPU."""
import gc
import inspect
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/data/wanghanzhen/projects/SpecDecoding/SpecForge')
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'results.jsonl').exists():
    raise RuntimeError('Refusing to append to an existing benchmark run')
sys.path.insert(0, str(ROOT))
from specforge.modeling.auto import AutoDraftModel
import specforge.modeling.draft.dflash as dflash
from specforge.modeling.draft.dspark import DSparkDraftModel, _sample

if os.environ.get('FIX_DSPARK_ALIGNMENT'):
    def corrected_sample_draft_tokens(self, target, draft_hidden, block_output_ids):
        # Training slot 0 predicts the token after the anchor. Serving needs
        # block_size-1 proposals, so retain slots [0, block_size-2].
        proposal_hidden = draft_hidden[:, -self.block_size:, :][:, :-1, :]
        base_logits = target.lm_head(proposal_hidden)
        if self.markov_head is None:
            return _sample(base_logits)
        sampled_tokens, _ = self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=block_output_ids[:, 0],
            hidden_states=proposal_hidden,
        )
        return sampled_tokens

    DSparkDraftModel._sample_draft_tokens = corrected_sample_draft_tokens

torch.set_num_threads(4)
torch.manual_seed(42)

# Copy the production method and add observations without changing its decisions.
source = textwrap.dedent(inspect.getsource(dflash.DFlashDraftModel.spec_generate))
source = source.replace('def spec_generate(', 'def measured_generate(', 1)
def add(old, new):
    global source
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new, 1)

add('    acceptance_lengths = []\n', '    acceptance_lengths = []\n    trace = []\n')
add('        block_output_ids = output_ids', '''        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        step_start = start
        block_output_ids = output_ids''')
add('        past_key_values_draft.crop(start)\n', '        events[1].record()\n        past_key_values_draft.crop(start)\n')
add('        block_output_ids[:, 1:] = draft_tokens\n', '        events[2].record()\n        block_output_ids[:, 1:] = draft_tokens\n')
add('        if temperature > 0 and draft_probs is not None:', '        events[3].record()\n        if temperature > 0 and draft_probs is not None:')
add('        output_ids[:, start : start + acceptance_length + 1] =', '''        trace.append({
                "start": step_start - num_input_tokens,
                "accepted": int(acceptance_length),
                "matches_tensor": (block_output_ids[:, 1:] == posterior[:, :-1]).detach(),
                "events": events,
            })
        output_ids[:, start : start + acceptance_length + 1] =''')
add('    return output_ids\n', '''    measured = []
    for row in trace:
        events = row.pop("events")
        row["matches"] = row.pop("matches_tensor")[0].cpu().tolist()
        row["backbone_ms"] = events[0].elapsed_time(events[1])
        row["head_ms"] = events[1].elapsed_time(events[2])
        row["draft_ms"] = events[0].elapsed_time(events[2])
        row["target_ms"] = events[2].elapsed_time(events[3])
        row["eligible"] = row["start"] + block_size <= output_ids.shape[1] - num_input_tokens
        measured.append(row)
    self._benchmark_trace = measured
    return output_ids
''')
ns = dict(dflash.__dict__)
exec(compile(source, '<measured_dspark_generate>', 'exec'), ns)
measured_generate = ns['measured_generate']
(OUT / 'instrumented_method.py').write_text(source)

samples_path = Path(os.environ['SAMPLES_PATH'])
samples = json.loads(samples_path.read_text())
(OUT / 'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
target_path = '/data/wanghanzhen/models/Qwen3-8B'
tokenizer = AutoTokenizer.from_pretrained(target_path)
attention = 'sdpa'
target = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16,
                                               attn_implementation=attention).cuda().eval()

def encode(prompt):
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.cuda()

eos = target.generation_config.eos_token_id
stops = list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]
def call_args(ids, n=256):
    return dict(target=target, input_ids=ids, max_new_tokens=n,
                stop_token_ids=stops, temperature=0.0)

order = ['noconv', 'fullconv']
if os.environ.get('REVERSE'):
    order.reverse()
results = []
configs = {}
for label in order:
    checkpoint = Path('/data/wanghanzhen/outputs/dspark_conv_eval_20260924') / label
    configs[label] = json.loads((checkpoint / 'config.json').read_text())
    draft = AutoDraftModel.from_pretrained(str(checkpoint), torch_dtype=torch.bfloat16,
                                           attn_implementation=attention).cuda().eval()
    assert draft.block_size == 7
    warm = encode('Explain why the sky is blue in one paragraph.')
    reference = draft.spec_generate(**call_args(warm, 64))
    observed = measured_generate(draft, **call_args(warm, 64))
    assert torch.equal(reference, observed), 'Instrumented path changed output'
    for sample in samples:
        ids = encode(sample['prompt'])
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        out = measured_generate(draft, **call_args(ids))
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start_time) * 1000
        output_ids = out[0, ids.shape[1]:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        row = {**sample, 'model': label, 'input_tokens': ids.shape[1],
               'output_tokens': len(output_ids), 'output_ids': output_ids,
               'output_text': output_text,
               'decode_wall_ms_with_prefill': wall_ms,
               'trace': draft._benchmark_trace}
        results.append(row)
        with (OUT / 'results.jsonl').open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({'model': label, 'dataset': sample['dataset'],
                          'index': sample['index'], 'steps': len(row['trace']),
                          'tokens': row['output_tokens']}), flush=True)
    del draft
    gc.collect()
    torch.cuda.empty_cache()

def last_number(text):
    boxed = re.findall(r'\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}', text)
    values = boxed or re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', text)
    return values[-1].replace(',', '') if values else None

summary = []
datasets = sorted({sample['dataset'] for sample in samples}) + ['all']
for model in order:
    for dataset in datasets:
        group = [r for r in results if r['model'] == model and
                 (dataset == 'all' or r['dataset'] == dataset)]
        steps = [t for r in group for t in r['trace']]
        eligible = [t for t in steps if t['eligible']]
        # Sum over actual contiguous accepted positions, not isolated matches.
        success = []
        reached = []
        single = []
        for i in range(6):
            reached.append(len(eligible) if i == 0 else success[-1])
            success.append(sum(all(t['matches'][:i+1]) for t in eligible))
            single.append(sum(t['matches'][i] for t in eligible))
        output_tokens = sum(r['output_tokens'] for r in group)
        row = {'model': model, 'dataset': dataset, 'questions': len(group),
               'steps': len(steps), 'eligible_steps': len(eligible),
               'accept_length_including_anchor': 1 + sum(t['accepted'] for t in eligible)/len(eligible),
               'isolated_match_rate': [x/len(eligible) for x in single],
               'cumulative_accept_rate': [x/len(eligible) for x in success],
               'conditional_accept_rate': [x/y if y else None for x,y in zip(success,reached)],
               'conditional_success': success, 'conditional_reached': reached,
               'output_tokens': output_tokens}
        labeled = [r for r in group if r.get('answer') is not None]
        if labeled:
            row['answer_accuracy'] = sum(
                last_number(r['output_text']) == str(r['answer']).replace(',', '')
                for r in labeled
            ) / len(labeled)
        for field in ['backbone_ms', 'head_ms', 'draft_ms', 'target_ms']:
            total = sum(t[field] for t in steps)
            row[field+'_per_step'] = total / len(steps)
            row[field+'_per_output_token'] = total / output_tokens
        row['total_gpu_ms_per_output_token'] = row['draft_ms_per_output_token'] + row['target_ms_per_output_token']
        summary.append(row)
checks = []
for sample in samples:
    a,b = [r for r in results if (r['dataset'],r['index']) ==
           (sample['dataset'], sample['index'])]
    checks.append({'dataset':sample['dataset'],'index':sample['index'],
                   'identical_output':a['output_ids']==b['output_ids']})
(OUT / 'summary.json').write_text(json.dumps({
    'summary':summary, 'output_checks':checks, 'configs':configs,
    'metadata':{'device':torch.cuda.get_device_name(), 'torch':torch.__version__,
                'attention':attention, 'temperature':0, 'max_new_tokens':256,
                'batch_size':1, 'block_size':7, 'sample_seed':42,
                'timing':'CUDA events on production DSpark decode; prefill excluded from GPU interval totals.'}
}, ensure_ascii=False, indent=2))
print('COMPLETE', flush=True)
