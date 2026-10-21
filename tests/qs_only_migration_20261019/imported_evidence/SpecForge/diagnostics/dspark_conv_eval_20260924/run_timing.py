"""Interleaved, warmed end-to-end timing for matched DSpark drafts."""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path('/data/wanghanzhen/projects/SpecDecoding/SpecForge')
sys.path.insert(0, str(ROOT))
from specforge.modeling.auto import AutoDraftModel

torch.set_num_threads(4)
torch.manual_seed(42)
samples = json.loads((ROOT / 'diagnostics/dspark_conv_eval_20260924/samples.json').read_text())
target_path = '/data/wanghanzhen/models/Qwen3-8B'
tokenizer = AutoTokenizer.from_pretrained(target_path)
target = AutoModelForCausalLM.from_pretrained(
    target_path, dtype=torch.bfloat16, attn_implementation='sdpa'
).cuda().eval()
base = Path('/data/wanghanzhen/outputs/dspark_conv_eval_20260924')
drafts = {
    name: AutoDraftModel.from_pretrained(str(base / name), dtype=torch.bfloat16,
                                         attn_implementation='sdpa').cuda().eval()
    for name in ('noconv', 'fullconv')
}
eos = target.generation_config.eos_token_id
stops = list(eos) if isinstance(eos, (list, tuple)) else [int(eos)]
ids = []
for sample in samples:
    formatted = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': sample['prompt']}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    ids.append(tokenizer(formatted, return_tensors='pt', add_special_tokens=False).input_ids.cuda())

def generate(name, i):
    return drafts[name].spec_generate(target=target, input_ids=ids[i], max_new_tokens=256,
                                      stop_token_ids=stops, temperature=0.0)

for i in range(5):
    for name in (('noconv', 'fullconv') if i % 2 == 0 else ('fullconv', 'noconv')):
        generate(name, i)
        torch.cuda.synchronize()
print('WARMUP_COMPLETE', flush=True)

results = []
for rep in range(2):
    for i, sample in enumerate(samples):
        pair = {}
        order = ('noconv', 'fullconv') if (rep+i) % 2 == 0 else ('fullconv', 'noconv')
        for name in order:
            torch.cuda.synchronize()
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            event_start.record()
            output = generate(name, i)
            event_end.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            gpu_ms = event_start.elapsed_time(event_end)
            pair[name] = output[0, ids[i].shape[1]:].tolist()
            results.append({'repeat': rep, 'dataset': sample['dataset'], 'index': sample['index'],
                            'model': name, 'output_tokens': len(pair[name]),
                            'wall_ms': wall_ms, 'gpu_ms': gpu_ms})
        assert pair['noconv'] == pair['fullconv'], (rep, sample['dataset'], sample['index'])
        print(json.dumps({'repeat': rep, 'dataset': sample['dataset'],
                          'index': sample['index'], 'done': True}), flush=True)

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({'metadata': {'device': torch.cuda.get_device_name(),
                                          'warmup_questions_per_model': 5,
                                          'repeats': 2, 'interleaved': True,
                                          'attention': 'sdpa', 'temperature': 0},
                            'results': results}, indent=2))
print('COMPLETE', flush=True)
