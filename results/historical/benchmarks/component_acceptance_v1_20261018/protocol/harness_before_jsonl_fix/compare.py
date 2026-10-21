"""Paired per-request acceptance differences, with explicit pair coverage."""
from pathlib import Path
import argparse,json
from collections import defaultdict
import numpy as np


def load(root):
    return {r['id']:r for p in Path(root).glob('requests_*.jsonl')
            for line in p.read_text().splitlines() if line for r in [json.loads(line)]}


def compare(first,second):
    a,b=load(first),load(second)
    assert set(a)==set(b),'Paired panels must contain exactly the same request IDs'
    groups=defaultdict(list);undefined=defaultdict(int)
    for identity in sorted(a):
        left,right=a[identity],b[identity];key=left['dataset']+'/'+left['category']
        if left['tau'] is None or right['tau'] is None:undefined[key]+=1
        else:groups[key].append(right['tau']-left['tau'])
    result={}
    for key in sorted(set(groups)|set(undefined)):
        values=np.asarray(groups[key]);rng=np.random.default_rng(42)
        boot=[float(rng.choice(values,len(values),replace=True).mean()) for _ in range(2000)] if len(values) else []
        result[key]={'defined_pairs':len(values),'undefined_pairs':undefined[key],
            'delta_request_macro_tau':float(values.mean()) if len(values) else None,
            'paired_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist() if boot else None}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    Path(args.output).write_text(json.dumps(compare(args.baseline,args.candidate),indent=2)+'\n')
