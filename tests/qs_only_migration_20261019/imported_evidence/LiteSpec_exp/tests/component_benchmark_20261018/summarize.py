"""Summarize only compatible complete shards; preserve individual task results."""
from pathlib import Path
import argparse
import json
import math
from collections import defaultdict
import numpy as np


def mean(values):
    return sum(values)/len(values) if values else None


def summarize(rows):
    cycles=[c for r in rows for c in r['cycles']]
    full=[c for c in cycles if not c['censored']]
    request_tau=[r['tau'] for r in rows if r['tau'] is not None]
    bootstrap=[]
    if request_tau:
        rng=np.random.default_rng(42);a=np.asarray(request_tau)
        bootstrap=[float(rng.choice(a,size=len(a),replace=True).mean()) for _ in range(2000)]
    return {'requests':len(rows),'cycles':len(cycles),'complete_cycles':len(full),
        'requests_without_complete_cycles':len(rows)-len(request_tau),
        'cycle_weighted_tau':mean([c['tau'] for c in full]),
        'request_macro_tau':mean(request_tau),
        'request_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist() if bootstrap else None,
        'mean_accepted_draft_tokens':mean([c['accepted_draft_tokens'] for c in full]),
        'committed_per_cycle':sum(c['committed_tokens'] for c in cycles)/len(cycles) if cycles else None,
        'censoring_rate':1-len(full)/len(cycles) if cycles else None,
        'position_conditional_acceptance':[
            sum(c['accepted_draft_tokens']>=j for c in full)/sum(c['accepted_draft_tokens']>=j-1 for c in full)
            if any(c['accepted_draft_tokens']>=j-1 for c in full) else None for j in range(1,8)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--manifest',required=True)
    args=p.parse_args();root=Path(args.results)
    expected={json.loads(l)['id'] for l in Path(args.manifest).read_text().splitlines()}
    rows=[json.loads(l) for f in sorted(root.glob('requests_*.jsonl')) for l in f.read_text().splitlines() if l]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate request IDs')
    found={r['id'] for r in rows};missing=sorted(expected-found);extra=sorted(found-expected)
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['category'])].append(r)
    result={'status':'complete' if not missing and not extra else 'incomplete',
        'missing':missing,'unexpected':extra,
        'groups':{'/'.join(k):summarize(v) for k,v in groups.items()}}
    regular=[v['request_macro_tau'] for k,v in result['groups'].items() if not k.startswith('longbenchv2/')]
    long=[v['request_macro_tau'] for k,v in result['groups'].items() if k.startswith('longbenchv2/')]
    result['regular_task_macro_tau']=mean(regular) if len(regular)==6 and all(v is not None for v in regular) else None
    result['long_domain_macro_tau']=mean(long) if len(long)==6 and all(v is not None for v in long) else None
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
