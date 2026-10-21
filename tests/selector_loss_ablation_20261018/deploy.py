"""Apply four scoped changes, preserving host-specific additions and backing up originals."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import sys

root=Path(sys.argv[1]).resolve()
base=root/'tests/selector_loss_ablation_20261018'
source=base/'source'
backup=base/'before_deployment'
changes={}
rel='litespec/core/composable/objectives/tv.py'
old=(root/rel).read_text()
line='        token_losses = 0.5 * (target_probabilities - draft_probabilities).abs().sum(-1)'
new='        # Project convention: TV loss is the full distribution L1 distance [0, 2].\n        token_losses = (target_probabilities - draft_probabilities).abs().sum(-1)'
assert old.count(line)==1, 'TV changed; inspect before applying'
changes[rel]=(old,old.replace(line,new))
rel='litespec/core/composable/objectives/__init__.py';old=(root/rel).read_text()
assert 'weighted_sum' not in old
new=old.replace('from .tv import TVObjective','from .tv import TVObjective\nfrom .weighted_sum import WeightedSumObjective')
assert new!=old
new=new.replace('    "bv": BVObjective,','    "bv": BVObjective,\n    "weighted_sum": WeightedSumObjective,')
changes[rel]=(old,new)
rel='litespec/modeling/draft/composable/configuration.py';old=(root/rel).read_text()
assert '"weighted_sum"' not in old and old.count('            "bv",')==1
changes[rel]=(old,old.replace('            "bv",','            "bv",\n            "weighted_sum",'))
rel='litespec/core/composable/objectives/weighted_sum.py';assert not (root/rel).exists()
changes[rel]=(None,(source/rel).read_text())
assert not backup.exists(), 'Deployment backup already exists; inspect before repeating'
backup.mkdir(parents=True)
def sha(text):return hashlib.sha256(text.encode()).hexdigest() if text is not None else None
for rel,(old,new) in changes.items():
    compile(new,rel,'exec')
    if old is not None:
        p=backup/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(old)
# Recheck every file before the first replacement.
for rel,(old,new) in changes.items():
    p=root/rel;assert (p.read_text() if p.exists() else None)==old, 'Concurrent edit: '+rel
for rel,(old,new) in changes.items():
    p=root/rel;temp=p.with_suffix('.loss_ablation_tmp');temp.write_text(new);os.replace(temp,p)
report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
        'files':{rel:{'before':sha(old),'after':sha(new)} for rel,(old,new) in changes.items()}}
(base/'deployment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
