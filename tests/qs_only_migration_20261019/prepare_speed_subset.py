"""Deterministic raw 4000-row input panel for both frameworks' speed probes."""
from pathlib import Path
import hashlib,heapq,json
source=Path('/data/temp/datasets/open_perfectblend_80k_qwen3_4b.jsonl')
base=Path('/data/temp/LiteSpec_exp/tests/system_benchmark_qs_20261019')
base.mkdir(exist_ok=True)
digest=hashlib.sha256();count=0
def records():
 global count
 with source.open('rb') as f:
  for i,raw in enumerate(f):
   digest.update(raw)
   if not raw.strip():continue
   json.loads(raw);count+=1
   key=hashlib.sha256(('42:'+str(i)).encode()).digest()
   yield key,i,raw
rows=heapq.nsmallest(4000,records())
assert len(rows)==4000 and digest.hexdigest()=='b8fb09cec39962888b08d457836bcb2bfff0bd988df51ce4910a065e24edc168'
out=base/'speed_4000_seed42.jsonl';entries=[]
with out.open('wb') as f:
 for _,index,raw in rows:
  assert raw.endswith(b'\n'),'Preserve source LF record boundary'
  f.write(raw);entries.append({'source_line_index':index,'raw_sha256':hashlib.sha256(raw).hexdigest()})
panel={'seed':42,'count':4000,'source_rows':count,'source':str(source),'source_sha256':digest.hexdigest(),'sample_path':str(out),'sample_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'format':'Original raw JSONL records, unchanged; LF boundaries','purpose':'Speed only for both frameworks/all modes. Native capture must verify exactly 4000 usable records; incomplete capture is not a completed probe.','rows':entries}
(base/'speed_4000_manifest.json').write_text(json.dumps(panel,indent=2)+'\n')
print(json.dumps({k:v for k,v in panel.items() if k!='rows'}))
