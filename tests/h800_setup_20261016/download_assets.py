import concurrent.futures, hashlib, json, os, shutil, time, urllib.request, urllib.parse
from pathlib import Path
ROOT=Path('/mnt/bd-1/task/339953/artifacts/LiteSpec')
LOCAL=Path('/data/temp')
REPORT=Path('/data/temp/LiteSpec_exp/tests/h800_setup_20261016/download_manifest.json')
records=[]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def files(repo):
 u=f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true'
 with urllib.request.urlopen(u,timeout=60) as r:d=json.load(r)
 assert d['Code']==200,d
 return [f for f in d['Data']['Files'] if f['Type']=='blob']
def download(repo,f,directory):
 target=directory/f['Path'];target.parent.mkdir(parents=True,exist_ok=True)
 size=f['Size'];sha=f['Sha256']
 if not(target.exists() and target.stat().st_size==size and digest(target)==sha):
  part=target.with_name(target.name+'.partial')
  for attempt in range(6):
   try:
    offset=part.stat().st_size if part.exists() else 0
    if offset>=size:part.unlink();offset=0
    url=f'https://modelscope.cn/models/{repo}/resolve/master/'+urllib.parse.quote(f['Path'],safe='/')
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
    print('DOWNLOAD',repo,f['Path'],'resume',offset,'size',size,flush=True)
    with urllib.request.urlopen(req,timeout=120) as r:
     append=offset>0 and r.status==206
     with part.open('ab' if append else 'wb') as out:
      while chunk:=r.read(8*1024*1024):out.write(chunk)
    assert part.stat().st_size==size,(f['Path'],part.stat().st_size,size)
    assert digest(part)==sha,('SHA256 mismatch',f['Path'])
    part.replace(target);break
   except Exception as e:
    print('RETRY',f['Path'],attempt,str(e),flush=True)
    if attempt==5:raise
    time.sleep(min(2**attempt,16))
 local=(LOCAL/'datasets'/f['Path']) if repo=='alexwangsjtu/transfer' else (LOCAL/'models'/'Qwen3-4B'/f['Path'])
 local.parent.mkdir(parents=True,exist_ok=True)
 if not(local.exists() and local.stat().st_size==size and digest(local)==sha):
  pending=local.with_name(local.name+'.partial')
  shutil.copyfile(target,pending)
  assert pending.stat().st_size==size and digest(pending)==sha
  pending.replace(local)
 result={'repo':repo,'file':f['Path'],'size':size,'sha256':sha,'cloud':str(target),'local':str(local),'verified':True}
 print('VERIFIED',f['Path'],size,flush=True)
 return result
dataset='open_perfectblend_80k_qwen3_4b.jsonl'
f=next(f for f in files('alexwangsjtu/transfer') if f['Path']==dataset)
records.append(download('alexwangsjtu/transfer',f,ROOT/'datasets'))
count=0;samples=[]
with (LOCAL/'datasets'/dataset).open() as stream:
 for line in stream:
  if not line.strip():continue
  row=json.loads(line);count+=1
  if len(samples)<5:
   sample={'row':count,'keys':list(row)}
   for key,val in row.items():
    if isinstance(val,list):sample[key]={'length':len(val),'first_item_keys':list(val[0]) if val and isinstance(val[0],dict) else None,'roles':[x.get('role',x.get('from')) for x in val[:8] if isinstance(x,dict)]}
    elif isinstance(val,str):sample[key]={'chars':len(val)}
    else:sample[key]={'type':type(val).__name__}
   samples.append(sample)
print('DATASET_ROWS',count,'FIRST_FIVE_SCHEMA',json.dumps(samples),flush=True)
REPORT.write_text(json.dumps({'dataset_rows':count,'first_five_schema':samples,'files':records},indent=2))
model_files=files('Qwen/Qwen3-4B')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 records.extend(pool.map(lambda f:download('Qwen/Qwen3-4B',f,ROOT/'models'/'Qwen3-4B'),model_files))
model=LOCAL/'models'/'Qwen3-4B'
index=json.loads((model/'model.safetensors.index.json').read_text())
assert all((model/name).is_file() for name in set(index['weight_map'].values()))
REPORT.write_text(json.dumps({'dataset_rows':count,'first_five_schema':samples,'files':records,'model_index_verified':True},indent=2))
print('ALL_ASSETS_VERIFIED',len(records),'dataset_rows',count,flush=True)
