from pathlib import Path
import hashlib,tomllib,urllib.request,urllib.parse,time
root=Path('/data/temp/LiteSpec_exp')
lock=tomllib.loads((root/'uv.lock').read_text())
p=next(p for p in lock['package'] if p['name']=='torch' and p['version']=='2.9.1+cu128')
w=next(w for w in p['wheels'] if 'cp311-cp311' in w['url'] and 'x86_64' in w['url'] and 'linux' in w['url'])
name=urllib.parse.unquote(w['url'].split('/')[-1])
out=Path('/data/temp/.cache/wheels')/name
url='https://mirrors.nju.edu.cn/pytorch/whl/cu128/'+urllib.parse.quote(name)
h=hashlib.sha256();count=0;last=0;start=time.monotonic()
with urllib.request.urlopen(url,timeout=120) as r,out.open('wb') as f:
 while b:=r.read(8*1024*1024):
  f.write(b);h.update(b);count+=len(b)
  if count-last>128*1024*1024:
   print('PROGRESS',count,round(time.monotonic()-start,1),flush=True);last=count
assert h.hexdigest()==w['hash'].split(':')[1],h.hexdigest()
print('TORCH_WHEEL_VERIFIED',out,count,flush=True)
