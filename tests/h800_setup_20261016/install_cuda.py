import concurrent.futures,hashlib,json,shutil,tarfile,urllib.request
from pathlib import Path
base='https://developer.download.nvidia.com/compute/cuda/redist/'
with urllib.request.urlopen(base+'redistrib_12.8.1.json',timeout=60) as r:meta=json.load(r)
cache=Path('/data/temp/.cache/cuda-redist');cache.mkdir(parents=True,exist_ok=True)
root=Path('/data/temp/.tools/cuda-12.8');root.mkdir(parents=True,exist_ok=True)
def install(name):
 item=meta[name]['linux-x86_64'];path=cache/Path(item['relative_path']).name
 if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
  print('DOWNLOAD',name,flush=True)
  with urllib.request.urlopen(base+item['relative_path'],timeout=180) as r,path.open('wb') as f:shutil.copyfileobj(r,f,8*1024*1024)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 extracted=cache/name;extracted.mkdir(exist_ok=True)
 with tarfile.open(path) as t:t.extractall(extracted,filter='data')
 source=next(extracted.iterdir())
 shutil.copytree(source,root,dirs_exist_ok=True)
 print('INSTALLED',name,meta[name]['version'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(install,['cuda_nvcc','cuda_cudart','cuda_cccl','libcurand']))
if not(root/'lib64').exists():(root/'lib64').symlink_to('lib')
print('CUDA_TOOLKIT_READY',root,flush=True)
