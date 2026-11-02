"""Copy and patch only the known SGLang0.5.9 capture runtime into an isolated path."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,shutil
OLD='5ceb399c675644ea77156e1f1d239b0bcb1aff1ebdef20d5d23b7950ab8b9651'
NEW='c28a82a4e5d806eae62a4e00135d2147f0462fd3f8fcb0290c7423de7c69cf95'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,help='Path to an explicit sglang0.5.9 package directory');p.add_argument('--output',type=Path,required=True,help='New isolated vendor directory; must not exist');a=p.parse_args()
 if a.source:source=a.source.resolve()
 else:
  d=importlib.metadata.distribution('sglang')
  if d.version!='0.5.9':p.error('This patch is specific to installed sglang==0.5.9')
  source=Path(d.locate_file('sglang')).resolve()
 kernel=source/'srt/layers/attention/fla/l2norm.py';vision=source/'srt/layers/attention/vision.py'
 if hashlib.sha256(kernel.read_bytes()).hexdigest()!=OLD:p.error('Unrecognized upstream L2 norm hash; refusing to patch')
 old='if _is_cuda:\n    from sgl_kernel.flash_attn import flash_attn_varlen_func\n'
 if vision.read_text().count(old)!=1:p.error('Unrecognized vision import contract')
 destination=a.output.resolve()
 if destination==source or source in destination.parents:p.error('Output must be outside source package')
 if destination.exists():p.error('Output exists; use a fresh directory')
 replacement=Path(__file__).parent/'sglang059/l2norm_dynamic.py';assert hashlib.sha256(replacement.read_bytes()).hexdigest()==NEW
 destination.mkdir(parents=True);target=destination/'sglang';shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (target/'srt/layers/attention/fla/l2norm.py').write_bytes(replacement.read_bytes())
 text=vision.read_text().replace(old,'''if _is_cuda:
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
    except ImportError:
        def flash_attn_varlen_func(*args, **kwargs):
            raise RuntimeError("Vision FA3 is unavailable; use MM SDPA for text-only Qwen3.5 capture")
''');(target/'srt/layers/attention/vision.py').write_text(text)
 receipt={'status':'patched','source':str(source),'isolated_package':str(target),'old_l2norm_sha256':OLD,'new_l2norm_sha256':NEW,'scope':'JIT specialization and optional vision import only; no numeric, precision, data or CUDA correctness guard changes'};(destination/'runtime_patch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
