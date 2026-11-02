"""CPU-only periodic original-path W&B offline sync; no training lifecycle control."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--entity',required=True);p.add_argument('--project',required=True);p.add_argument('--interval',type=int,default=1800);p.add_argument('--once',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--receipts',type=Path,default=Path('./cache/wandb_sync'));a=p.parse_args()
 if a.interval<=0:p.error('Interval must be positive')
 if not a.root.is_dir():p.error('Root must be an existing directory containing offline-run-*')
 env={k:v for k,v in os.environ.items() if not k.startswith(('WANDB_SERVICE','_WANDB_SERVICE'))};env.update(CUDA_VISIBLE_DEVICES='',WANDB_MODE='online')
 a.receipts.mkdir(parents=True,exist_ok=True);lock=(a.receipts/'sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  results=[]
  for run in sorted(a.root.glob('offline-run-*')):
   if not run.is_dir():continue
   command=[sys.executable,'-m','wandb','beta','sync','--yes','--no-skip-synced','--entity',a.entity,'--project',a.project,str(run.resolve())]
   if a.dry_run:print(json.dumps(command));continue
   log=a.receipts/(run.name+'.log')
   with log.open('ab') as f:
    begin=f.tell();r=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   with log.open('rb') as f:f.seek(begin);segment=f.read().decode(errors='replace')
   results.append({'run':str(run),'exit_code':r.returncode,'cli_complete':r.returncode==0 and 'Finished syncing' in segment and 'ERROR' not in segment})
  (a.receipts/'latest.json').write_text(json.dumps({'checked_at_unix':time.time(),'results':results,'note':'CLI sync success alone does not prove training complete or cloud finalization'},indent=2)+'\n')
  if a.once or a.dry_run:return
  time.sleep(a.interval)
if __name__=='__main__':main()
