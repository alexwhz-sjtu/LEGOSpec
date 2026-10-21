from pathlib import Path
import json,os,subprocess,time
root=Path('/data/temp/LiteSpec_exp')
setup=root/'tests/h800_setup_20261016'
run=json.loads((setup/'run.json').read_text())
run_dir=Path(run['run_dir'])
offline=run_dir/'wandb'/'offline-run-20261016_160455-ivanjb7x'
assert offline.is_dir()
state=setup/'wandb_live_sync.json'
if state.exists():
 old=json.loads(state.read_text())
 p=Path('/proc')/str(old['pid'])/'cmdline'
 if p.exists() and b'wandb' in p.read_bytes():
  print(json.dumps(old,indent=2));raise SystemExit(0)
env=os.environ.copy();env['WANDB_MODE']='online';env['PYTHONUNBUFFERED']='1'
command=[str(root/'.venv/bin/wandb'),'beta','sync','--live','--no-skip-synced','--project','spec-decoding',str(offline)]
log=run_dir/'wandb_live_sync.log'
with log.open('ab',buffering=0) as f:
 process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result={'pid':process.pid,'training_pid':run['pid'],'run_id':'ivanjb7x','project':'spec-decoding','offline_directory':str(offline),'log':str(log),'started_at':time.time()}
state.write_text(json.dumps(result,indent=2))
(run_dir/'wandb_live_sync.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
