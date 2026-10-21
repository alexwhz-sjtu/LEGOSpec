"""QS-only launch gate. Existing full benchmark and imported CE final precede GPU use."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,time,traceback
BASE=Path('/data/temp/LiteSpec_exp/tests/staged_component_search_qs_20261019')
BENCH=BASE.parent/'component_benchmark_20261018'
STATE=BASE/'gate_state.json'
def read(p):return json.loads(Path(p).read_text())
def write(x):
 x['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=STATE.with_suffix('.tmp');p.write_text(json.dumps(x,indent=2)+'\n');p.replace(STATE)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def guard():
 for p,h in read(BASE/'continuation_hashes.json').items():assert sha(p)==h,'Protected source changed: '+p
def predecessors_done():
 q=read(BENCH/'benchmark_queue.json');assert q['status']!='failed','Existing benchmark failed: '+str(q.get('error'))
 if q['status']!='completed':return False
 assert len(q['jobs'])==14
 for p,h in q['prepared_hashes'].items():assert sha(p)==h
 for j in q['jobs']:
  assert j['status']=='completed' and j['returncodes']==[0]*8
  root=Path(j['output']);s=read(root/'summary.json')
  assert s['status']=='complete' and not s['smoke'] and not s['missing'] and not s['unexpected']
  assert s['identity']['manifest_sha256']=='bdfd20787753294352983d7712951c1b03dd5cc148a0f2ef746b235bd7ce1c79'
  for i in range(8):
   assert read(root/f'metadata_{i}.json')==s['identity']
   c=read(root/f'complete_{i}.json');assert c['status']=='completed' and c['requests']==132
 return True
def import_done():
 p=BASE/'imported_rnn_ce_verified.json'
 if not p.exists():return False
 r=read(p);assert r['verified'] and r['returncode']==0 and r['optimizer_step']==59724 and r['wandb_state']=='finished'
 assert r['source_pid']==2948515 and str(r['source_start_ticks'])=='807593000'
 for f in r['files']:assert Path(f['path']).stat().st_size==f['size'] and sha(f['path'])==f['sha256']
 return True
def start(script,log):
 python='/data/temp/LiteSpec_exp/.venv/bin/python';setup='/data/temp/LiteSpec_exp/tests/h800_setup_20261016/env.sh'
 with (BASE/log).open('ab',buffering=0) as output:
  return subprocess.Popen(['bash','-c','source "$1"; exec "$2" "$3"','qs-gate',setup,python,str(BASE/script)],cwd='/data/temp/LiteSpec_exp',stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
def main():
 with (BASE/'gate.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(STATE) if STATE.exists() else {'status':'prepared'}
  if state['status'] in ['dispatched','failed']:return
  state.update(status='waiting_benchmark_and_import',pid=os.getpid());write(state)
  try:
   guard();q=read(BASE/'queue.json');assert q['status']=='prepared' and all(j['status']=='pending' for j in q['jobs'])
   while True:
    ready_bench=predecessors_done();ready_import=import_done()
    state.update(benchmark_ready=ready_bench,import_ready=ready_import);write(state)
    if ready_bench and ready_import:break
    time.sleep(45)
   guard();train=start('supervisor.py','supervisor.log')
   for _ in range(200):
    q=read(BASE/'queue.json')
    if q.get('supervisor_pid')==train.pid:break
    assert train.poll() is None,'Supervisor exited before state attachment'
    time.sleep(.1)
   else:raise RuntimeError('Supervisor PID not recorded')
   follow=start('continuation.py','continuation.log')
   state.update(status='dispatched',supervisor_pid=train.pid,controller_pid=follow.pid);write(state)
  except Exception as e:
   state.update(status='failed',error=type(e).__name__+': '+str(e));write(state);traceback.print_exc();raise
if __name__=='__main__':main()
