"""One-shot dependency queue for the authorized Selector+DFlare experiment."""
from pathlib import Path
import datetime,fcntl,hashlib,json,math,os,re,socket,subprocess,time,traceback

ROOT=Path('/data/temp/LiteSpec_exp')
HOME=ROOT/'tests/queued_selector_dflare_20261017'
STATE=HOME/'queue.json'

def process_matches(pid,start_ticks=None):
    try:
        tail=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return tail[0]!='Z' and (start_ticks is None or tail[19]==str(start_ticks))
    except (FileNotFoundError,ProcessLookupError):
        return False

def validate_completed_metadata(meta):
    cursor=meta['cursor'];phases=meta['phase_plan']
    assert phases and cursor['phase_index']==len(phases),'Source training has not completed all phases'
    assert all(p['epochs']==6 for p in phases),'Unexpected source epoch schedule'
    assert meta['optimizer_step']==meta['scheduler_total_steps']>0,'Source stopped before scheduled optimizer steps'
    assert cursor['optimizer_step']==meta['optimizer_step']
    assert cursor['phase_step']==0 and cursor['next_batch']==0
    assert meta['distributed_training_state']['world_size']==8
    return {'optimizer_step':meta['optimizer_step'],'phase_index':cursor['phase_index']}

def validate_final(directory):
    directory=Path(directory)
    for name in ['config.json','model.safetensors','training_state.pt','training_config.yaml']+[f'training_state_rank_{i:05d}.pt' for i in range(8)]:
        p=directory/name
        assert p.is_file() and p.stat().st_size>0,'Missing final checkpoint file: '+str(p)
    import torch
    meta=torch.load(directory/'training_state.pt',map_location='cpu',weights_only=True)
    return validate_completed_metadata(meta)

def verify_configuration(state):
    for filename,expected in state['prepared_hashes'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,'Prepared configuration changed: '+filename
    old=json.loads((HOME/'source_config.json').read_text())
    new=json.loads(Path(state['draft_config']).read_text())
    assert new['composable_config']['context']['type']=='dflare'
    new['composable_config']['context']['type']='dflash'
    assert old==new,'Model recipe changed beyond the authorized fusion change'

def metrics_from_log(path):
    if not Path(path).exists():return []
    text=Path(path).read_text(errors='replace')
    entries=re.findall(r'composable optimizer_step=(\d+) micro_step=(\d+) loss=([^\s]+)',text)
    return [{'step':int(s),'loss':float(loss)} for s,_,loss in entries]

def update(state,status,**fields):
    changed=state.get('status')!=status
    state.update(status=status,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    body=json.dumps(state,indent=2)
    temporary=STATE.with_suffix('.tmp');temporary.write_text(body);temporary.replace(STATE)
    (Path(state['next_run_dir'])/'queue_status.json').write_text(body)
    if changed or fields:print(json.dumps({'status':status,**fields}),flush=True)

def resources_free():
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    if p.stdout.strip():return False
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',29501))
    except OSError:return False
    return True

def main():
    HOME.mkdir(exist_ok=True)
    with (HOME/'supervisor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Queue supervisor already active',flush=True);return
        state=json.loads(STATE.read_text())
        update(state,'waiting_source',supervisor_pid=os.getpid())
        try:
            if not state.get('next_pid'):
                while process_matches(state['source_pid'],state['source_start_ticks']):
                    time.sleep(30)
                update(state,'verifying_source')
                completion=validate_final(state['source_final_dir'])
                update(state,'waiting_resources',source_completion=completion)
                while not resources_free():time.sleep(30)
                verify_configuration(state)
                assert not (Path(state['next_run_dir'])/'checkpoints').exists(),'Next checkpoint directory already exists; let the trainer reserve it'
                env=os.environ.copy()
                env.update(WANDB_MODE='online',WANDB_ENTITY='alex-wang_sjtu-sjtu',WANDB_PROJECT='spec-decoding',WANDB_RUN_ID=state['next_wandb_id'],WANDB_RESUME='never',WANDB_DIR=state['next_run_dir'],LITESPEC_RUN_DIR=state['next_run_dir'],PYTHONUNBUFFERED='1',PYTHON_BIN=str(ROOT/'.venv/bin/python'))
                log=Path(state['next_run_dir'])/'train.log'
                with log.open('ab',buffering=0) as f:
                    p=subprocess.Popen(['bash','scripts/composable/run_training_composable.sh',state['config']],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
                update(state,'training_started',next_pid=p.pid,next_start_ticks=ticks,log=str(log),wandb_url=f"https://wandb.ai/alex-wang_sjtu-sjtu/spec-decoding/runs/{state['next_wandb_id']}")
            while process_matches(state['next_pid'],state.get('next_start_ticks')):
                metrics=metrics_from_log(state['log'])
                if metrics and not all(math.isfinite(m['loss']) for m in metrics[-3:]):
                    raise RuntimeError('Next training produced a non-finite loss')
                if len(metrics)>=2 and metrics[-1]['step']>metrics[-2]['step']:
                    update(state,'training_verified',verified_metrics=metrics[-3:])
                    return
                time.sleep(30)
            raise RuntimeError('Next training exited before two valid optimizer-step reports')
        except BaseException as e:
            update(state,'failed',error=type(e).__name__+': '+str(e))
            traceback.print_exc();raise

if __name__=='__main__':main()

