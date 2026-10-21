"""Wait for authorized training completion; evaluate fixed logical shards on idle GPUs."""
from pathlib import Path
import fcntl,hashlib,json,os,socket,subprocess,time,traceback
from available_gpu_scheduler import available_gpus, run_shards

BASE=Path(__file__).resolve().parent
STATE=BASE/'benchmark_queue.json'


def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)


def hashes(state):
    for name,want in state['prepared_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,'Protected file changed: '+name


def free():
    return bool(available_gpus())


def validate_predecessor(path):
    q=json.loads(Path(path).read_text())
    assert q['status']!='failed','Preceding training failed: '+str(q.get('error'))
    if q['status']!='completed':return False
    for name,want in q['prepared_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,'Predecessor hash changed: '+name
    for job in q['jobs']:
        assert job['status']=='completed' and job.get('returncode')==0
        assert job['completion']['optimizer_step']==59724
        final=Path(job['final_checkpoint'])
        for name in ['model.safetensors','config.json','training_state.pt']+[f'training_state_rank_{i:05d}.pt' for i in range(8)]:
            assert (final/name).stat().st_size>0,'Missing final: '+str(final/name)
        with (final/'model.safetensors').open('rb') as f:
            n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n))
        end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
        assert (final/'model.safetensors').stat().st_size==8+n+end,'Truncated final'
    return True


def main():
    with (BASE/'benchmark_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(STATE.read_text());state['supervisor_pid']=os.getpid()
        if state['status'] in ['completed','failed']:return
        try:
            hashes(state)
            state['status']='waiting_predecessor';save(state)
            while not all(validate_predecessor(p) for p in state.get('predecessors',[])):
                time.sleep(30)
            state['status']='waiting_manifest';save(state)
            while not Path(state['manifest_metadata']).exists():time.sleep(30)
            metadata=json.loads(Path(state['manifest_metadata']).read_text())
            manifest=Path(state['manifest'])
            assert hashlib.sha256(manifest.read_bytes()).hexdigest()==metadata['hashes'][manifest.name]
            state['manifest_sha256']=metadata['hashes'][manifest.name]
            for job in state['jobs']:
                if job['status']=='completed':continue
                assert job['status']=='pending','Ambiguous previous benchmark launch; inspect PIDs/claims'
                state['status']='waiting_resources';save(state)
                while not free():time.sleep(30)
                hashes(state)
                out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
                claim=out/'launch_claim.json'
                with claim.open('x') as f:json.dump({'model':job['model'],'temperature':job['temperature']},f)
                state['status']='running';job['status']='launching';save(state)
                children=run_shards(BASE,state,job,save)
                assert all(child.returncode==0 for child in children), 'Benchmark worker exits were not all zero'
                summary=subprocess.run([state['python'],str(BASE/'harness/summarize.py'),
                    '--results',job['output'],'--manifest',state['manifest']],text=True,capture_output=True)
                (out/'summary.log').write_text(summary.stdout+summary.stderr)
                assert summary.returncode==0,'Summary failed'
                result=json.loads((out/'summary.json').read_text());assert result['status']=='complete'
                job.update(status='completed',returncodes=[p.returncode for p in children]);save(state)
            state['status']='completed';save(state)
        except Exception as error:
            state.update(status='failed',error=type(error).__name__+': '+str(error));save(state)
            traceback.print_exc();raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--state',default=str(STATE))
    STATE=Path(parser.parse_args().state)
    main()
