"""Two new, authorized DFlare+Parallel fits, then the immutable formal panel."""
from pathlib import Path
import datetime,fcntl,hashlib,importlib.util,json,math,os,subprocess,time,traceback

BASE=Path(__file__).resolve().parent
STATE=BASE/'controller_state.json'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ticks(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return x[19] if x[0]!='Z' else None
    except FileNotFoundError:return None
def event(s,status):
    s.update(status=status,updated_at=now());write(STATE,s)
    print(json.dumps({'at':now(),'status':status}),flush=True)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def protected():
    for p,h in read(BASE/'protected_hashes.json').items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,'Protected file changed: '+p
def spawn(s,key,script,args=()):
    if s.get(key):
        assert ticks(s[key]['pid'])==s[key]['start_ticks'],'Controller child stopped unexpectedly: '+key
        return
    protected()
    with (BASE/(key+'.log')).open('ab',buffering=0) as log:
        child=subprocess.Popen([s['python'],str(script),*args],env=dict(os.environ,TRAIN_WORLD_SIZE=str(s['world_size'])),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s[key]={'pid':child.pid,'start_ticks':ticks(child.pid)};write(STATE,s)
def wait(s,path,key):
    while True:
        q=read(path)
        assert q['status']!='failed',str(q.get('error'))
        if q['status']=='completed':return q
        assert ticks(s[key]['pid'])==s[key]['start_ticks'],'Supervisor stopped before completion: '+key
        time.sleep(15)
def validate_training(s,q):
    import torch,wandb
    supervisor=module(BASE/'supervisor.py','new_dflare_supervisor')
    api=wandb.Api(timeout=60);evidence=[]
    for job in q['jobs']:
        assert job['returncode']==0
        final,result=supervisor.validate_final(job)
        for rank in range(q['world_size']):
            # These are our own newly generated rank checkpoints, including
            # NumPy RNG state which is not accepted by the weights-only loader.
            x=torch.load(Path(final)/f'training_state_rank_{rank:05d}.pt',map_location='cpu',weights_only=False,mmap=True)
            meta=x.get('metadata',x)
            supervisor.validate_metadata(meta,job)
            assert meta['distributed_training_state']['rank']==rank
            assert meta['scheduler_learning_rate']==.0006 and meta['scheduler_warmup_ratio']==.04
            assert meta['muon_config']['lr']==.02 and meta['muon_config']['adamw_lr']==.0006
            assert meta['target_layer_ids']==[1,9,17,25,33] and meta['block_size']==8
            assert meta['batch_size']==1 and meta['accumulation_steps']==8//q['world_size']
            del x
        for attempt in range(40):
            api.flush();run=api.run('alex-wang_sjtu-sjtu/spec-decoding/'+job['wandb_id'])
            if run.state=='finished' and run.lastHistoryStep>=59724:break
            time.sleep(30)
        else:raise AssertionError('W&B did not finish: '+job['wandb_id'])
        summary=dict(run.summary)
        required=['detail/composable/hard_label/token_accuracy','train/walk_accepted_length','train/expected_accept_len']
        values={k:summary[k] for k in required}
        assert all(math.isfinite(float(v)) for v in values.values())
        evidence.append(dict(label=job['label'],final=final,completion=result,rank_states=q['world_size'],wandb_id=job['wandb_id'],wandb_state=run.state,metrics=values))
    write(BASE/'training_verified.json',dict(verified_at=now(),jobs=evidence))
    return evidence
def finish(s,q):
    verify=module(BASE/'verify_formal.py','new_formal_verifier')
    verify.main(recheck_all=True)
    old=read(BASE/'plan.json')['comparisons'];pairs=[]
    for job in q['jobs']:
        baseline=next(x for x in old if x['candidate_label']==job['label'] and x['temperature']==job['temperature'])
        dest=Path(job['output'])/'paired_vs_dflash_parallel.json'
        code=subprocess.run([s['python'],str(BASE/'benchmark/harness/compare.py'),'--baseline',baseline['baseline'],
            '--candidate',job['output'],'--output',str(dest)],check=True).returncode
        rows=read(dest);assert sum(x['defined_pairs']+x['undefined_pairs'] for x in rows.values())==1056
        pairs.append(dict(candidate=job['label'],temperature=job['temperature'],baseline=baseline['baseline'],path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    result=dict(status='completed_verified',verified_at=now(),training=read(BASE/'training_verified.json'),
        formal_jobs=4,formal_requests=4224,shards=32,paired_comparisons=pairs,
        summaries=[dict(label=j['label'],temperature=j['temperature'],summary=str(Path(j['output'])/'summary.json')) for j in q['jobs']],
        system_profiling='not_started',plan=str(BASE/'plan.json'))
    write(BASE/'final_results.json',result)
    write(Path(s['series_dir'])/'final_results.json',result)
def main():
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=read(STATE)
        if s['status'] in ('completed','failed'):return
        s.update(controller_pid=os.getpid(),controller_start_ticks=ticks(os.getpid()))
        try:
            protected();event(s,'training')
            path=BASE/'queue.json'
            if read(path)['status']!='completed':spawn(s,'training_supervisor',BASE/'supervisor.py')
            q=wait(s,path,'training_supervisor')
            event(s,'verifying_training');validate_training(s,q);protected()
            event(s,'formal_benchmark')
            bench=BASE/'benchmark/new_models_test_queue.json'
            if read(bench)['status']!='completed':spawn(s,'benchmark_supervisor',BASE/'benchmark/benchmark_queue.py',['--state',str(bench)])
            q=wait(s,bench,'benchmark_supervisor');protected();finish(s,q);event(s,'completed')
        except Exception as e:
            s['error']=type(e).__name__+': '+str(e);event(s,'failed');traceback.print_exc();raise
if __name__=='__main__':main()
