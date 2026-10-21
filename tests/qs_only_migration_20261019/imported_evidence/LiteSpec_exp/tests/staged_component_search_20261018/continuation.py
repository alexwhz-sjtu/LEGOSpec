"""S1 -> disjoint-panel loss selection -> context/conv S2/S3 -> test panel.

No training is launched directly here: all launches use the guarded supervisor.
This controller never reads the test panel to select a loss.
"""
from pathlib import Path
import copy,datetime,fcntl,hashlib,importlib.util,json,os,subprocess,time,traceback

BASE=Path(__file__).resolve().parent
BENCH=BASE.parent/'component_benchmark_20261018'
STATE=BASE/'continuation_state.json'


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def event(state,label):
    state['updated_at']=now();write(STATE,state);print(json.dumps({'at':now(),'event':label,'status':state['status']}),flush=True)
def protected():
    for name,want in read(BASE/'continuation_hashes.json').items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==want,'Continuation file changed: '+name
def live(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def wait_queue(path):
    while True:
        q=read(path)
        assert q['status']!='failed','Queue failed: '+str(q.get('error'))
        if q['status']=='completed':return q
        assert live(q['supervisor_pid']),'Queue supervisor stopped; inspect receipt and process identities'
        time.sleep(30)
def validate_training(q):
    spec=importlib.util.spec_from_file_location('frozen_supervisor',BASE/'supervisor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_hashes(q)
    import wandb
    api=wandb.Api(timeout=60)
    for job in q['jobs']:
        assert job['status']=='completed' and job.get('returncode')==0
        final,completion=module.validate_final(job)
        assert completion['optimizer_step']==59724
        for attempt in range(20):
            api.flush()
            run=api.run('alex-wang_sjtu-sjtu/spec-decoding/'+job['wandb_id'])
            if run.state=='finished' and run.lastHistoryStep>=59724:break
            time.sleep(30)
        else:raise AssertionError('Online run is not finished: '+job['wandb_id'])


def panel(state,name,models,manifest):
    queue=BENCH/(name+'_queue.json')
    if not queue.exists():
        root=Path(state['series_dir'])/name;root.mkdir(exist_ok=True)
        jobs=[]
        for model in models:
            for temperature in [0,1]:
                jobs.append({'label':model['label'],'model':model['model'],
                    'proposal':model['proposal'],'loss':model['loss'],'temperature':temperature,
                    'status':'pending','output':str(root/model['label']/f'T{temperature}')})
        static=read(BASE/'continuation_hashes.json')
        write(queue,{'status':'prepared','python':state['python'],'source':state['snapshot'],
            'target':state['target'],'manifest':str(BENCH/'data'/manifest),
            'manifest_metadata':str(BENCH/'data/manifest_metadata.json'),
            'jobs':jobs,'prepared_hashes':static,'predecessors':[]})
    existing=read(queue)
    if existing['status']=='completed':return existing
    assert existing['status']!='failed','Benchmark panel failed: '+str(existing.get('error'))
    if existing.get('supervisor_pid'):
        assert live(existing['supervisor_pid']),'Panel stopped with an ambiguous launch; inspect before recovery'
    else:
        with (BASE/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen([state['python'],str(BENCH/'benchmark_queue.py'),'--state',str(queue)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state[name+'_pid']=child.pid;event(state,name+'_launched')
        for attempt in range(100):
            existing=read(queue)
            if existing.get('supervisor_pid'):break
            time.sleep(.1)
    return wait_queue(queue)


def choose_loss(panel_queue):
    names=['gsm8k','alpaca','mbpp','livecodebench','math500']
    scores={};tiebreak=['hard_ce','nla','ce01_tv09','dpace']
    for loss in tiebreak:
        values=[]
        for proposal in ['rnn','parallel']:
            for temperature in [0,1]:
                matches=[j for j in panel_queue['jobs'] if j['loss']==loss and
                    j['proposal']==proposal and j['temperature']==temperature]
                assert len(matches)==1 and matches[0]['status']=='completed'
                result=read(Path(matches[0]['output'])/'summary.json')
                assert result['status']=='complete'
                for task in names:
                    value=result['groups'][task+'/all']['request_macro_tau']
                    assert value is not None and 1<=value<=8
                    values.append(value)
        scores[loss]=sum(values)/len(values)
    best=max(scores.values());winner=next(x for x in tiebreak if scores[x]>=best-.01)
    return {'winner':winner,'scores':scores,'rule':'mean over 5 tasks x 2 temperatures x 2 proposals; within .01 prefer CE,NLA,CE+TV,D-PACE','selected_at':now()}


def create_phase2(state,first,choice):
    import yaml
    directory=BASE/'phase2';directory.mkdir(exist_ok=True);queue=directory/'queue.json'
    if queue.exists():return queue
    assert not (directory/'launch_claim.json').exists()
    for name,source in [('supervisor.py',BASE/'supervisor.py'),('record_exit.py',BASE/'record_exit.py'),
                        ('gpu_preflight.py',BASE/'gpu_preflight_phase2.py')]:
        (directory/name).write_bytes(source.read_bytes())
    configs=directory/'configs';configs.mkdir(exist_ok=True)
    launch_base=yaml.safe_load(Path(first['jobs'][0]['config']).read_text())
    jobs=[]
    for number,stage in [(8,'dflare_context'),(9,'dflash_layer_conv')]:
        label=f'{number:02d}_{stage}_{choice["winner"]}'
        rd=Path(state['series_dir'])/label;rd.mkdir(exist_ok=True)
        draft=BASE/'configs'/(stage+'_'+choice['winner']+'.json')
        launch=copy.deepcopy(launch_base);launch['launcher']['output_dir']=str(rd/'checkpoints')
        launch['arguments'].update(draft_config=str(draft),wandb_name=first['series_name']+'_'+label)
        path=configs/(label+'.yaml');path.write_text(yaml.safe_dump(launch,sort_keys=False))
        run_id=hashlib.sha256((first['series_name']+label).encode()).hexdigest()[:8]
        jobs.append({'number':number,'label':label,'proposal':'rnn','loss':choice['winner'],
            'status':'pending','draft_config':str(draft),'config':str(path),'run_dir':str(rd),
            'log':str(rd/'train.log'),'wandb_id':run_id,
            'wandb_url':'https://wandb.ai/alex-wang_sjtu-sjtu/spec-decoding/runs/'+run_id,'expected_steps':59724})
    frozen=read(BASE/'continuation_hashes.json')
    for file in directory.rglob('*'):
        if file.is_file() and file.suffix in ['.py','.yaml']:
            frozen[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    payload={k:first[k] for k in ['series_name','series_dir','snapshot','python','data_sha256']}
    payload.update(status='prepared',cursor=0,jobs=jobs,prepared_hashes=frozen,
                   preflight_status='pending',preflight_passed=False,selection_receipt=choice)
    write(queue,payload)
    return queue


def main():
    with (BASE/'continuation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(STATE)
        if state['status'] in ['completed','failed']:return
        state['controller_pid']=os.getpid();event(state,'controller_attached')
        try:
            protected();state['status']='waiting_stage1';event(state,'waiting_stage1')
            first=wait_queue(BASE/'queue.json');validate_training(first)
            models=[{'label':j['label'],'proposal':j['proposal'],'loss':j['loss'],
                     'model':j['final_checkpoint']} for j in first['jobs']]
            receipt=read(BENCH/'reuse_A4_receipt.json')
            for name,want in receipt['hashes'].items():
                assert hashlib.sha256((BENCH/'reuse_A4'/name).read_bytes()).hexdigest()==want
            models.append({'label':'00_reused_A4_parallel_hard_ce','proposal':'parallel',
                           'loss':'hard_ce','model':str(BENCH/'reuse_A4')})
            state['status']='selection_benchmark';event(state,'stage1_completed_selection_started')
            selected=panel(state,'selection',models,'selection_manifest.jsonl')
            protected();choice=choose_loss(selected)
            existing=BASE/'loss_selection.json'
            if existing.exists():
                assert read(existing)['winner']==choice['winner'],'Winner changed'
            else:write(existing,choice)
            state['winner']=choice;state['status']='stage2_training';event(state,'loss_selected')
            phase2=create_phase2(state,first,choice);second=read(phase2)
            if second['status']=='prepared':
                with (BASE/'phase2/supervisor.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen([state['python'],str(BASE/'phase2/supervisor.py')],
                        env=dict(os.environ,OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                for attempt in range(100):
                    if read(phase2).get('supervisor_pid'):break
                    time.sleep(.1)
            second=wait_queue(phase2);validate_training(second)
            models.extend({'label':j['label'],'proposal':j['proposal'],'loss':j['loss'],
                           'model':j['final_checkpoint']} for j in second['jobs'])
            state['status']='test_benchmark';event(state,'context_and_conv_finished_test_started')
            tested=panel(state,'new_models_test',models,'manifest.jsonl')
            for temperature in [0,1]:
                base_job=next(j for j in tested['jobs'] if j['label']=='00_reused_A4_parallel_hard_ce'
                              and j['temperature']==temperature)
                for candidate in [j for j in tested['jobs'] if j['temperature']==temperature
                                  and j['label']!=base_job['label']]:
                    code=subprocess.run([state['python'],str(BENCH/'harness/compare.py'),
                        '--baseline',base_job['output'],'--candidate',candidate['output'],
                        '--output',str(Path(candidate['output'])/'paired_vs_A4.json')]).returncode
                    assert code==0,'Paired acceptance comparison failed'
            write(BASE/'final_results.json',{'selection':choice,'models':models,
                'test_summary_paths':[str(Path(j['output'])/'summary.json') for j in tested['jobs']]})
            state['status']='completed';event(state,'all_staged_experiments_and_benchmarks_completed')
        except Exception as error:
            state.update(status='failed',error=type(error).__name__+': '+str(error));event(state,'continuation_failed')
            traceback.print_exc();raise


if __name__=='__main__':main()
