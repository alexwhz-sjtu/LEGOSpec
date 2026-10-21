"""Prepare an isolated two-fit extension; immutable historical source and results."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,os,shutil,uuid
import yaml

ROOT=Path('/share/dai-sys/wanghanzhen/projects/DAC27')
PROJECT=ROOT/'LiteSpec_exp'
BASE=PROJECT/'tests/dflare_parallel_20261021'
OLD=PROJECT/'tests/h8001_component_search_20261020'
OLDBENCH=PROJECT/'tests/component_benchmark_h8001_20261020'
BENCH=BASE/'benchmark'
SERIES=ROOT/'runs/dflare_parallel_20261021'
WORLD=int(os.environ.get('TRAIN_WORLD_SIZE','8'))
assert WORLD in (4,8)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

assert not (BASE/'queue.json').exists(),'Do not overwrite an existing queue'
old=read(OLD/'queue.json');assert old['status']=='completed'
guards=read(OLD/'continuation_hashes.json')
for previous in (old,read(OLD/'phase2/queue.json')):
    for p,h in previous['prepared_hashes'].items():
        assert p not in guards or guards[p]==h,'Conflicting historical guards: '+p
        guards[p]=h
assert len(guards)==239
for p,h in guards.items():assert sha(p)==h,'Historical hash drift: '+p
assert read(OLDBENCH/'new_models_test_queue.json')['status']=='completed'
BASE.mkdir(exist_ok=True);BENCH.mkdir(exist_ok=True);(BASE/'configs').mkdir(exist_ok=True);SERIES.mkdir(exist_ok=False)
for name in ['harness','data','gpu_locks']:(BENCH/name).symlink_to(OLDBENCH/name,target_is_directory=True)
for name in ['benchmark_queue.py','available_gpu_scheduler.py']:shutil.copy2(OLDBENCH/name,BENCH/name)
shutil.copy2(OLD/'record_exit.py',BASE/'record_exit.py')
shutil.copy2(OLD/'environment.sh',BASE/'environment.sh')
supervisor=(OLD/'supervisor.py').read_text()
supervisor=supervisor.replace("STATE = BASE / 'queue.json'", "STATE = BASE / 'queue.json'\nWORLD_SIZE=int(json.loads(STATE.read_text())['world_size'])")
start=supervisor.index('def resources_free():');end=supervisor.index('\ndef assert_no_duplicate',start)
supervisor=supervisor[:start]+"def resources_free():\n    from resource_leases import free\n    return free(WORLD_SIZE,29521)\n\n"+supervisor[end:]
for a,b in [('range(8)','range(WORLD_SIZE)'),("['world_size'] == 8","['world_size'] == WORLD_SIZE"),('len(active) == 8','len(active) == WORLD_SIZE'),("'--nproc_per_node=8'","'--nproc_per_node='+str(WORLD_SIZE)"),("checked['world_size']==8","checked['world_size']==WORLD_SIZE")]:
    assert a in supervisor,a;supervisor=supervisor.replace(a,b)
(BASE/'supervisor.py').write_text(supervisor)
preflight=(OLD/'gpu_preflight_phase2.py').read_text()
a=preflight.index('RECIPES=');b=preflight.index('\ndef config',a)
preflight=preflight[:a]+"RECIPES=[{'type':'hard_ce'},{'type':'weighted_sum','components':[{'type':'hard_ce','weight':.1},{'type':'tv','weight':.9}]}]"+preflight[b:]
preflight=preflight.replace("[(p,r) for p in ['dflare','conv'] for r in RECIPES]","[('dflare',r) for r in RECIPES]")
preflight=preflight.replace("proposal='rnn'","proposal='parallel'")
preflight=preflight.replace('dist.get_world_size() == 8',"dist.get_world_size() == int(os.environ['TRAIN_WORLD_SIZE'])")
preflight=preflight.replace("'world_size':8","'world_size':dist.get_world_size()")
(BASE/'gpu_preflight.py').write_text(preflight)

template=yaml.safe_load((OLD/'phase2/configs/09_dflare_context_ce01_tv09.yaml').read_text())
recipe=read(OLD/'configs/dflare_context_ce01_tv09.json')
parallel=read(OLD/'configs/07_dflash_mha_parallel_ce01_tv09.json')['composable_config']['proposal']
jobs=[]
for number,loss,objective in [(11,'hard_ce',{'type':'hard_ce'}),(12,'ce01_tv09',{'type':'weighted_sum','components':[{'type':'hard_ce','weight':.1},{'type':'tv','weight':.9}]})]:
    label=f'{number:02d}_dflare_mha_parallel_{loss}';run=SERIES/label;run.mkdir()
    cfg=copy.deepcopy(recipe);cfg['composable_config']['proposal']=copy.deepcopy(parallel);cfg['composable_config']['objective']=objective
    assert cfg['composable_config']['context']=={'type':'dflare','target_layer_ids':[1,9,17,25,33]}
    draft=BASE/'configs'/(label+'.json');write(draft,cfg)
    launch=copy.deepcopy(template);launch['launcher'].update(nproc_per_node=WORLD,master_port=29521,output_dir=str(run/'checkpoints'))
    launch['arguments'].update(draft_config=str(draft),accumulation_steps=8//WORLD,wandb_name='dflare_parallel_20261021_'+label)
    assert launch['arguments']['batch_size']==1 and WORLD*launch['arguments']['accumulation_steps']==8
    for k,v in {'num_epochs':6,'seed':42,'learning_rate':.0006,'muon_adamw_lr':.0006,'muon_lr':.02,'warmup_ratio':.04,'use_muon':True,'max_length':4096,'num_anchors':512,'block_size':8}.items():assert launch['arguments'][k]==v
    path=BASE/'configs'/(label+'.yaml');path.write_text(yaml.safe_dump(launch,sort_keys=False))
    runid=uuid.uuid4().hex[:8]
    jobs.append(dict(number=number,label=label,proposal='parallel',loss=loss,status='pending',draft_config=str(draft),config=str(path),run_dir=str(run),log=str(run/'train.log'),wandb_id=runid,wandb_url='https://wandb.ai/alex-wang_sjtu-sjtu/spec-decoding/runs/'+runid,expected_steps=59724))

formal=read(OLDBENCH/'new_models_test_queue.json');benchjobs=[];comparisons=[]
for j in jobs:
    baseline_label='reused_A4_parallel_hard_ce' if j['loss']=='hard_ce' else '07_dflash_mha_parallel_ce01_tv09'
    for t in (0,1):
        output=SERIES/'formal_test'/j['label']/f'T{t}'
        benchjobs.append(dict(label=j['label'],model=str(Path(j['run_dir'])/'checkpoints/final'),proposal='parallel',loss=j['loss'],temperature=t,status='pending',output=str(output)))
        baseline=next(x for x in formal['jobs'] if x['label']==baseline_label and x['temperature']==t)
        comparisons.append(dict(candidate_label=j['label'],temperature=t,baseline=baseline['output']))
verify=(PROJECT/'tests/available_gpu_scheduler_20261021/verify_completed.py').read_text()
verify=verify.replace("BENCH = PROJECT/'tests/component_benchmark_h8001_20261020'","BENCH = Path(__file__).resolve().parent/'benchmark'")
verify=verify.replace("new_models_test_notification_state.json","verification_state.json")
(BASE/'verify_formal.py').write_text(verify)
write(BENCH/'verification_state.json',{'completed':[]})
write(BASE/'notification_state.json',dict(started_verified=[],completed=[],benchmark_completed=[],all_completed_notified=False))
data=Path(template['launcher']['train_data']);assert sha(data)==old['data_sha256']
manifest=Path(formal['manifest']);assert sha(manifest)=='bdfd20787753294352983d7712951c1b03dd5cc148a0f2ef746b235bd7ce1c79'
plan=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorized_request='Train DFlare+Parallel with CE and .1CE+.9TV from scratch, then formal T0/T1 testing',world_size=WORLD,global_batch=8,accumulation_steps=8//WORLD,expected_steps=59724,epochs=6,seed=42,target_features=[1,9,17,25,33],block_size=8,source_snapshot=old['snapshot'],data_sha256=old['data_sha256'],training_jobs=[j['label'] for j in jobs],formal_jobs=4,requests_per_job=1056,comparisons=comparisons,unchanged_historical_protection=len(guards),resource_policy='Only idle GPUs; never terminate or claim other users processes; training waits for required rank count, benchmark schedules 8 logical shards on 1-8 idle GPUs')
write(BASE/'plan.json',plan)
protected=dict(guards)
for p in list(BASE.glob('*.py'))+list((BASE/'configs').glob('*'))+[BASE/'environment.sh',BASE/'plan.json',BENCH/'benchmark_queue.py',BENCH/'available_gpu_scheduler.py']:
    protected[str(p)]=sha(p)
for p in BASE.rglob('*.py'):
    if p.is_symlink() or 'harness' in p.parts:continue
    ast.parse(p.read_text(),filename=str(p))
write(BASE/'protected_hashes.json',protected)
write(BASE/'queue.json',dict(status='prepared',cursor=0,series_name='dflare_parallel_20261021',series_dir=str(SERIES),snapshot=old['snapshot'],python=old['python'],jobs=jobs,data_sha256=old['data_sha256'],prepared_hashes=protected,preflight_passed=False,preflight_status='pending',world_size=WORLD,execution_host='H800_1'))
write(BASE/'controller_state.json',dict(status='prepared',python=old['python'],world_size=WORLD,series_dir=str(SERIES)))
write(BENCH/'new_models_test_queue.json',dict(status='prepared',python=formal['python'],source=formal['source'],target=formal['target'],manifest=formal['manifest'],manifest_metadata=formal['manifest_metadata'],jobs=benchjobs,prepared_hashes=protected,predecessors=[]))
print(json.dumps(dict(status='prepared',world_size=WORLD,jobs=jobs,protected_files=len(protected)),indent=2))
