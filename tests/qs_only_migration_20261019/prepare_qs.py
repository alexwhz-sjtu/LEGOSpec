"""Prepare a new QS-only pipeline; does not alter an active source or queue."""
from pathlib import Path
import json,hashlib,shutil,copy,datetime
import yaml

PROJECT=Path('/data/temp/LiteSpec_exp')
BASE=PROJECT/'tests/staged_component_search_qs_20261019'
IMPORTED=PROJECT/'tests/qs_only_migration_20261019/imported_evidence/LiteSpec_exp/tests/staged_component_search_20261018'
BENCH=PROJECT/'tests/component_benchmark_20261018'
ARCHIVE=Path('/mnt/bd-1/task/339953/artifacts/LiteSpec/migrations/Inf_H800_2_20261019')
SERIES=Path('/mnt/bd-1/task/339953/artifacts/LiteSpec/runs/staged_component_search_qs_20261019')
assert not (BASE/'queue.json').exists(),'Do not replace a deployed queue'
BASE.mkdir(exist_ok=True);SERIES.mkdir(exist_ok=True)
shutil.copytree(IMPORTED/'snapshot',BASE/'snapshot',ignore=shutil.ignore_patterns('.cache','__pycache__','cache'))
for name in ['supervisor.py','record_exit.py','gpu_preflight.py','gpu_preflight_phase2.py','continuation.py']:
 shutil.copy2(IMPORTED/name,BASE/name)
recipes="RECIPES=[{'type':'hard_ce'},{'type':'dpace','alpha':.5},{'type':'nla','loss_decay_gamma':4.0},{'type':'weighted_sum','components':[{'type':'hard_ce','weight':.1},{'type':'tv','weight':.9}]},{'type':'bv'}]"
for name in ['gpu_preflight.py','gpu_preflight_phase2.py']:
 p=BASE/name;s=p.read_text();a=s.index('RECIPES=');b=s.index('\ndef config',a);s=s[:a]+recipes+s[b:];p.write_text(s)
(BASE/'configs').mkdir()
launch=yaml.safe_load((IMPORTED/'configs/01_dflash_mha_rnn_hard_ce.yaml').read_text())
launch['launcher']['target_model']='/data/temp/models/Qwen3-4B'
launch['launcher']['train_data']='/data/temp/datasets/open_perfectblend_80k_qwen3_4b.jsonl'
launch['arguments']['cache_dir']=str(BASE/'cache/train')
cases=[(p,l) for p in ['rnn','parallel'] for l in ['dpace','nla','ce01_tv09','bv']]
objectives={'hard_ce':{'type':'hard_ce'},'dpace':{'type':'dpace','alpha':.5},'nla':{'type':'nla','loss_decay_gamma':4.0},'ce01_tv09':{'type':'weighted_sum','components':[{'type':'hard_ce','weight':.1},{'type':'tv','weight':.9}]},'bv':{'type':'bv'}}
jobs=[]
for number,(proposal,loss) in enumerate(cases,1):
 label=f'{number:02d}_dflash_mha_{proposal}_{loss}'+('_gamma4' if loss=='nla' else '')
 old=IMPORTED/'configs'/('01_dflash_mha_rnn_hard_ce.json' if proposal=='rnn' else '05_dflash_mha_parallel_dpace.json')
 config=json.loads(old.read_text());config['composable_config']['objective']=objectives[loss]
 path=BASE/'configs'/(label+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
 rd=SERIES/label;rd.mkdir();cfg=copy.deepcopy(launch);cfg['launcher']['output_dir']=str(rd/'checkpoints')
 cfg['arguments'].update(draft_config=str(path),wandb_name='staged_component_search_qs_20261019_'+label)
 y=path.with_suffix('.yaml');y.write_text(yaml.safe_dump(cfg,sort_keys=False))
 wid=hashlib.sha256(('staged_component_search_qs_20261019:'+label).encode()).hexdigest()[:8]
 jobs.append({'number':number,'label':label,'proposal':proposal,'loss':loss,'status':'pending','draft_config':str(path),'config':str(y),'run_dir':str(rd),'log':str(rd/'train.log'),'wandb_id':wid,'wandb_url':'https://wandb.ai/alex-wang_sjtu-sjtu/spec-decoding/runs/'+wid,'expected_steps':59724})
for stage in ['dflare_context','dflash_layer_conv']:
 for loss,objective in objectives.items():
  source=IMPORTED/'configs'/(stage+'_'+('hard_ce' if loss=='bv' else loss)+'.json')
  cfg=json.loads(source.read_text());cfg['composable_config']['objective']=objective
  (BASE/'configs'/(stage+'_'+loss+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
reuse=[{'label':'reused_RNN_hard_ce_from_inf_final','proposal':'rnn','loss':'hard_ce','model':str(ARCHIVE/'current_rnn_ce/final')},
 {'label':'reused_A4_parallel_hard_ce','proposal':'parallel','loss':'hard_ce','model':'/mnt/bd-1/task/339953/artifacts/LiteSpec/runs/retrain_1234_metrics_v2_20261017_220434/04_dflash_mha_parallel/checkpoints/final'}]
(BASE/'reuse_models.json').write_text(json.dumps(reuse,indent=2)+'\n')
p=BASE/'continuation.py';s=p.read_text().replace("tiebreak=['hard_ce','nla','ce01_tv09','dpace']","tiebreak=['hard_ce','nla','ce01_tv09','dpace','bv']").replace('prefer CE,NLA,CE+TV,D-PACE','prefer CE,NLA-gamma4,CE+TV,D-PACE,BV').replace("[(8,'dflare_context'),(9,'dflash_layer_conv')]","[(9,'dflare_context'),(10,'dflash_layer_conv')]")
a=s.index("            receipt=read(BENCH/'reuse_A4_receipt.json')");b=s.index("            state['status']='selection_benchmark'",a)
s=s[:a]+"            reuse=read(BASE/'reuse_models.json')\n            imported=read(BASE/'imported_rnn_ce_verified.json')\n            assert imported['verified'] and imported['returncode']==0 and imported['optimizer_step']==59724 and imported['wandb_state']=='finished'\n            for item in imported['files']:\n                assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']\n            models.extend(reuse)\n"+s[b:]
s=s.replace("'00_reused_A4_parallel_hard_ce'","'reused_A4_parallel_hard_ce'")
p.write_text(s)
state={'status':'prepared','cursor':0,'series_name':'staged_component_search_qs_20261019','series_dir':str(SERIES),'snapshot':str(BASE/'snapshot'),'python':str(PROJECT/'.venv/bin/python'),'jobs':jobs,'data_sha256':'b8fb09cec39962888b08d457836bcb2bfff0bd988df51ce4910a065e24edc168','preflight_status':'pending','preflight_passed':False,'execution_host':'QS_H800','reuse':reuse,'notes':'Eight new jobs; reuse one user-authorized finishing Inf RNN+CE and A4. NLA objective gamma4; BV added. All GPU work on QS. Gate waits complete existing 14 benchmarks and verified current Inf final migration.'}
controller={'status':'waiting_qs_prerequisites','python':state['python'],'snapshot':state['snapshot'],'series_dir':str(SERIES),'target':'/data/temp/models/Qwen3-4B','stage1_queue':str(BASE/'queue.json'),'selection_rule':'seed43 disjoint 5-task x 2 temperature x 2 proposal; 5 losses including BV and NLA-gamma4; 10 models; test never selects','new_training_count':10,'execution_host':'QS_H800'}
(BASE/'queue.json').write_text(json.dumps(state,indent=2)+'\n');(BASE/'continuation_state.json').write_text(json.dumps(controller,indent=2)+'\n')
(BASE/'notification_state.json').write_text(json.dumps({'started_verified':[],'completed':[]},indent=2)+'\n')
print(json.dumps({'prepared_jobs':len(jobs),'models_with_reuse':10,'base':str(BASE),'series':str(SERIES)}))
