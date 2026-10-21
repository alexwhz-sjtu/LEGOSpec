"""Prepare the authorized three-run series without touching the running queue."""
from pathlib import Path
import copy
import datetime
import hashlib
import json
import shutil
import subprocess
import tarfile

import yaml

ROOT = Path('/data/temp/LiteSpec_exp')
BASE = ROOT / 'tests/selector_loss_ablation_20261018'
OLD = ROOT / 'tests/retrain_1234_metrics_v2_20261017'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (BASE/'queue.json').exists(), 'Queue already prepared; inspect instead of replacing'
    old = json.loads((OLD/'queue.json').read_text())
    baseline = old['jobs'][0]
    recipe = json.loads(Path(baseline['draft_config']).read_text())
    settings = yaml.safe_load(Path(baseline['config']).read_text())
    assert recipe['composable_config']['objective'] == {'type':'hard_ce'}
    assert recipe['composable_config']['context']['type'] == 'dflash'
    assert recipe['composable_config']['proposal']['type'] == 'selector'
    assert all(x['attention']['representation']=='mha' and x['attention']['access']=='dense'
               for x in recipe['composable_config']['backbone']['layers'])
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    series = 'selector_base_loss_ablation_' + stamp
    directory = Path('/mnt/bd-1/task/339953/artifacts/LiteSpec/runs')/series
    directory.mkdir()
    (BASE/'configs').mkdir(exist_ok=True)
    (directory/'configs').mkdir()
    objectives = [('01_dflash_mha_selector_dpace', {'type':'dpace','alpha':0.5}),
                  ('02_dflash_mha_selector_nla', {'type':'nla'}),
                  ('03_dflash_mha_selector_ce01_tv09', {'type':'weighted_sum','components':[
                      {'type':'hard_ce','weight':0.1},{'type':'tv','weight':0.9}]})]
    jobs = []
    from wandb.util import generate_id
    for i,(label,objective) in enumerate(objectives,1):
        c=copy.deepcopy(recipe); c['composable_config']['objective']=objective
        check=copy.deepcopy(c); check['composable_config']['objective']={'type':'hard_ce'}
        assert check==recipe, 'Only the base objective may change'
        config=BASE/'configs'/f'{label}.json'
        config.write_text(json.dumps(c,indent=2)+'\n')
        y=copy.deepcopy(settings);run=directory/label;run.mkdir()
        y['launcher']['output_dir']=str(run/'checkpoints')
        y['arguments']['draft_config']=str(config)
        y['arguments']['wandb_name']=series+'_'+label
        ys=BASE/'configs'/f'{label}.yaml';ys.write_text(yaml.safe_dump(y,sort_keys=False))
        for path in [config,ys]:shutil.copy2(path,directory/'configs'/path.name)
        wandb_id=generate_id()
        jobs.append({'number':i,'label':label,'status':'pending','draft_config':str(config),
                     'config':str(ys),'run_dir':str(run),'log':str(run/'train.log'),
                     'wandb_id':wandb_id,'wandb_url':'https://wandb.ai/alex-wang_sjtu-sjtu/spec-decoding/runs/'+wandb_id,
                     'expected_steps':59724})
    snapshot=BASE/'snapshot'
    state={'status':'prepared','cursor':0,'series_name':series,'series_dir':str(directory),
           'snapshot':str(snapshot),'python':str(ROOT/'.venv/bin/python'),
           'env_file':old['env_file'],'jobs':jobs,'predecessor_queue':str(OLD/'queue.json'),
           'data_sha256':old['data_sha256'],'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    files=[p for p in snapshot.rglob('*') if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts]
    files+=list((BASE/'configs').glob('*'))
    files += [BASE/n for n in ['supervisor.py','gpu_preflight.py','test_losses.py']]
    state['prepared_hashes']={str(p):digest(p) for p in files}
    # Keep dataset identity and seed identical to the first baseline.
    assert digest(Path(settings['launcher']['train_data']))==state['data_sha256']
    validation=[]
    for j in jobs:
        p=subprocess.run([state['python'],'scripts/config/launch_composable.py','--config',j['config'],'--dry-run'],
                         cwd=snapshot,capture_output=True,text=True)
        (BASE/(j['label']+'_dryrun.log')).write_text(p.stdout+p.stderr)
        assert p.returncode==0,p.stderr
        validation.append({'number':j['number'],'dry_run_returncode':p.returncode,'only_objective_changed':True})
    (BASE/'validation.json').write_text(json.dumps(validation,indent=2))
    shutil.copy2(BASE/'validation.json',directory/'validation.json')
    for name in ['test_losses.py','gpu_preflight.py','supervisor.py','unit_tests.log']:
        shutil.copy2(BASE/name,directory/name)
    with tarfile.open(directory/'source_snapshot.tar.gz','w:gz') as archive:
        for p in files:
            if p.is_relative_to(snapshot):archive.add(p,arcname=str(Path('snapshot')/p.relative_to(snapshot)),recursive=False)
    text=json.dumps(state,indent=2)+'\n'
    (BASE/'queue.json').write_text(text);(directory/'queue_status.json').write_text(text)
    (BASE/'notification_state.json').write_text(json.dumps({'started_verified':[],'completed':[]},indent=2))
    print(json.dumps({k:state[k] for k in ['series_dir','series_name','jobs']},indent=2))


if __name__ == '__main__':main()
