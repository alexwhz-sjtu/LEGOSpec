import json,os,subprocess,time
from pathlib import Path
import yaml
root=Path('/data/temp/LiteSpec_exp')
setup=root/'tests/h800_setup_20261016'
run=json.loads((setup/'run.json').read_text())
config=yaml.safe_load((root/'scripts/config/composable.yaml').read_text())
assert config['launcher']['nproc_per_node']==8
assert config['arguments']['num_epochs']==6 and config['arguments']['use_muon'] is True
assert not config['launcher'].get('train_hidden_states')
assert config['launcher']['output_dir']==run['checkpoint_dir']
assert 'ALL_8GPU_COMPUTE_AND_NCCL_PASSED' in (setup/'gpu_nccl_validation.log').read_text()
assert 'ALL_CUDA_EXTENSION_VALIDATIONS_PASSED' in (setup/'cuda_extension_validation.log').read_text()
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:
  cmd=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if 'scripts.composable.train' in cmd:
   raise RuntimeError(f'Training already running at PID {entry.name}; refusing duplicate launch')
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
with Path(run['log']).open('ab',buffering=0) as log:
 p=subprocess.Popen(['bash','scripts/composable/run_training_composable.sh','scripts/config/composable.yaml'],cwd=root,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
run['pid']=p.pid;run['started_at']=time.time()
(setup/'run.json').write_text(json.dumps(run,indent=2))
(Path(run['run_dir'])/'run.json').write_text(json.dumps(run,indent=2))
print(json.dumps(run,indent=2),flush=True)

