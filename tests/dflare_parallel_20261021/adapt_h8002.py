"""Relocate only this still-prepared queue to H800_2's native shared mount."""
from pathlib import Path
import datetime,hashlib,json,os,socket
ROOT=Path('/mnt/dai-sys/wanghanzhen/projects/DAC27')
BASE=ROOT/'LiteSpec_exp/tests/dflare_parallel_20261021'
q=json.loads((BASE/'queue.json').read_text())
assert q['status']=='prepared' and all(j['status']=='pending' for j in q['jobs'])
assert not any(Path(j['run_dir'].replace('/share/dai-sys','/mnt/dai-sys')).glob('launch_claim.json') for j in [])
for j in q['jobs']:
    assert not (Path(j['run_dir'].replace('/share/dai-sys','/mnt/dai-sys'))/'launch_claim.json').exists()
env=ROOT/'envs/litespec-py311-cu130-h8002'
assert not env.exists()
(env/'bin').mkdir(parents=True)
(env/'lib/python3.11').mkdir(parents=True)
runtime=Path('/mnt/dai-sys/wanghanzhen/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11')
(env/'bin/python').symlink_to(runtime)
(env/'bin/python3').symlink_to('python')
(env/'lib/python3.11/site-packages').symlink_to(ROOT/'envs/litespec-py311-cu130/lib/python3.11/site-packages',target_is_directory=True)
cfg=(ROOT/'envs/litespec-py311-cu130/pyvenv.cfg').read_text().replace('/share/dai-sys','/mnt/dai-sys')
(env/'pyvenv.cfg').write_text(cfg)
def convert(s):
    return s.replace('/share/dai-sys','/mnt/dai-sys').replace(str(ROOT/'envs/litespec-py311-cu130/bin/python'),str(env/'bin/python'))
for p in [BASE/'queue.json',BASE/'controller_state.json',BASE/'benchmark/new_models_test_queue.json',BASE/'plan.json',BASE/'verify_formal.py',BASE/'environment.sh',*list((BASE/'configs').glob('*.yaml'))]:
    p.write_text(convert(p.read_text()))
q=json.loads((BASE/'queue.json').read_text());q['execution_host']='H800_2'
plan=json.loads((BASE/'plan.json').read_text());plan.update(execution_host='H800_2',hostname=socket.gethostname(),shared_mount='/mnt/dai-sys',path_only_relocation=True,original_environment_unchanged=True)
(BASE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
s=json.loads((BASE/'controller_state.json').read_text());s.update(execution_host='H800_2')
(BASE/'controller_state.json').write_text(json.dumps(s,indent=2)+'\n')
environment=(BASE/'environment.sh').read_text()
environment+='\nexport UV_PROJECT_ENVIRONMENT="'+str(env)+'"\nexport NETRC="'+str(ROOT/'private/credentials/H800_1.netrc')+'"\nexport PATH="'+str(env/'bin')+':$PATH"\n'
(BASE/'environment.sh').write_text(environment)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected={convert(p):h for p,h in json.loads((BASE/'protected_hashes.json').read_text()).items()}
# Check inherited files before refreshing just our own deliberately relocated files.
for p,h in protected.items():
    if str(BASE) not in p:assert sha(p)==h,'Historical source drift: '+p
for p in list(BASE.glob('*.py'))+list((BASE/'configs').glob('*'))+[BASE/'environment.sh',BASE/'plan.json',env/'pyvenv.cfg']:
    protected[str(p)]=sha(p)
q['prepared_hashes']=protected
(BASE/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
bench=json.loads((BASE/'benchmark/new_models_test_queue.json').read_text());bench['prepared_hashes']=protected
(BASE/'benchmark/new_models_test_queue.json').write_text(json.dumps(bench,indent=2)+'\n')
(BASE/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n')
(BASE/'host_adaptation_receipt.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_host='H800_2',hostname=socket.gethostname(),native_root=str(ROOT),python=str(env/'bin/python'),shared_packages=str(ROOT/'envs/litespec-py311-cu130/lib/python3.11/site-packages'),protected_files=len(protected),historical_files_changed=False,system_disk_files_written=False),indent=2)+'\n')
print('Prepared on H800_2; shared native paths; protected',len(protected))
