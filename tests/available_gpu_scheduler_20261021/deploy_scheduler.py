"""Audited replacement of an idle, guarded benchmark controller only."""
from pathlib import Path
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/share/dai-sys/wanghanzhen/projects/DAC27')
PROJECT = ROOT/'LiteSpec_exp'
BASE = PROJECT/'tests/h8001_component_search_20261020'
BENCH = PROJECT/'tests/component_benchmark_h8001_20261020'
PREPARED = PROJECT/'tests/available_gpu_scheduler_20261021'
QUEUE = BENCH/'new_models_test_queue.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temp = path.with_name(path.name+'.gpu_scheduler_update.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def process(pid):
    try:
        d = Path(f'/proc/{pid}')
        parts = (d/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=parts[0], start_ticks=parts[19],
                    command=(d/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def main():
    q = read(QUEUE)
    c = read(BASE/'continuation_state.json')
    assert q['status'] == 'waiting_resources', 'Only replace a waiting controller'
    assert all(j['status'] in ('completed', 'pending') for j in q['jobs'])
    assert c['status'] == 'test_benchmark'
    benchmark = process(q['supervisor_pid'])
    controller = process(c['controller_pid'])
    assert benchmark and controller
    assert any(Path(arg).resolve() == (BENCH/'benchmark_queue.py').resolve()
               for arg in benchmark['command'].split())
    assert any(Path(arg).resolve() == (BASE/'continuation.py').resolve()
               for arg in controller['command'].split())
    assert benchmark['start_ticks'] == '188458853'
    assert controller['start_ticks'] == '181340748'
    assert not (PREPARED/'deployment_receipt.json').exists(), 'Inspect existing deployment before retry'

    # Freeze the waiting benchmark before stopping its parent monitor.
    os.kill(benchmark['pid'], signal.SIGSTOP)
    q = read(QUEUE)
    if q['status'] != 'waiting_resources' or any(j['status'] not in ('completed', 'pending') for j in q['jobs']):
        os.kill(benchmark['pid'], signal.SIGCONT)
        raise RuntimeError('A benchmark began before the safe replacement point')
    os.kill(controller['pid'], signal.SIGTERM)
    os.kill(benchmark['pid'], signal.SIGTERM)
    os.kill(benchmark['pid'], signal.SIGCONT)
    for _ in range(100):
        alive = [x for x in (process(controller['pid']), process(benchmark['pid']))
                 if x and x['state'] != 'Z']
        if not alive:
            break
        time.sleep(.1)
    assert not alive, 'Old controllers have not exited'

    authoritative = [BASE/'queue.json', BASE/'phase2/queue.json', BASE/'continuation_hashes.json',
                     BENCH/'selection_queue.json', QUEUE,
                     ROOT/'runs/component_search_h8001_20261020/queue_status.json']
    with contextlib.ExitStack() as stack:
        for filename in (BASE/'continuation.lock', BENCH/'benchmark_queue.lock'):
            lock = stack.enter_context(filename.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        documents = {path: read(path) for path in authoritative if path.exists()}
        union = {}
        for path, document in documents.items():
            guard = document if path.name == 'continuation_hashes.json' else document['prepared_hashes']
            for name, want in guard.items():
                assert name not in union or union[name] == want, 'Conflicting frozen hashes'
                union[name] = want
        for name, want in union.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == want, 'Protected file drift: '+name
        original_jobs = [job for job in q['jobs'] if job['status'] == 'completed']
        for job in original_jobs:
            assert job['returncodes'] == [0]*8
        audit = PREPARED/'original_controllers_and_guards'
        audit.mkdir()
        paths = list(documents)+[BENCH/'benchmark_queue.py']
        archive = []
        for index, path in enumerate(paths):
            backup = audit/f'{index:02d}_{path.name}'
            backup.write_bytes(path.read_bytes())
            archive.append(dict(original=str(path), backup=str(backup),
                                sha256=hashlib.sha256(backup.read_bytes()).hexdigest()))
        (audit/'inventory.json').write_text(json.dumps(archive, indent=2)+'\n')
        source = BENCH/'benchmark_queue.py'
        helper = BENCH/'available_gpu_scheduler.py'
        candidate = PREPARED/'benchmark_queue.candidate.py'
        compile(candidate.read_text(), str(source), 'exec')
        compile((PREPARED/'available_gpu_scheduler.py').read_text(), str(helper), 'exec')
        source.write_bytes(candidate.read_bytes())
        helper.write_bytes((PREPARED/'available_gpu_scheduler.py').read_bytes())
        new_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
        for path, document in documents.items():
            guard = document if path.name == 'continuation_hashes.json' else document['prepared_hashes']
            changed = False
            for name in list(guard):
                if Path(name).resolve() == source.resolve():
                    guard[name] = new_source_hash
                    changed = True
            assert changed, 'Scheduler was not protected in '+str(path)
            guard[str(helper)] = helper_hash
            write(path, document)
        for name, want in union.items():
            if Path(name).resolve() != source.resolve():
                assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == want
        after = read(QUEUE)
        assert [j for j in after['jobs'] if j['status'] == 'completed'] == original_jobs
        receipt = dict(status='installed', authorized_change='Use idle GPUs instead of requiring all eight',
                       at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       old_benchmark=benchmark, old_controller=controller,
                       scheduler_sha256=new_source_hash, helper_sha256=helper_hash,
                       protected_files_before=len(union), protected_files_after=len(union)+1,
                       completed_jobs_preserved=len(original_jobs),
                       unchanged='Training snapshot, recipes, weights, harness, panels, seeds, logical 8 shards',
                       backups=str(audit))
        write(PREPARED/'deployment_receipt.json', receipt)

    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    with (BASE/'new_models_test.log').open('ab', buffering=0) as log:
        benchmark_child = subprocess.Popen([q['python'], str(BENCH/'benchmark_queue.py'), '--state', str(QUEUE)],
            cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(100):
        updated = read(QUEUE)
        if updated.get('supervisor_pid') == benchmark_child.pid:
            break
        assert benchmark_child.poll() is None, 'Replacement benchmark exited during attachment'
        time.sleep(.1)
    assert updated.get('supervisor_pid') == benchmark_child.pid
    with (BASE/'continuation.log').open('ab', buffering=0) as log:
        controller_child = subprocess.Popen([q['python'], str(BASE/'continuation.py')],
            cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(status='dispatched', benchmark_supervisor=process(benchmark_child.pid),
                   continuation_controller=process(controller_child.pid))
    write(PREPARED/'deployment_receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
