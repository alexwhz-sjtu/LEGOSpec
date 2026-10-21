"""Run three authorized base-logit objectives after the preceding four-run queue."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
import traceback

BASE = Path(__file__).resolve().parent
STATE = BASE / 'queue.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_matches(pid, ticks=None):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and (ticks is None or fields[19] == str(ticks))
    except (FileNotFoundError, ProcessLookupError):
        return False


def workers(pid):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if fields[0] != 'Z' and fields[1] == str(pid) and b'scripts.composable.train' in args:
                found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return sorted(found)


def save(state, event=None):
    state['updated_at'] = now()
    body = json.dumps(state, indent=2) + '\n'
    for p in [STATE, Path(state['series_dir']) / 'queue_status.json']:
        temp = p.with_suffix('.tmp')
        temp.write_text(body)
        temp.replace(p)
    if event:
        print(json.dumps({'at': now(), 'event': event, 'status': state['status'],
                          'cursor': state['cursor']}), flush=True)


def verify_hashes(state):
    for name, expected in state['prepared_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'Prepared file changed: ' + name


def validate_order(state, index):
    assert state['cursor'] == index
    assert all(j['status'] == 'completed' for j in state['jobs'][:index]), 'Previous experiment is not complete'
    assert all(j['status'] == 'pending' for j in state['jobs'][index + 1:]), 'A later experiment already started'


def resources_free():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        return False
    try:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 29501))
    except OSError:
        return False
    return True


def assert_no_duplicate(job):
    targets = {job['config'].encode(), job['draft_config'].encode()}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = set((p / 'cmdline').read_bytes().split(b'\0'))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert not (targets & args), 'Existing training uses this experiment: ' + p.name


def output_dir(job, required=False):
    log = Path(job['log'])
    if log.exists():
        with log.open(errors='replace') as f:
            for line in f:
                match = re.search(r'Training output directory: ([^\r\n]+)', line)
                if match:
                    p = Path(match.group(1).strip()).resolve()
                    assert p.parent == Path(job['run_dir']).resolve(), 'Output escaped its run directory'
                    return p
    if required:
        raise AssertionError('Training did not record its actual checkpoint directory')
    return None


def metrics(job):
    p = Path(job['log'])
    if not p.exists():
        return []
    with p.open('rb') as f:
        f.seek(max(0, p.stat().st_size - 500000))
        text = f.read().decode(errors='replace')
    entries = re.findall(r'composable optimizer_step=(\d+) micro_step=(\d+) loss=([^\s]+)', text)
    return [{'step': int(s), 'loss': float(loss)} for s, _, loss in entries[-3:]]


def validate_metadata(meta, job):
    cursor = meta['cursor']
    phases = meta['phase_plan']
    assert phases and cursor['phase_index'] == len(phases), 'Not all phases completed'
    assert all(p['epochs'] == 6 and p.get('max_batches') is None for p in phases), 'Unexpected epoch schedule'
    assert meta['optimizer_step'] == meta['scheduler_total_steps'] == job['expected_steps'], 'Incomplete optimizer steps'
    assert cursor['optimizer_step'] == meta['optimizer_step']
    assert cursor['phase_step'] == 0 and cursor['next_batch'] == 0
    assert meta['distributed_training_state']['world_size'] == 8
    return {'optimizer_step': meta['optimizer_step'], 'phases': len(phases)}


def validate_final(job):
    directory = output_dir(job, required=True) / 'final'
    names = ['config.json', 'model.safetensors', 'training_state.pt', 'training_config.yaml']
    names += [f'training_state_rank_{i:05d}.pt' for i in range(8)]
    for name in names:
        p = directory / name
        assert p.is_file() and p.stat().st_size > 0, 'Missing final checkpoint file: ' + str(p)
    weights = directory / 'model.safetensors'
    with weights.open('rb') as f:
        size = int.from_bytes(f.read(8), 'little')
        assert 0 < size < weights.stat().st_size
        header = json.loads(f.read(size))
    end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
    assert weights.stat().st_size == 8 + size + end, 'Truncated model.safetensors'
    expected = json.loads(Path(job['draft_config']).read_text())
    actual = json.loads((directory / 'config.json').read_text())
    assert actual['composable_config'] == expected['composable_config'], 'Final model recipe changed'
    import torch
    meta = torch.load(directory / 'training_state.pt', map_location='cpu', weights_only=True)
    result = validate_metadata(meta, job)
    return str(directory), result


def launch(state, job):
    verify_hashes(state)
    assert_no_duplicate(job)
    assert not (Path(job['run_dir']) / 'checkpoints').exists(), 'Do not pre-create checkpoint directories'
    # An ambiguous crash during spawning requires inspection, never a second spawn.
    claim = Path(job['run_dir']) / 'launch_claim.json'
    with claim.open('x') as f:
        json.dump({'at': now(), 'config': job['config']}, f)
    job['status'] = 'launching'
    state['status'] = 'launching'
    save(state, 'launching_experiment_' + str(job['number']))
    env = os.environ.copy()
    env.update(WANDB_MODE='online', WANDB_ENTITY='alex-wang_sjtu-sjtu', WANDB_PROJECT='spec-decoding',
               WANDB_RUN_ID=job['wandb_id'], WANDB_RESUME='never', WANDB_RUN_GROUP=state['series_name'],
               WANDB_DIR=job['run_dir'], LITESPEC_RUN_DIR=job['run_dir'], PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=state['snapshot'], PYTHON_BIN=state['python'])
    with Path(job['log']).open('ab', buffering=0) as log:
        p = subprocess.Popen(['bash', 'scripts/composable/run_training_composable.sh', job['config']],
                             cwd=state['snapshot'], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = (Path('/proc') / str(p.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    job.update(status='running', pid=p.pid, start_ticks=ticks, started_at=now())
    state['status'] = 'running'
    save(state, 'started_experiment_' + str(job['number']))
    return p


def run(state):
    wait_for_predecessor(state)
    run_preflight(state)
    for index in range(state['cursor'], len(state['jobs'])):
        validate_order(state, index)
        job = state['jobs'][index]
        child = None
        if job['status'] == 'pending':
            state['status'] = 'waiting_resources'
            save(state, 'waiting_for_resources')
            while not resources_free():
                time.sleep(30)
            child = launch(state, job)
        else:
            assert job['status'] == 'running' and job.get('pid'), 'Ambiguous launch state requires inspection'
            state['status'] = 'running'
            save(state, 'reattached_experiment_' + str(job['number']))
        while process_matches(job['pid'], job['start_ticks']):
            found = metrics(job)
            if found:
                assert all(math.isfinite(m['loss']) for m in found), 'Non-finite training loss'
                changed = found != job.get('latest_metrics')
                job['latest_metrics'] = found
                output = output_dir(job)
                if output:
                    job['checkpoint_dir'] = str(output)
                if len(found) >= 2 and found[-1]['step'] > found[-2]['step'] and not job.get('verified_at'):
                    active = workers(job['pid'])
                    if len(active) == 8:
                        job.update(verified_at=now(), verified_workers=active)
                        save(state, 'verified_experiment_' + str(job['number']))
                elif changed:
                    save(state)
            time.sleep(30)
        if child is not None:
            code = child.wait()
            job['returncode'] = code
            assert code == 0, 'Training exited with code ' + str(code)
        verify_hashes(state)
        final, completion = validate_final(job)
        found = metrics(job)
        assert found and found[-1]['step'] == job['expected_steps'], 'Final training log is incomplete'
        assert all(math.isfinite(m['loss']) for m in found)
        with Path(job['log']).open('rb') as f:
            f.seek(max(0, Path(job['log']).stat().st_size - 30000))
            tail = f.read().decode(errors='replace')
        assert 'Traceback (most recent call last)' not in tail, 'Training ended with a traceback'
        job.update(status='completed', final_checkpoint=final, completion=completion,
                   latest_metrics=found, completed_at=now())
        state['cursor'] = index + 1
        state['status'] = 'between_experiments'
        save(state, 'completed_experiment_' + str(job['number']))
    state['status'] = 'completed'
    state['completed_at'] = now()
    save(state, 'all_three_experiments_completed')


def wait_for_predecessor(state):
    if state.get('predecessor_verified_at'):
        return
    state['status'] = 'waiting_predecessor'
    save(state, 'waiting_for_previous_four_experiments')
    while True:
        verify_hashes(state)
        previous = json.loads(Path(state['predecessor_queue']).read_text())
        verify_hashes(previous)
        assert previous['status'] != 'failed', 'Preceding queue failed: ' + str(previous.get('error'))
        if previous['status'] == 'completed':
            assert len(previous['jobs']) == 4 and previous['cursor'] == 4
            for job in previous['jobs']:
                assert job['status'] == 'completed' and job.get('returncode') == 0, 'Predecessor exit not verified'
                validate_final(job)
                assert not process_matches(job['pid'], job['start_ticks'])
            state['predecessor_verified_at'] = now()
            save(state, 'previous_four_experiments_verified')
            return
        assert process_matches(previous['supervisor_pid']), 'Preceding queue supervisor stopped; inspect before continuing'
        time.sleep(30)


def run_preflight(state):
    if state.get('preflight_passed_at'):
        return
    state['status'] = 'waiting_resources'
    save(state, 'waiting_for_preflight_resources')
    while not resources_free():
        verify_hashes(state)
        time.sleep(30)
    verify_hashes(state)
    state['status'] = 'preflight'
    save(state, 'eight_gpu_preflight_started')
    report = BASE / 'gpu_preflight_report.json'
    assert not report.exists(), 'Preflight report already exists without verified state; inspect before retry'
    env = os.environ.copy()
    env.update(PYTHONPATH=state['snapshot'], PYTHONDONTWRITEBYTECODE='1',
               PREFLIGHT_REPORT=str(report), OMP_NUM_THREADS='2')
    with (BASE/'gpu_preflight.log').open('ab', buffering=0) as log:
        result = subprocess.run([state['python'], '-m', 'torch.distributed.run', '--standalone',
                                 '--nnodes=1', '--nproc_per_node=8', str(BASE/'gpu_preflight.py')],
                                cwd=state['snapshot'], env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, timeout=900)
    assert result.returncode == 0, 'Eight GPU preflight failed; see gpu_preflight.log'
    proof = json.loads(report.read_text())
    assert proof['passed'] and proof['world_size'] == 8 and len(proof['rows']) == 6
    (Path(state['series_dir'])/'gpu_preflight_report.json').write_text(report.read_text())
    state['preflight_passed_at'] = now()
    save(state, 'eight_gpu_preflight_passed')


def main():
    with (BASE / 'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Queue supervisor already active', flush=True)
            return
        state = json.loads(STATE.read_text())
        if state['status'] in ('failed', 'completed'):
            print('Queue is terminal: ' + state['status'], flush=True)
            return
        state['supervisor_pid'] = os.getpid()
        save(state, 'supervisor_attached')
        try:
            run(state)
        except Exception as error:
            state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            save(state, 'queue_failed')
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
