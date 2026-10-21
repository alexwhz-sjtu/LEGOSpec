"""Schedule eight immutable logical benchmark shards on currently idle GPUs."""
from pathlib import Path
import datetime
import fcntl
import json
import os
import subprocess
import time


def available_gpus():
    inventory = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    applications = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    occupied = {line.strip() for line in applications.stdout.splitlines() if line.strip()}
    idle = []
    for line in inventory.stdout.splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(',')]
        # A GPU with any compute process belongs to that process, even at zero utilization.
        if uuid not in occupied and int(memory) <= 1024 and int(utilization) <= 5:
            idle.append({'index': int(index), 'uuid': uuid})
    return sorted(idle, key=lambda gpu: gpu['index'])


def process_ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def run_shards(base, state, job, save):
    """Keep shard IDs, seeds and outputs unchanged; alter only GPU placement/concurrency."""
    base, output = Path(base), Path(job['output'])
    locks = base / 'gpu_locks'
    locks.mkdir(exist_ok=True)
    pending = list(range(8))
    active, children = {}, {}
    job['gpu_scheduling'] = 'idle-gpus-eight-fixed-logical-shards-v1'
    job['shard_states'] = {}

    def stop_own_children():
        for process, lease, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, lease, _ in active.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            lease.close()

    try:
        while pending or active:
            changed = False
            for shard, (process, lease, gpu) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                job['shard_states'][str(shard)].update(status='completed' if code == 0 else 'failed',
                                                       returncode=code)
                lease.close()
                del active[shard]
                changed = True
                if code != 0:
                    save(state)
                    raise RuntimeError(f'Benchmark shard {shard} pid {process.pid} exited {code}')
            if changed:
                save(state)

            occupied_by_us = {gpu['uuid'] for _, _, gpu in active.values()}
            candidates = available_gpus() if pending else []
            for gpu in candidates:
                if not pending or gpu['uuid'] in occupied_by_us:
                    continue
                lease = (locks / (gpu['uuid'] + '.lock')).open('a')
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lease.close()
                    continue
                # Recheck after acquiring our advisory lock; never take another user's GPU.
                if gpu['uuid'] not in {item['uuid'] for item in available_gpus()}:
                    lease.close()
                    continue
                shard = pending.pop(0)
                command = [state['python'], str(base/'harness/run.py'), '--source', state['source'],
                           '--target', state['target'], '--draft', job['model'],
                           '--manifest', state['manifest'], '--output', job['output'],
                           '--temperature', str(job['temperature']), '--shard', str(shard),
                           '--shards', '8', '--device', 'cuda:0']
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'],
                                   OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                                   PYTHONPATH=state['source'], PYTHONDONTWRITEBYTECODE='1')
                try:
                    with (output/f'worker_{shard}.log').open('ab', buffering=0) as log:
                        process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                except BaseException:
                    lease.close()
                    raise
                active[shard] = process, lease, gpu
                children[shard] = process
                occupied_by_us.add(gpu['uuid'])
                receipt = dict(shard=shard, shards=8, gpu_index=gpu['index'], gpu_uuid=gpu['uuid'],
                               pid=process.pid, start_ticks=process_ticks(process.pid),
                               device='cuda:0', status='running',
                               launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                with (output/f'shard_launch_{shard}.json').open('x') as file:
                    json.dump(receipt, file, indent=2)
                job['shard_states'][str(shard)] = receipt
                job.update(status='running', pids=[children[index].pid for index in sorted(children)])
                save(state)
                print(json.dumps({'event': 'benchmark_shard_launched', **receipt}), flush=True)
            if pending or active:
                time.sleep(2)
        assert sorted(children) == list(range(8))
        return [children[index] for index in range(8)]
    except BaseException:
        stop_own_children()
        raise
