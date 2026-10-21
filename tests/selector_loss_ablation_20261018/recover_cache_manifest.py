"""User-authorized repair of runtime-cache hashes after a successful D-PACE exit."""
from pathlib import Path
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess

BASE = Path('/data/temp/LiteSpec_exp/tests/selector_loss_ablation_20261018')
spec = importlib.util.spec_from_file_location('queue_supervisor', BASE/'supervisor.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def retained_manifest(state):
    cache = Path(state['snapshot']) / '.cache'
    removed = {name: value for name, value in state['prepared_hashes'].items()
               if Path(name).is_relative_to(cache)}
    retained = {name: value for name, value in state['prepared_hashes'].items()
                if name not in removed}
    assert removed and retained
    for name, expected in retained.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'Non-cache drift: '+name
    return retained, removed


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup = BASE / ('cache_guard_repair_' + stamp)
    with (BASE/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        s = json.loads((BASE/'queue.json').read_text())
        before = copy.deepcopy(s)
        assert s['status'] == 'failed' and s['cursor'] == 0
        assert s['error'].startswith('AssertionError: Prepared file changed: '+s['snapshot']+'/.cache/')
        assert not q.process_matches(s['supervisor_pid'])
        j = s['jobs'][0]
        assert j['status'] == 'running' and j['returncode'] == 0
        assert not q.process_matches(j['pid'],j['start_ticks'])
        final, meta = q.validate_final(j)
        metrics = q.metrics(j)
        assert metrics[-1]['step'] == 59724 and all(math.isfinite(x['loss']) for x in metrics)
        proof = json.loads((BASE/'dpace_completion_before_repair.json').read_text())
        assert proof['final'] == final and proof['returncode'] == 0
        assert proof['wandb_state'] == 'finished' and proof['summary']['_step'] == 59724
        for pending in s['jobs'][1:]:
            assert pending['status'] == 'pending' and not pending.get('pid')
            assert not (Path(pending['run_dir'])/'launch_claim.json').exists()
            q.assert_no_duplicate(pending)
        retained, removed = retained_manifest(s)
        backup.mkdir()
        (backup/'queue_before.json').write_text(json.dumps(before,indent=2))
        (backup/'removed_runtime_cache_hashes.json').write_text(json.dumps(removed,indent=2))
        shutil.copy2(BASE/'supervisor.py', backup/'supervisor_unchanged.py')
        s['prepared_hashes'] = retained
        s['status'] = 'running'
        s.pop('error')
        s['manifest_policy'] = 'Frozen source/configuration/helpers only; snapshot/.cache is mutable runtime cache'
        s['cache_guard_repaired_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        q.verify_hashes(s)
        # Leave job 1 and cursor intact: the existing supervisor validates its
        # recorded successful exit and final before advancing; it never relaunches it.
        assert s['jobs'] == before['jobs'] and s['cursor'] == before['cursor']
        q.save(s,'user_authorized_runtime_cache_manifest_repair')
        audit={'at':s['cache_guard_repaired_at'],'retained_hash_count':len(retained),
               'removed_cache_hash_count':len(removed),'all_retained_hashes_unchanged':True,
               'training_code_configs_and_supervisor_unchanged':True,
               'completed_job_pid':j['pid'],'completed_job_returncode':0,'completed_final':final,
               'backup':str(backup)}
        (backup/'repair_audit.json').write_text(json.dumps(audit,indent=2))
        cloud=Path(s['series_dir'])/backup.name
        shutil.copytree(backup,cloud)
    with (BASE/'supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([s['python'],str(BASE/'supervisor.py')],cwd=BASE,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=os.environ.copy())
    ticks=(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    audit.update(new_supervisor_pid=p.pid,new_supervisor_start_ticks=ticks)
    (BASE/'cache_guard_recovery_receipt.json').write_text(json.dumps(audit,indent=2))
    (Path(s['series_dir'])/'cache_guard_recovery_receipt.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':main()
