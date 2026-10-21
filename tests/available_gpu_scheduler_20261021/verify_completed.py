"""Read formal outputs and update only the independent notification ledger."""
from pathlib import Path
import datetime
import hashlib
import json
import math

PROJECT = Path('/share/dai-sys/wanghanzhen/projects/DAC27/LiteSpec_exp')
BENCH = PROJECT/'tests/component_benchmark_h8001_20261020'
MANIFEST_SHA = 'bdfd20787753294352983d7712951c1b03dd5cc148a0f2ef746b235bd7ce1c79'


def read(path):
    return json.loads(Path(path).read_text())


def main(recheck_all=False):
    queue = read(BENCH/'new_models_test_queue.json')
    assert queue['status'] != 'failed', queue.get('error')
    ledger_path = BENCH/'new_models_test_notification_state.json'
    ledger = read(ledger_path)
    manifest = Path(queue['manifest'])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == MANIFEST_SHA
    expected = [json.loads(line)['id'] for line in manifest.open() if line.strip()]
    assert len(expected) == len(set(expected)) == 1056
    frozen = read(queue['manifest_metadata'])
    weight_cache, verified = {}, []
    for job in queue['jobs']:
        key = job['label']+'/T'+str(job['temperature'])
        if job['status'] != 'completed' or (not recheck_all and key in ledger.get('completed', [])):
            continue
        assert job['returncodes'] == [0]*8, key
        if job.get('shard_states'):
            assert set(job['shard_states']) == {str(i) for i in range(8)}
            assert all(s['status'] == 'completed' and s['returncode'] == 0
                       for s in job['shard_states'].values())
        output = Path(job['output'])
        summary = read(output/'summary.json')
        assert summary['status'] == 'complete' and not summary['smoke']
        assert not summary['missing'] and not summary['unexpected']
        identity = summary['identity']
        assert identity['manifest_sha256'] == MANIFEST_SHA and not identity['smoke']
        assert identity['temperature'] == job['temperature']
        assert identity['top_p'] == (1. if job['temperature'] == 0 else .95)
        assert identity['max_new_tokens'] == 512 and identity['shards'] == 8
        assert Path(identity['draft']).resolve() == Path(job['model']).resolve()
        assert Path(identity['target']).resolve() == Path(queue['target']).resolve()
        assert identity['target_hashes'] == frozen['target_hashes']
        model = Path(job['model'])
        for filename, want in identity['draft_hashes'].items():
            file = model/filename
            if str(file) not in weight_cache:
                weight_cache[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            assert weight_cache[str(file)] == want
        assert hashlib.sha256((model/'config.json').read_bytes()).hexdigest() == identity['config_sha256']
        for filename, want in identity['harness_hashes'].items():
            assert hashlib.sha256((BENCH/'harness'/filename).read_bytes()).hexdigest() == want
        all_ids = []
        for rank in range(8):
            assert read(output/f'metadata_{rank}.json') == identity
            done = read(output/f'complete_{rank}.json')
            assert done['status'] == 'completed'
            file = output/f'requests_{rank}.jsonl'
            assert file.read_bytes().endswith(b'\n')
            rows = [json.loads(line) for line in file.open() if line.strip()]
            assert len(rows) == done['requests']
            ids = []
            for row in rows:
                assert row['status'] == 'ok' and row['temperature'] == job['temperature']
                tau = row['tau']
                assert ((tau is None and row['complete_cycles'] == 0) or
                        (tau is not None and math.isfinite(tau) and 1 <= tau <= 8))
                ids.append(row['id'])
            assert len(ids) == len(set(ids))
            assert set(ids) == {value for index, value in enumerate(expected) if index % 8 == rank}
            all_ids.extend(ids)
        assert len(all_ids) == len(set(all_ids)) == 1056 and set(all_ids) == set(expected)
        entry = dict(key=key, requests=1056, shards=8, returncodes=job['returncodes'],
                     summary=str(output/'summary.json'),
                     verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        verified.append(entry)
        if not recheck_all:
            print('VERIFIED', json.dumps(entry), flush=True)
    if verified:
        ledger['completed'] = sorted(set(ledger.get('completed', [])) | {e['key'] for e in verified})
        ledger.setdefault('completion_evidence', {}).update({e['key']: e for e in verified})
        temp = ledger_path.with_name(ledger_path.name+'.verification.tmp')
        temp.write_text(json.dumps(ledger, indent=2)+'\n')
        temp.replace(ledger_path)
    print('VERIFIED_COUNT', len(ledger.get('completed', [])), 'OF', len(queue['jobs']))
    if recheck_all:
        print('RECHECKED_ALL', len(verified), 'REQUESTS', sum(e['requests'] for e in verified))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    main(parser.parse_args().all)
