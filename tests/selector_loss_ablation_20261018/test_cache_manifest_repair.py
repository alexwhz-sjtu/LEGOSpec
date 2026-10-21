import hashlib
from pathlib import Path
import pytest
from recover_cache_manifest import retained_manifest


@pytest.mark.parametrize('changed_source', [None, 'model.py', 'config.json', 'supervisor.py'])
def test_cache_is_mutable_but_sources_and_config_remain_protected(tmp_path, changed_source):
    snapshot=tmp_path/'snapshot';cache=snapshot/'.cache/tvm-ffi';cache.mkdir(parents=True)
    paths=[snapshot/'model.py',tmp_path/'config.json',tmp_path/'supervisor.py',cache/'build.ninja']
    for p in paths:p.write_text('original')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (cache/'build.ninja').write_text('normal runtime update')
    if changed_source:
        p=next(p for p in paths if p.name==changed_source);p.write_text('unexpected source change')
        with pytest.raises(AssertionError,match='Non-cache drift'):
            retained_manifest({'snapshot':str(snapshot),'prepared_hashes':hashes})
    else:
        kept,removed=retained_manifest({'snapshot':str(snapshot),'prepared_hashes':hashes})
        assert len(kept)==3 and list(removed)==[str(cache/'build.ninja')]
        assert all(kept[k]==hashes[k] for k in kept)
