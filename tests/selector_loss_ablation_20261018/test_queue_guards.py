import json
from unittest.mock import patch
import pytest
import supervisor as q


def test_failed_predecessor_prevents_launch(tmp_path):
    p=tmp_path/'queue.json';p.write_text(json.dumps({'status':'failed','prepared_hashes':{},'error':'nonzero exit'}))
    s={'predecessor_queue':str(p),'prepared_hashes':{}}
    with patch.object(q,'save'),patch.object(q,'launch') as launch:
        with pytest.raises(AssertionError,match='Preceding queue failed'):q.wait_for_predecessor(s)
        launch.assert_not_called()


def test_incomplete_predecessor_cannot_be_treated_as_completed(tmp_path):
    p=tmp_path/'queue.json';p.write_text(json.dumps({'status':'completed','cursor':3,'jobs':[{}]*4,'prepared_hashes':{}}))
    with patch.object(q,'save'),patch.object(q,'validate_final') as validate:
        with pytest.raises(AssertionError):q.wait_for_predecessor({'predecessor_queue':str(p),'prepared_hashes':{}})
        validate.assert_not_called()


def test_live_predecessor_keeps_waiting_without_launch(tmp_path):
    p=tmp_path/'queue.json';p.write_text(json.dumps({'status':'running','supervisor_pid':123,'prepared_hashes':{}}))
    with patch.object(q,'save'),patch.object(q,'process_matches',return_value=True),patch.object(q.time,'sleep',side_effect=InterruptedError),patch.object(q,'launch') as launch:
        with pytest.raises(InterruptedError):q.wait_for_predecessor({'predecessor_queue':str(p),'prepared_hashes':{}})
        launch.assert_not_called()


def test_failed_preflight_blocks_training(tmp_path):
    with patch.object(q,'BASE',tmp_path),patch.object(q,'save'),patch.object(q,'verify_hashes'),patch.object(q,'resources_free',return_value=True),patch.object(q.subprocess,'run') as run:
        run.return_value.returncode=1
        with pytest.raises(AssertionError,match='preflight failed'):
            q.run_preflight({'python':'python','snapshot':str(tmp_path)})
