import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import supervisor as q


class SerialQueueGuards(unittest.TestCase):
    def metadata(self):
        return {'cursor': {'phase_index': 1, 'optimizer_step': 59724, 'phase_step': 0,
                           'next_batch': 0}, 'phase_plan': [{'epochs': 6, 'max_batches': None}],
                'optimizer_step': 59724, 'scheduler_total_steps': 59724,
                'distributed_training_state': {'world_size': 8}}

    def test_complete_schedule_is_accepted(self):
        self.assertEqual(q.validate_metadata(self.metadata(), {'expected_steps': 59724})['optimizer_step'], 59724)

    def test_early_exit_and_smoke_runs_are_rejected(self):
        for mutate in [lambda m: m.update(optimizer_step=100),
                       lambda m: m['cursor'].update(phase_index=0),
                       lambda m: m['phase_plan'][0].update(max_batches=20),
                       lambda m: m['distributed_training_state'].update(world_size=1)]:
            m = self.metadata()
            mutate(m)
            with self.assertRaises(AssertionError):
                q.validate_metadata(m, {'expected_steps': 59724})

    def test_experiments_cannot_be_reordered(self):
        state = {'cursor': 1, 'jobs': [{'status': 'completed'}, {'status': 'pending'},
                                     {'status': 'pending'}, {'status': 'pending'}]}
        q.validate_order(state, 1)
        state['jobs'][0]['status'] = 'running'
        with self.assertRaises(AssertionError):
            q.validate_order(state, 1)
        state['jobs'][0]['status'] = 'completed'
        state['jobs'][3]['status'] = 'running'
        with self.assertRaises(AssertionError):
            q.validate_order(state, 1)

    def test_snapshot_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.py'
            p.write_text('original')
            state = {'prepared_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}}
            q.verify_hashes(state)
            p.write_text('changed')
            with self.assertRaises(AssertionError):
                q.verify_hashes(state)

    def test_actual_output_suffix_is_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'train.log'
            p.write_text('Training output directory: ' + str(Path(d) / 'checkpoints_1') + '\n')
            self.assertEqual(q.output_dir({'log': str(p), 'run_dir': d}, required=True), Path(d) / 'checkpoints_1')

    def test_missing_output_or_foreign_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'train.log'
            p.write_text('no output record\n')
            with self.assertRaises(AssertionError):
                q.output_dir({'log': str(p), 'run_dir': d}, required=True)
            p.write_text('Training output directory: /tmp/another_run/checkpoints\n')
            with self.assertRaises(AssertionError):
                q.output_dir({'log': str(p), 'run_dir': d}, required=True)

    def test_existing_launch_claim_prevents_a_second_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'launch_claim.json').write_text('{}')
            job = {'run_dir': d, 'config': d + '/next.yaml', 'draft_config': d + '/next.json'}
            with patch.object(q.subprocess, 'Popen') as spawn:
                with self.assertRaises(FileExistsError):
                    q.launch({'prepared_hashes': {}}, job)
                spawn.assert_not_called()

    def test_process_identity_rejects_reused_pids(self):
        fields = (Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()
        self.assertTrue(q.process_matches(os.getpid(), fields[19]))
        self.assertFalse(q.process_matches(os.getpid(), '0'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
