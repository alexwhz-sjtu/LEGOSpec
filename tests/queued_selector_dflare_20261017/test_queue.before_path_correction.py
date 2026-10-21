import copy,json,tempfile,unittest
from pathlib import Path
from supervisor import validate_completed_metadata,verify_configuration,process_matches

class QueueGuards(unittest.TestCase):
    def metadata(self):
        return {'cursor':{'phase_index':1,'optimizer_step':59724,'phase_step':0,'next_batch':0},'phase_plan':[{'epochs':6}],'optimizer_step':59724,'scheduler_total_steps':59724,'distributed_training_state':{'world_size':8}}
    def test_complete_source_is_accepted(self):
        self.assertEqual(validate_completed_metadata(self.metadata())['optimizer_step'],59724)
    def test_incomplete_phase_is_rejected(self):
        m=self.metadata();m['cursor']['phase_index']=0
        with self.assertRaises(AssertionError):validate_completed_metadata(m)
    def test_premature_exit_is_rejected(self):
        m=self.metadata();m['optimizer_step']=100
        with self.assertRaises(AssertionError):validate_completed_metadata(m)
    def test_wrong_world_size_is_rejected(self):
        m=self.metadata();m['distributed_training_state']['world_size']=1
        with self.assertRaises(AssertionError):validate_completed_metadata(m)
    def test_prepared_recipe_has_exactly_the_authorized_change(self):
        s=json.loads(Path(__file__).with_name('queue.json').read_text());verify_configuration(s)
    def test_changed_config_hash_is_rejected(self):
        s=json.loads(Path(__file__).with_name('queue.json').read_text());s['prepared_hashes'][s['config']]='0'*64
        with self.assertRaises(AssertionError):verify_configuration(s)
    def test_source_process_identity_is_checked(self):
        s=json.loads(Path(__file__).with_name('queue.json').read_text())
        self.assertTrue(process_matches(s['source_pid'],s['source_start_ticks']))
        self.assertFalse(process_matches(s['source_pid'],'0'))

if __name__=='__main__':unittest.main(verbosity=2)

