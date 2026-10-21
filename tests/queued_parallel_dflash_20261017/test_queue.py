import copy,json,tempfile,unittest,os
from pathlib import Path
from supervisor import validate_completed_metadata,verify_configuration,process_matches,resolve_training_output

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
        pid=os.getpid()
        ticks=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
        self.assertTrue(process_matches(pid,ticks))
        self.assertFalse(process_matches(pid,'0'))

    def test_suffixed_output_directory_is_resolved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);log=root/'train.log'
            log.write_text('Training output directory: '+str(root/'checkpoints_1')+'\n')
            self.assertEqual(resolve_training_output(root,log),root/'checkpoints_1')
    def test_output_directory_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);log=root/'train.log'
            log.write_text('Training output directory: /tmp/other-run/checkpoints\n')
            with self.assertRaises(AssertionError):resolve_training_output(root,log)

if __name__=='__main__':unittest.main(verbosity=2)

