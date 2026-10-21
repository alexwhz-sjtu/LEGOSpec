"""CPU-only checks for occupied-GPU exclusion and fixed-shard scheduling."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('idle_scheduler', Path(__file__).with_name('available_gpu_scheduler.py'))
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class SchedulerTests(unittest.TestCase):
    def test_busy_gpu_exclusion_even_when_utilization_zero(self):
        inventory = '0, GPU-a, 4, 0\n1, GPU-b, 68999, 0\n2, GPU-c, 4, 0\n3, GPU-d, 4, 65\n'
        responses = [subprocess.CompletedProcess([], 0, inventory),
                     subprocess.CompletedProcess([], 0, 'GPU-b\nGPU-c\n')]
        with patch.object(scheduler.subprocess, 'run', side_effect=responses):
            self.assertEqual(scheduler.available_gpus(), [{'index': 0, 'uuid': 'GPU-a'}])

    def simulate(self, first_exit=0):
        launches, live = [], {}

        class Child:
            def __init__(self, command, **kwargs):
                self.shard = int(command[command.index('--shard')+1])
                self.gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
                self.pid = 900000 + self.shard
                self.returncode = None
                self.polls = 0
                if self.gpu in live:
                    raise AssertionError('Two active workers assigned the same GPU')
                live[self.gpu] = self
                launches.append((self.shard, self.gpu, command))

            def poll(self):
                self.polls += 1
                if self.returncode is None and self.polls >= 3:
                    self.returncode = first_exit if self.shard == 0 else 0
                    live.pop(self.gpu, None)
                return self.returncode

            def terminate(self):
                self.returncode = -15
                live.pop(self.gpu, None)

            def wait(self, timeout=None):
                return self.returncode

            kill = terminate

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            base = Path(directory)
            output = base/'output'
            output.mkdir()
            state = dict(python='python', source='frozen', target='target', manifest='manifest')
            job = dict(output=str(output), model='model', temperature=1)
            gpus = [{'index': 0, 'uuid': 'GPU-a'}, {'index': 2, 'uuid': 'GPU-c'}]
            with patch.object(scheduler, 'available_gpus', return_value=gpus), \
                 patch.object(scheduler.subprocess, 'Popen', Child), \
                 patch.object(scheduler, 'process_ticks', return_value='fixed'), \
                 patch.object(scheduler.time, 'sleep'):
                scheduler.run_shards(base, state, job, lambda _: None)
            self.assertEqual([item[0] for item in launches], list(range(8)))
            self.assertEqual({item[1] for item in launches}, {'GPU-a', 'GPU-c'})
            for _, _, command in launches:
                self.assertEqual(command[command.index('--shards')+1], '8')
                self.assertEqual(command[command.index('--device')+1], 'cuda:0')
            self.assertFalse(live)

    def test_eight_shards_complete_once_using_only_two_idle_gpus(self):
        self.simulate()

    def test_worker_failure_stops_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, 'exited 7'):
            self.simulate(first_exit=7)


if __name__ == '__main__':
    unittest.main()
