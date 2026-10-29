"""Capture timing and persistence without experiment-directory dependencies."""
from collections import defaultdict
from contextlib import contextmanager
import os
import functools
from pathlib import Path
import json
import threading
import time
import resource

BOOT = time.perf_counter()

class CaptureMetrics:
    def __init__(self, profiling=False):
        self.profiling = profiling
        self.cpu = defaultdict(float)
        self.counts = defaultdict(int)
        self.events = []
        self.lock = threading.Lock()
        self.first_batch = None
        self.last_batch = None
        self.capture_end = None
        self.tokens = 0
        self.padded_tokens = 0
        self.peak_pinned_bytes = 0
        self.raw_cpu=[]

    @contextmanager
    def time(self, key):
        start = time.perf_counter()
        trace=os.environ.get('OPT_NVTX')=='1'
        if trace:
            import torch
            torch.cuda.nvtx.range_push(key)
        try:
            yield
        finally:
            if trace:torch.cuda.nvtx.range_pop()
            with self.lock:
                elapsed=time.perf_counter()-start
                self.cpu[key] += elapsed
                self.counts[key] += 1
                if self.profiling:
                    self.raw_cpu.append((key,start,elapsed,threading.get_ident()))

    def wrap(self, obj, name, key):
        original = getattr(obj, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            with self.time(key):
                return original(*args, **kwargs)
        setattr(obj, name, wrapped)

    @contextmanager
    def gpu(self, key, stream=None):
        if not self.profiling:
            yield
            return
        import torch
        enqueued=time.perf_counter()
        first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        first.record(stream)
        try:
            yield
        finally:
            last.record(stream)
            with self.lock:
                self.events.append((key, first, last,enqueued))

    def batch(self, values):
        self.batch_counts(int(values['attention_mask'].sum()),values['input_ids'].numel())

    def batch_counts(self, valid, padded):
        now = time.perf_counter()
        if self.first_batch is None:
            self.first_batch = now
        self.last_batch = now
        self.tokens += valid
        self.padded_tokens += padded
        self.counts['batches']+=1
        if os.environ.get('OPT_PROFILE')=='1':
            import torch
            if self.counts['batches']==2:torch.cuda.cudart().cudaProfilerStart()
            if self.counts['batches']==10:torch.cuda.cudart().cudaProfilerStop()

    def finish(self, directory, extra=None):
        import torch
        gpu = defaultdict(float)
        raw_gpu=[]
        for key, first, last,enqueued in self.events:
            last.synchronize()
            elapsed=first.elapsed_time(last)/1000
            gpu[key] += elapsed
            raw_gpu.append((key,enqueued,elapsed))
        value = {
            'rank': int(os.environ.get('RANK', 0)), 'pid': os.getpid(),
            'wall_s': time.perf_counter()-BOOT,
            'cpu_stage_s': dict(self.cpu), 'stage_calls': dict(self.counts),
            'cuda_stage_s': dict(gpu), 'valid_tokens': self.tokens,
            'padded_tokens': self.padded_tokens,
            'first_batch_offset_s': None if self.first_batch is None else self.first_batch-BOOT,
            'first_batch_monotonic': self.first_batch,
            'capture_end_monotonic': self.capture_end or self.last_batch,
            'last_batch_offset_s': None if self.last_batch is None else self.last_batch-BOOT,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **(extra or {})}
        value['peak_pinned_bytes']=self.peak_pinned_bytes
        value['host_allocator']=torch.cuda.memory.host_memory_stats()
        value['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        value['runtime_backend']=getattr(self,'runtime_backend',{})
        directory.mkdir(parents=True, exist_ok=True)
        raw_path=directory/f'rank_{value["rank"]}_raw_timings.json'
        raw_path.write_text(json.dumps({'cpu_columns':['stage','monotonic_start_s','elapsed_s','thread_id'],
            'cuda_columns':['stage','monotonic_enqueue_s','cuda_event_elapsed_s'],
            'cpu':self.raw_cpu,'cuda':raw_gpu})+'\n')
        value['raw_timings']=str(raw_path)
        (directory/f'rank_{value["rank"]}_metrics.json').write_text(json.dumps(value, indent=2)+'\n')


def persist_cache(root, metrics):
    with metrics.time('fsync'):
        for path in root.rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp'):
                with path.open('rb') as handle: os.fsync(handle.fileno())
        for directory in [root/'shards',root]:
            if directory.exists():
                fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
                try: os.fsync(fd)
                finally: os.close(fd)
