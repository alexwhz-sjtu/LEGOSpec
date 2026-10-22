"""Bounded direct tar writer with streaming SHA and asynchronous durable commit."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tarfile

from scripts.data import prepare_hidden_states as legacy
from .hidden_tar_raw import RawSample


def sync_dir(path, metrics):
    with metrics.time('commit_directory_fsync'):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def durable_json(path, value, metrics):
    temporary = path.with_name('.' + path.name + '.tmp')
    with metrics.time('commit_index_encode'):
        encoded = json.dumps(value, sort_keys=True).encode() + b'\n'
    with metrics.time('commit_index_write'):
        with temporary.open('wb') as handle:
            handle.write(encoded)
            handle.flush()
            with metrics.time('commit_index_fsync'):
                os.fsync(handle.fileno())
    with metrics.time('commit_index_rename'):
        os.replace(temporary, path)
    sync_dir(path.parent, metrics)


class StreamTar:
    def __init__(self, root, rank, sequence, args, metrics):
        self.metrics, self.args = metrics, args
        self.final = root / 'shards' / f'rank_{rank:05d}_part_{sequence:08d}.tar'
        self.temp = self.final.with_name('.' + self.final.name + '.tmp')
        self.handle = self.temp.open('wb', buffering=args.tar_buffer_bytes)
        self.digest = hashlib.sha256()
        self.offset = 0
        self.indices, self.metadata, self.names, self.ranges = [], [], [], []

    def write(self, value):
        value = memoryview(value).cast('B')
        for begin in range(0, len(value), self.args.tar_buffer_bytes):
            part = value[begin:begin + self.args.tar_buffer_bytes]
            with self.metrics.time('tar_write_bytes'):
                count = self.handle.write(part)
            if count != len(part):
                raise OSError('Short tar write')
            if self.args.tar_stream_sha:
                with self.metrics.time('sha256_stream'):
                    self.digest.update(part)
            self.offset += count

    def member(self, name, value):
        size = value.size if isinstance(value, RawSample) else len(value)
        header = tarfile.TarInfo(name)
        header.size, header.mtime = size, 0
        self.write(header.tobuf(format=tarfile.USTAR_FORMAT))
        start = self.offset
        if isinstance(value, RawSample):
            self.write(value.header)
            for part in value.parts:
                self.write(part)
        else:
            self.write(value)
        self.write(b'\0' * (-size % 512))
        return [start, size]

    def seal(self, meta):
        with self.metrics.time('commit_tar_footer'):
            self.member(legacy.METADATA_MEMBER, json.dumps(meta, sort_keys=True).encode())
            self.write(b'\0' * 1024)
            self.write(b'\0' * (-self.offset % 10240))

    def publish(self, meta, sidecar):
        try:
            with self.metrics.time('shard_commit'):
                with self.metrics.time('commit_flush'):
                    self.handle.flush()
                with self.metrics.time('commit_file_fsync'):
                    os.fsync(self.handle.fileno())
                with self.metrics.time('commit_close'):
                    self.handle.close()
                if self.args.tar_stream_sha:
                    meta['sha256'] = self.digest.hexdigest()
                else:
                    with self.metrics.time('sha256_readback'):
                        meta['sha256'] = legacy.sha256_file(self.temp)
                with self.metrics.time('commit_data_rename'):
                    os.replace(self.temp, self.final)
                sync_dir(self.final.parent, self.metrics)
                durable_json(sidecar, meta, self.metrics)
            return meta
        finally:
            self.handle.close()


class StreamTarSink:
    def __init__(self, args, root, identity, rank, metrics):
        self.args, self.root, self.identity, self.rank, self.metrics = args, Path(root), identity, rank, metrics
        self.directory = self.root / 'ranks' / f'rank_{rank:05d}'
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.root / 'shards').mkdir(exist_ok=True)
        for path in (self.directory, self.directory.parent, self.root / 'shards', self.root):
            sync_dir(path, metrics)
        self.done = set()
        self.sequence = 0
        for path in sorted(self.directory.glob('part_*.json')):
            meta = json.loads(path.read_text())
            if not args.resume or meta['input_identity'] != identity:
                raise ValueError('Committed tar identity mismatch')
            if legacy.sha256_file(self.root / meta['path']) != meta['sha256']:
                raise ValueError('Committed tar checksum mismatch')
            if self.done.intersection(meta['source_indices']):
                raise ValueError('Duplicate committed source ID')
            self.done.update(meta['source_indices'])
            self.sequence = max(self.sequence, meta['sequence'] + 1)
        if args.resume:
            for path in (self.root / 'shards').glob(f'.rank_{rank:05d}_part_*.tar.tmp'):
                path.unlink()
        self.seen = set(self.done)
        self.writer = None
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=args.tar_commit_workers) if args.tar_async_commit else None

    def reap(self, wait=False):
        for future in self.pending:
            if future.done():
                future.result()  # Surface any completed failure, including out-of-order completion.
        while self.pending and (wait or self.pending[0].done()):
            self.done.update(self.pending.popleft().result()['source_indices'])
            if not wait:
                continue

    def add(self, index, value, metadata):
        self.reap()
        if index in self.seen:
            raise ValueError('Duplicate source ID')
        size = value.size if isinstance(value, RawSample) else len(value)
        if self.writer is not None and self.writer.offset + 512 + size > self.args.tar_shard_bytes:
            self.commit()
        if self.writer is None:
            self.writer = StreamTar(self.root, self.rank, self.sequence, self.args, self.metrics)
        extension = 'bin' if isinstance(value, RawSample) else 'pt'
        name = f'samples/{index:012d}.{extension}'
        with self.metrics.time('tar_write'):
            span = self.writer.member(name, value)
        self.writer.indices.append(index)
        self.writer.metadata.append(metadata)
        self.writer.names.append(name)
        self.writer.ranges.append(span)
        self.seen.add(index)

    def commit(self):
        writer = self.writer
        if writer is None:
            return
        meta = {'schema_version': 2, 'kind': 'final_rank', 'rank': self.rank, 'sequence': self.sequence,
                'input_identity': self.identity, 'path': str(writer.final.relative_to(self.root)),
                'sample_count': len(writer.indices), 'source_indices': writer.indices,
                'sample_metadata': writer.metadata, 'member_names': writer.names, 'member_ranges': writer.ranges}
        writer.seal(meta)
        sidecar = self.directory / f'part_{self.sequence:08d}.json'
        self.writer = None
        self.sequence += 1
        if self.executor:
            try:
                with self.metrics.time('commit_queue_wait'):
                    while len(self.pending) >= self.args.tar_pending_commits:
                        self.done.update(self.pending.popleft().result()['source_indices'])
                    self.reap()
                self.pending.append(self.executor.submit(writer.publish, meta, sidecar))
            except BaseException:
                writer.handle.close()
                raise
        else:
            self.done.update(writer.publish(meta, sidecar)['source_indices'])

    def finish(self):
        try:
            self.commit()
            with self.metrics.time('commit_drain'):
                self.reap(wait=True)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)


def finalize(root, identity, metadata, count, metrics, raw=False):
    root = Path(root)
    shards, indices, ordinal = [], [], 0
    with metrics.time('final_index_read'):
        for path in sorted((root / 'ranks').glob('rank_*/part_*.json')):
            shard = json.loads(path.read_text())
            assert shard['input_identity'] == identity and (root / shard['path']).is_file()
            size = shard['sample_count']
            assert size == len(shard['source_indices']) == len(shard['member_names']) == len(shard['member_ranges'])
            shard['success_start'], shard['success_end'] = ordinal, ordinal + size - 1
            ordinal += size
            indices.extend(shard['source_indices'])
            shards.append(shard)
    assert len(indices) == count and sorted(indices) == list(range(count)), 'Incomplete or duplicate source coverage'
    manifest = {**metadata, 'schema_version': 2, 'cache_format': 'tar-raw-v1' if raw else 'tar-indexed-v2',
                'logical_order': 'source_index', 'complete': True, 'input_identity': identity,
                'successful_samples': count, 'failed_samples': 0, 'shard_size': None, 'shards': shards}
    durable_json(root / 'manifest.json', manifest, metrics)
    return manifest
