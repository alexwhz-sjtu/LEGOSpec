"""Direct final tar shards, committed per rank and indexed by logical sample ID."""
from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.data import prepare_hidden_states as legacy

FORMAT = 'tar-indexed-v2'


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


class IndexedTarSink:
    """One owner streams samples; only sidecar-published shards are committed."""
    def __init__(self, args, root, identity, rank, metrics):
        if args.compress:
            raise ValueError('Indexed tar requires uncompressed seekable archives')
        self.args, self.root, self.identity = args, Path(root), identity
        self.rank, self.metrics = rank, metrics
        self.directory = self.root / 'ranks' / f'rank_{rank:05d}'
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.root / 'shards').mkdir(exist_ok=True)
        for directory in (self.directory, self.directory.parent, self.root / 'shards', self.root):
            fsync_directory(directory)
        self.done = set()
        self.seen = set()
        self.sequence = 0
        self.writer = None
        self.members, self.ranges = [], []
        for path in sorted(self.directory.glob('part_*.json')):
            meta = json.loads(path.read_text())
            if not args.resume or meta['input_identity'] != identity:
                raise ValueError('Committed tar belongs to another capture')
            data = self.root / meta['path']
            if legacy.sha256_file(data) != meta['sha256']:
                raise ValueError('Committed tar checksum mismatch')
            if self.done.intersection(meta['source_indices']):
                raise ValueError('Duplicate committed tar source IDs')
            self.done.update(meta['source_indices'])
            self.sequence = max(self.sequence, meta['sequence'] + 1)
        self.seen.update(self.done)
        if args.resume:
            for path in (self.root / 'shards').glob(f'.rank_{rank:05d}_part_*.tar.tmp'):
                path.unlink()

    def add(self, index, value, metadata):
        if index in self.seen:
            raise ValueError(f'Duplicate source ID {index}')
        if self.writer is not None:
            cap = self.args.tar_shard_bytes
            if (cap and self.writer.archive.offset + 512 + len(value) > cap) or (
                not cap and self.writer.sample_count >= self.args.shard_size
            ):
                self.commit()
        if self.writer is None:
            self.writer = legacy.TarShardWriter(
                self.root / 'shards', f'rank_{self.rank:05d}_part', self.sequence,
                False, 'final_rank', self.identity)
            self.writer.archive.copybufsize = self.args.tar_buffer_bytes
        start = self.writer.archive.offset
        self.writer.add_sample(index, value, sample_metadata=metadata)
        # Names are fixed ASCII shorter than the USTAR name field: one 512-byte header.
        assert self.writer.archive.offset == start + 512 + ((len(value) + 511) // 512) * 512
        self.members.append(f'samples/{index:012d}.pt')
        self.ranges.append([start + 512, len(value)])
        self.seen.add(index)

    def commit(self):
        writer = self.writer
        if writer is None:
            return
        meta = {'schema_version': 2, 'kind': 'final_rank', 'rank': self.rank,
                'sequence': self.sequence, 'input_identity': self.identity,
                'path': str(writer.final_path.relative_to(self.root)),
                'sample_count': writer.sample_count, 'source_indices': writer.source_indices,
                'sample_metadata': writer.sample_metadata, 'member_names': self.members,
                'member_ranges': self.ranges}
        with self.metrics.time('shard_commit'):
            legacy.add_bytes(writer.archive, legacy.METADATA_MEMBER,
                             json.dumps(meta, sort_keys=True).encode())
            writer.archive.close()
            with self.metrics.time('fsync'):
                with writer.temp_path.open('rb') as handle:
                    os.fsync(handle.fileno())
            meta['sha256'] = legacy.sha256_file(writer.temp_path)
            os.replace(writer.temp_path, writer.final_path)
            fsync_directory(writer.final_path.parent)
            durable_json(self.directory / f'part_{self.sequence:08d}.json', meta)
        self.done.update(meta['source_indices'])
        self.sequence += 1
        self.writer = None
        self.members, self.ranges = [], []

    def finish(self):
        self.commit()


def finalize(root, identity, metadata, count):
    root = Path(root)
    shards, indices = [], []
    ordinal = 0
    for path in sorted((root / 'ranks').glob('rank_*/part_*.json')):
        shard = json.loads(path.read_text())
        assert shard['input_identity'] == identity
        assert (root / shard['path']).is_file()
        size = shard['sample_count']
        assert size == len(shard['source_indices']) == len(shard['member_names']) == len(shard['member_ranges'])
        shard['success_start'], shard['success_end'] = ordinal, ordinal + size - 1
        ordinal += size
        indices.extend(shard['source_indices'])
        shards.append(shard)
    assert len(indices) == count and sorted(indices) == list(range(count)), 'Incomplete or duplicate source coverage'
    manifest = {**metadata, 'schema_version': 2, 'cache_format': FORMAT,
                'logical_order': 'source_index', 'complete': True, 'input_identity': identity,
                'successful_samples': count, 'failed_samples': 0, 'shard_size': None,
                'shards': shards}
    durable_json(root / 'manifest.json', manifest)
    return manifest
