"""Versioned raw tensor members inside ordinary uncompressed USTAR files.

Only the requested logical tensor bytes are written, never the backing storage.
The caller owns tensor buffers until write_member returns. No mmap lifetime leaks
into the Dataset: loaded tensors own a bytearray after the file read completes.
"""
from __future__ import annotations

import json
import math
import struct

import torch

MAGIC = b'LGRAW001'
DTYPES = {str(t): t for t in (torch.bfloat16, torch.float16, torch.float32,
          torch.float64, torch.int64, torch.int32, torch.int16, torch.int8,
          torch.uint8, torch.bool)}


class RawSample:
    def __init__(self, payload):
        self.parts = []
        descriptors = []
        offset = 0
        fields = [(key, payload[key]) for key in ('input_ids', 'attention_mask', 'loss_mask')]
        fields += [(f'hidden/{int(key)}', value) for key, value in sorted(payload['hidden_states'].items())]
        fields += [('final_hidden_state', payload['final_hidden_state'])]
        for name, tensor in fields:
            if tensor.device.type != 'cpu' or str(tensor.dtype) not in DTYPES:
                raise ValueError('Raw tar requires a supported CPU tensor')
            tensor = tensor.detach().contiguous()
            value = memoryview(tensor.view(torch.uint8).numpy()).cast('B')
            descriptors.append({'name': name, 'dtype': str(tensor.dtype),
                                'shape': list(tensor.shape), 'offset': offset, 'nbytes': len(value)})
            self.parts.append(value)
            offset += len(value)
        header = json.dumps({'version': 1, 'sample_index': payload['sample_index'],
                             'tensors': descriptors}, sort_keys=True, separators=(',', ':')).encode()
        self.header = MAGIC + struct.pack('<Q', len(header)) + header
        self.header += b'\0' * (-len(self.header) % 64)
        self.size = len(self.header) + offset


def load_raw(handle, start, size, *, headers_only=False):
    """Token audit reads only headers/tokens; normal loading reads all tensors."""
    handle.seek(start)
    prefix = handle.read(16)
    if len(prefix) != 16 or prefix[:8] != MAGIC:
        raise ValueError('Invalid raw tensor member magic')
    length = struct.unpack('<Q', prefix[8:])[0]
    if not 0 < length <= min(1024 * 1024, size - 16):
        raise ValueError('Invalid raw tensor header length')
    info = json.loads(handle.read(length))
    if info.get('version') != 1:
        raise ValueError('Unsupported raw tensor member version')
    data_start = (16 + length + 63) // 64 * 64
    output = {'schema_version': 2, 'sample_index': info['sample_index'], 'hidden_states': {}}
    names = set()
    previous_end = 0
    for desc in info['tensors']:
        name, dtype, shape = desc['name'], DTYPES[desc['dtype']], desc['shape']
        if name in names or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('Invalid raw tensor shape or duplicate field')
        names.add(name)
        offset, nbytes = desc['offset'], desc['nbytes']
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if type(offset) is not int or type(nbytes) is not int or offset != previous_end or nbytes != expected or data_start + offset + nbytes > size:
            raise ValueError('Invalid raw tensor byte range')
        previous_end = offset + nbytes
        if headers_only and (name.startswith('hidden/') or name == 'final_hidden_state'):
            value = torch.empty(shape, dtype=dtype, device='meta')
        else:
            handle.seek(start + data_start + offset)
            data = bytearray(nbytes)
            if handle.readinto(data) != nbytes:
                raise ValueError('Truncated raw tensor')
            value = torch.frombuffer(data, dtype=dtype).reshape(shape)
        if name.startswith('hidden/'):
            output['hidden_states'][int(name.split('/')[1])] = value
        elif name in ('input_ids', 'attention_mask', 'loss_mask', 'final_hidden_state'):
            output[name] = value
        else:
            raise ValueError('Unknown raw tensor field')
    if names != {'input_ids', 'attention_mask', 'loss_mask', 'final_hidden_state',
                 'hidden/1', 'hidden/9', 'hidden/17', 'hidden/25', 'hidden/33'}:
        raise ValueError('Missing raw tensor fields')
    if data_start + previous_end != size:
        raise ValueError('Trailing raw tensor bytes')
    return output
