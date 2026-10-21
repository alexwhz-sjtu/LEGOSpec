"""JSONL uses LF record boundaries; Unicode line separators belong to strings."""
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)
