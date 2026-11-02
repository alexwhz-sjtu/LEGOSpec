"""Loong text prompts: all source documents in listed order, without length trimming.

Derived from python_adapter_template.py. Set LOONG_DOC_ROOT to the extracted doc/
directory. Match the official document wrappers and legal prediction input rule;
disable the official optional shuffling and document-dropping length guard.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import fnmatch
import json
import os
from pathlib import Path
import re
from typing import Any


def document_root() -> Path:
    value = os.environ.get('LOONG_DOC_ROOT')
    if not value:
        raise ValueError('Set LOONG_DOC_ROOT to the extracted Loong doc directory')
    root = Path(value).resolve()
    if not root.is_dir():
        raise ValueError(f'Loong document directory does not exist: {root}')
    return root


def decoded_archive_name(name: str) -> str:
    """Recover ZIP UTF-8 names stored without the UTF-8 flag; leave files intact."""
    try:
        return name.encode('cp437').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


@lru_cache(maxsize=4)
def financial_files(root: Path) -> tuple[tuple[Path, str], ...]:
    # The official loader uses glob(...)[0]. Preserve extracted directory order
    # rather than silently changing ambiguous references through alphabetic sort.
    return tuple((p, decoded_archive_name(p.name))
                 for p in (root/'financial').glob('*.txt'))


@lru_cache(maxsize=4)
def legal_records(root: Path) -> dict[str, Any]:
    return json.loads((root/'legal/legal.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=4)
def financial_resolution(path: str) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding='utf-8'))['mapping']


def document_content(row: Mapping[str, Any], name: str, index: int) -> str:
    root = document_root()
    kind = row['type']
    if kind == 'paper':
        path = (root/'paper'/name).resolve()
        if not path.is_relative_to(root/'paper'):
            raise ValueError('Paper document path leaves the document directory')
        content = path.read_text(encoding='utf-8')
        title = content.split('\n', 1)[0].strip('#').strip()
        return f'{title}\n{content}\n\n'
    if kind == 'financial':
        pattern = f'*2024-{name}*.txt' if int(row['level']) != 4 else f'*{name}*.txt'
        matches = [(p, title) for p, title in financial_files(root)
                   if fnmatch.fnmatchcase(title, pattern)]
        if not matches:
            raise ValueError(f'No financial document for {name!r}')
        pinned = os.environ.get('LOONG_FINANCIAL_MAP')
        if pinned:
            selected = (root/financial_resolution(pinned)[str(row['level'])+':'+name]).resolve()
            matches = [(p,title) for p,title in matches if p.resolve()==selected]
            if len(matches)!=1:
                raise ValueError(f'Pinned financial document mismatch: {name!r}')
        path, title = matches[0]
        title = Path(title).stem.split('-')[-1]
        return f'《{title}》\n' + path.read_text(encoding='utf-8') + '\n\n'
    if kind == 'legal':
        record = legal_records(root)[name]
        content = record['content']
        prediction = int(row['level']) == 4 and '阅读以上判决文书，我将给你若干份判决结果：' in row['instruction']
        if not prediction:
            content += record['result']
        return f'《判决文书{index + 1}》\n{content}\n\n'
    raise ValueError(f'Unsupported Loong document type: {kind!r}')


def prepare_prompt(row: Mapping[str, Any]) -> str:
    names = row['doc']
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ValueError('Loong doc must be a nonempty list of document names')
    template = row['prompt_template']
    if not isinstance(template, str) or '{docs}' not in template or '{instruction}' not in template:
        raise ValueError('Loong prompt template must include docs and instruction')
    values = {'docs': ''.join(document_content(row, name, i) for i, name in enumerate(names)),
              'instruction': row['instruction'], 'question': row['question']}
    if not all(isinstance(v, str) for v in values.values()):
        raise ValueError('Loong template variables must contain text')
    if values['question'] and '{question}' not in template:
        raise ValueError('Loong prompt template would omit a nonempty question')
    # Replace only the three declared placeholders, never evaluate JSON braces
    # or treat instructions inside records as executable code.
    return re.sub(r'\{(docs|instruction|question)\}', lambda match: values[match[1]], template)


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    return {'prompt': prepare_prompt(row), 'source': 'iic/Loong', 'category': row['type']}
