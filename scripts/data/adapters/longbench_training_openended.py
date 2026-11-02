"""Training prompts without brevity constraints; original evaluation templates stay reproducible.

Keep source evidence and demonstrations intact. Remove only the current few-shot
query's empty terminal completion cue, in addition to explicit task choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

QUESTION_TASKS = {
    'narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh', 'hotpotqa',
    '2wikimqa', 'musique', 'dureader', 'qmsum', 'triviaqa',
}
LINE_LABEL = re.compile(r'(?m)^[ \t]*(?:\(([A-Ha-h])\)|（([A-Ha-h])）|([A-Ha-h])[.)、．:：])[ \t]+')
INLINE_LABEL = re.compile(r'(?<!\S)(?:\(([A-Ha-h])\)|（([A-Ha-h])）|([A-Ha-h])[.)、．:：])[ \t]+')
OPTION_HEADER = re.compile(r'(?im)(?:^|\n)[ \t]*(?:options?|choices?|选项|备选答案)[ \t]*[:：][ \t]*$')


@lru_cache(maxsize=1)
def templates() -> dict[str, str]:
    path = Path(__file__).with_name('longbench_training_templates.json')
    return json.loads(path.read_text(encoding='utf-8'))


def clean_input(task: str, text: str) -> tuple[str, dict[str, Any] | None]:
    """Strip only a question's trailing, sequential A/B/... choice block."""
    if task not in QUESTION_TASKS:
        return text, None
    offset = 0
    if task == 'triviaqa':
        # Its input includes an evidence passage; A/B lists inside that passage
        # are source material rather than choices for the final question.
        marker = '\nQuestion:\n'
        if marker not in text:
            return text, None
        offset = text.rfind(marker) + len(marker)
    question = text[offset:]
    for pattern in [LINE_LABEL, INLINE_LABEL]:
        matches = list(pattern.finditer(question))
        labels = [next(value for value in match.groups() if value).upper() for match in matches]
        if len(labels) < 2 or labels != list('ABCDEFGH'[:len(labels)]):
            continue
        prefix = question[:matches[0].start()].rstrip()
        header = OPTION_HEADER.search(prefix)
        if header:
            prefix = prefix[:header.start()].rstrip()
        elif not prefix.endswith(('?', '？')):
            continue
        if not prefix:
            raise ValueError('Removing options would leave an empty question')
        return text[:offset] + prefix, dict(choices=len(labels), removed_text=question[matches[0].start():])
    return text, None


COMPLETION_CUES = {
    'trec': 'Type', 'triviaqa': 'Answer', 'samsum': 'Summary', 'lsht': '类别',
}


def clean_completion_cue(task: str, text: str) -> tuple[str, bool]:
    cue = COMPLETION_CUES.get(task)
    if cue is None:
        return text, False
    pattern = r'(?i)\n[ \t]*' + re.escape(cue) + r'[ \t]*[:：][ \t\r\n]*\Z'
    match = re.search(pattern, text)
    if match is None:
        return text, False
    cleaned = text[:match.start()].rstrip()
    if not cleaned:
        raise ValueError('Removing completion cue would leave no current task')
    return cleaned, True


def prepare_prompt(row: Mapping[str, Any]) -> tuple[str, dict[str, Any] | None]:
    task = row['dataset']
    template = templates()[task]
    context, question = row['context'], row['input']
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError('LongBench context and input must be strings')
    if question and '{input}' not in template:
        raise ValueError(f'Official template for {task} would omit a nonempty input')
    question, removed = clean_input(task, question)
    question, _ = clean_completion_cue(task, question)
    return template.format(context=context, input=question), removed


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Render the training task using context and input, without answer metadata."""
    prompt, _ = prepare_prompt(row)
    return {
        "prompt": prompt,
        "source": 'zai-org/LongBench',
        "category": row['dataset'],
    }
