"""Extract text user turns from Nemotron IFChat; the converter selects first/multi."""

from collections.abc import Mapping
from typing import Any


def extract(row: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude all source system prompts and assistant completions."""
    messages = row['messages']
    if not isinstance(messages, list):
        raise ValueError('messages must be an array')
    return {
        'messages': [
            {'role': 'user', 'content': message['content']}
            for message in messages
            if isinstance(message, Mapping) and message.get('role') == 'user'
        ],
        'source': 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v2',
        'category': None,
    }
