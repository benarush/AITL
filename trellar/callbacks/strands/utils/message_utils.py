from typing import Any, Optional

from ..._common import compact_json


def _text_of_blocks(blocks: Any) -> str:
    """Join the text / json content blocks of a Strands message or tool result."""
    parts: list[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(compact_json(block["json"]))
    return "\n".join(parts)


def _last_user_text(messages: Any) -> Optional[str]:
    """Text of the last user message (tool-result-only user turns are skipped)."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _text_of_blocks(msg.get("content"))
            if text:
                return text
    return None
