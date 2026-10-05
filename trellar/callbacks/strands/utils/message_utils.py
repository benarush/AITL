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


def _llm_human_input(messages: Any) -> Optional[str]:
    """Build the ``input.human`` string for an LLM call from Strands messages.

    The current message is the last user turn that has text. Earlier user and
    assistant text turns are folded into the same string::

        Human: turn 1 text
        AI LLM: turn 1 response text

        Current message - <last user text>

    With no earlier turns, returns just the plain current text. Tool-result-only
    user messages and tool-call-only assistant messages have no text and are
    skipped; anything after the current user turn is ignored. Returns ``None``
    when there is no user text at all.
    """
    turns: list[tuple[str, str]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_of_blocks(msg.get("content"))
        if text:
            turns.append((role, text))

    last_user_idx = next(
        (i for i in range(len(turns) - 1, -1, -1) if turns[i][0] == "user"), None
    )
    if last_user_idx is None:
        return None

    current = turns[last_user_idx][1]
    history_lines = [
        f"{'Human' if role == 'user' else 'AI LLM'}: {text}"
        for role, text in turns[:last_user_idx]
    ]
    if not history_lines:
        return current
    return "\n".join(history_lines) + f"\n\nCurrent message - {current}"


def _last_user_text(messages: Any) -> Optional[str]:
    """Text of the last user message (tool-result-only user turns are skipped)."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _text_of_blocks(msg.get("content"))
            if text:
                return text
    return None
