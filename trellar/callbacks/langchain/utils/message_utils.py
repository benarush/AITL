import json
from typing import Any, Optional


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a LangChain BaseMessage to a plain dict (role + content + extras)."""
    if not (hasattr(msg, "type") and hasattr(msg, "content")):
        return {"raw": str(msg)}

    result: dict[str, Any] = {"role": msg.type, "content": msg.content}

    additional = getattr(msg, "additional_kwargs", {})
    if additional:
        # Capture tool_calls, function_call, etc.
        result["additional_kwargs"] = additional

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def _content_to_str(content: Any) -> str:
    """Convert a message content value to a plain string.

    LangChain message content can be a str, a list of dicts (multimodal),
    or any other JSON-serializable value for structured outputs.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _extract_llm_input(messages: list[Any]) -> dict[str, Optional[str]]:
    """Extract structured system/human fields from a list of LangChain messages.

    Returns a dict with:
    - ``system``: content of the first SystemMessage, or ``None``
    - ``human``: the active prompt, or ``None``

    The active prompt is the *last* human turn, because that is what the LLM
    is responding to.  When earlier Human/AI turns exist before it, they are
    folded into the same string (no extra dict key)::

        Human: turn 1 text
        AI LLM: turn 1 response text
        Human: turn 2 text
        AI LLM: turn 2 response text

        Current message - <last human text>

    With no prior history, ``human`` is just the plain last human text.  Only
    human and non-empty AI text turns are kept in the history; tool messages
    and tool-call-only AI messages are skipped.

    Handles both LangChain ``BaseMessage`` objects (standard) and plain dicts
    (e.g. when Phoenix auto-instrumentation serialises messages before passing
    them to the callback).  Also accepts ``"user"`` as a synonym for ``"human"``
    and ``"assistant"`` as a synonym for ``"ai"`` to cover OpenAI-style role names.
    """
    system: Optional[str] = None
    # (role, content) pairs for human/ai turns, in order; role is "human" or "ai".
    turns: list[tuple[str, str]] = []

    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            # Standard LangChain BaseMessage object
            role = str(msg.type).lower()
            content = _content_to_str(msg.content)
        elif isinstance(msg, dict):
            # Serialised dict — may use "role" (OpenAI/Phoenix) or "type" (LangChain)
            role = str(msg.get("role") or msg.get("type") or "").lower()
            content = _content_to_str(msg.get("content") or "")
        else:
            continue

        if role == "system" and system is None:
            system = content
        elif role in ("human", "user"):
            turns.append(("human", content))
        elif role in ("ai", "assistant"):
            turns.append(("ai", content))

    last_human_idx = next(
        (i for i in range(len(turns) - 1, -1, -1) if turns[i][0] == "human"), None
    )
    if last_human_idx is None:
        return {"system": system, "human": None}

    current = turns[last_human_idx][1]
    history_lines = [
        f"{'Human' if role == 'human' else 'AI LLM'}: {content}"
        for role, content in turns[:last_human_idx]
        # Tool-call-only AI messages have empty text; skip them.
        if content or role == "human"
    ]
    if not history_lines:
        return {"system": system, "human": current}

    human = "\n".join(history_lines) + f"\n\nCurrent message - {current}"
    return {"system": system, "human": human}
