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
    - ``human``: content of the last HumanMessage, or ``None``

    For multi-turn conversation histories the *last* human turn is used as the
    active prompt because that is what the LLM is responding to.

    Handles both LangChain ``BaseMessage`` objects (standard) and plain dicts
    (e.g. when Phoenix auto-instrumentation serialises messages before passing
    them to the callback).  Also accepts ``"user"`` as a synonym for ``"human"``
    to cover OpenAI-style role names.
    """
    system: Optional[str] = None
    human: Optional[str] = None

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
            # "user" is the OpenAI/Phoenix style; keep overwriting so the last wins.
            human = content

    return {"system": system, "human": human}
