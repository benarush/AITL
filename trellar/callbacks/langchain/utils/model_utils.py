from typing import Any, Optional


def _extract_model_name(serialized: dict[str, Any]) -> Optional[str]:
    """
    Pull the real model identifier out of a serialized LLM dict.

    LangChain puts the *class* name in ``serialized["name"]`` (e.g. "ChatOpenAI")
    but the actual model string (e.g. "gpt-4o") lives inside ``kwargs``.
    """
    kwargs = serialized.get("kwargs", {})
    name = kwargs.get("model_name") or kwargs.get("model")
    if not name:
        # Fall back to the class name so we always have something.
        name = serialized.get("name")
    return name
