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


def _first_int(source: Any, *keys: str) -> Optional[int]:
    """First value under *keys* in *source* that is a real number, else None."""
    for key in keys:
        value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _extract_token_counts(
    usage_metadata: Any, llm_output: Optional[dict[str, Any]]
) -> tuple[Optional[int], Optional[int]]:
    """Return ``(input_tokens, output_tokens)`` for one model call; ``None`` if unreported.

    ``AIMessage.usage_metadata`` (the LangChain standard) wins; otherwise fall back
    to ``llm_output["token_usage"|"usage"]``, which uses ``prompt_tokens`` /
    ``completion_tokens`` (OpenAI) or ``input_tokens`` / ``output_tokens`` (Anthropic).
    """
    llm_output = llm_output or {}
    for source in (usage_metadata, llm_output.get("token_usage"), llm_output.get("usage")):
        if not source:
            continue
        inp = _first_int(source, "input_tokens", "prompt_tokens")
        out = _first_int(source, "output_tokens", "completion_tokens")
        if inp is not None or out is not None:
            return inp, out
    return None, None
