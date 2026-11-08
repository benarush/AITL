"""Fallback reasoning extraction for providers we have no dedicated module for.

Used only when no provider module matched the hint (Ollama, DeepSeek, Bedrock Nova,
LiteLLM, unknown classes ...). It looks for the union of every known reasoning key plus
the generic ``think`` / ``reasoningConfig`` spellings, still strictly from an allowlist.
Twin of ``aitl_ts_lib/.../reasoningExtraction/extractGeneric.ts``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from . import extract_anthropic, extract_gemini, extract_openai
from ._helpers import pick_keys

KEYS = tuple(
    dict.fromkeys(  # de-duplicate, keep order
        (
            *extract_anthropic.KEYS,
            *extract_gemini.KEYS,
            *extract_openai.KEYS,
            "think",
            "reasoningConfig",
        )
    )
)


def matches(hint: str) -> bool:
    """The generic extractor is never chosen by hint; it is the dispatcher's fallback."""
    return False


def extract(sources: Iterable[Mapping]) -> Optional[dict[str, Any]]:
    """Any known reasoning setting found in *sources*, or ``None``."""
    return pick_keys(sources, KEYS)
