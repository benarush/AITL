"""Reasoning extraction for Google Gemini models.

Gemini exposes thinking as flat options (``thinking_budget``, ``thinking_level``,
``include_thoughts``) or as one nested ``thinking_config`` object (Strands keeps it in
``params``). Whatever is found is returned as-is. The camelCase spellings are what the
JS SDKs use; they are listed here too so the Python and TS allowlists stay identical.
Twin of ``aitl_ts_lib/.../reasoningExtraction/extractGemini.ts``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._helpers import pick_keys

KEYS = (
    "thinking_budget",
    "thinking_level",
    "include_thoughts",
    "thinking_config",
    # langchain-google-genai stores the thinking level under this name (``thinking_level`` is
    # only an alias), so a constructor-only config carries it as ``reasoning_effort``.
    "reasoning_effort",
    "thinkingBudget",
    "thinkingLevel",
    "includeThoughts",
    "thinkingConfig",
)


def matches(hint: str) -> bool:
    """True when the (lower-cased) provider/model hint looks like Gemini / Google."""
    return any(token in hint for token in ("gemini", "google", "vertex", "genai"))


def extract(sources: Iterable[Mapping]) -> Optional[dict[str, Any]]:
    """The Gemini thinking settings found in *sources*, or ``None``."""
    return pick_keys(sources, KEYS)
