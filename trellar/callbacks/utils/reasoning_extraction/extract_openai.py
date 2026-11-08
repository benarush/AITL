"""Reasoning extraction for OpenAI models (GPT-5, o-series, gpt-oss, Azure OpenAI).

OpenAI uses ``reasoning_effort`` (``"low" | "medium" | "high"``) or a nested
``reasoning`` object such as ``{"effort": "high", "summary": "auto"}``. Values are
returned as-is. Twin of ``aitl_ts_lib/.../reasoningExtraction/extractOpenai.ts``.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from ._helpers import pick_keys

KEYS = ("reasoning_effort", "reasoning", "reasoningEffort")

# o1 / o3-mini / o4 ... but not a stray "o" inside another word.
_O_SERIES = re.compile(r"(?<![a-z0-9])o\d")


def matches(hint: str) -> bool:
    """True when the (lower-cased) provider/model hint looks like OpenAI."""
    return any(token in hint for token in ("openai", "gpt", "azure")) or bool(_O_SERIES.search(hint))


def extract(sources: Iterable[Mapping]) -> Optional[dict[str, Any]]:
    """The OpenAI reasoning settings found in *sources*, or ``None``."""
    return pick_keys(sources, KEYS)
