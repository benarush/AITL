"""Reasoning extraction for Anthropic Claude models.

Claude is configured with a single ``thinking`` object, e.g.
``{"type": "enabled", "budget_tokens": 2000}``. It is returned as-is.

Covers LangChain ``ChatAnthropic`` (``thinking`` on the model), Strands
``AnthropicModel`` (``params["thinking"]``) and Claude on Bedrock
(``additional_request_fields["thinking"]``) - the latter two are reached through
``flatten_sources``. Twin of ``aitl_ts_lib/.../reasoningExtraction/extractAnthropic.ts``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._helpers import pick_keys

KEYS = ("thinking",)


def matches(hint: str) -> bool:
    """True when the (lower-cased) provider/model hint looks like Anthropic/Claude."""
    return "anthropic" in hint or "claude" in hint


def extract(sources: Iterable[Mapping]) -> Optional[dict[str, Any]]:
    """The Anthropic thinking settings found in *sources*, or ``None``."""
    return pick_keys(sources, KEYS)
