"""Single entry point for "what reasoning/thinking settings was this model given?".

How it works (read this first):

1. Callers hand over the model's raw config dicts, a *provider hint* (class name /
   module / serialized id) and the model name.
2. ``flatten_sources`` turns the config dicts into an ordered list of plain mappings,
   also looking one level into ``params`` / ``model_kwargs`` / ``additional_request_fields``.
3. Each provider module (``extract_anthropic``, ``extract_gemini``, ``extract_openai``)
   answers "does this hint look like me?"; every module that says yes extracts its own
   allowlisted keys. If none says yes, ``extract_generic`` is used instead.
4. The per-provider results are merged and returned exactly as found, or ``None``.

Only allowlisted keys are ever read, so secrets in the config (``api_key``, headers ...)
cannot leak, and any failure degrades to ``None`` instead of raising.
Twin of ``aitl_ts_lib/.../reasoningExtraction/extractReasoning.ts``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from . import extract_anthropic, extract_gemini, extract_generic, extract_openai
from ._helpers import flatten_sources

logger = logging.getLogger(__name__)

# Provider modules tried by hint, in a fixed order. Each exposes ``matches`` / ``extract``.
_PROVIDERS = (extract_anthropic, extract_gemini, extract_openai)


def _hint_text(provider_hint: Any, model_name: Any) -> str:
    """Lower-cased text built from the hint (str, or list such as a LangChain id) and model name."""
    parts: list[Any] = []
    if isinstance(provider_hint, (list, tuple)):
        parts.extend(provider_hint)
    else:
        parts.append(provider_hint)
    parts.append(model_name)
    return " ".join(str(part) for part in parts if part).lower()


def extract_reasoning(
    config_sources: Optional[Iterable[Any]],
    provider_hint: Any = None,
    model_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Return the model's reasoning settings as a dict (values untouched), or ``None``.

    Args:
        config_sources: raw config objects, constructor config first and per-call
            config last (later ones win). Non-dicts are ignored.
        provider_hint: class name, module path or serialized id (str or list) that
            identifies the provider.
        model_name: the model id, e.g. ``"claude-sonnet-4-5"`` (also used for matching).
    """
    try:
        sources = flatten_sources(config_sources or ())
        if not sources:
            return None
        hint = _hint_text(provider_hint, model_name)
        extractors = [provider for provider in _PROVIDERS if provider.matches(hint)] or [extract_generic]
        merged: dict[str, Any] = {}
        for extractor in extractors:
            merged.update(extractor.extract(sources) or {})
        return merged or None
    except Exception:  # telemetry must never break the agent
        logger.debug("reasoning extraction failed", exc_info=True)
        return None
