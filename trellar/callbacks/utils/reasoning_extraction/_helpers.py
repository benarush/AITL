"""Small, provider-neutral building blocks for the reasoning extractors.

Everything here is defensive: it accepts anything (``None``, non-dicts, objects whose
attribute access raises) and never raises, so a weird model config can only ever
result in "nothing found", never in a broken agent run.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Optional

# Sub-dicts of a model config that commonly carry extra, provider-specific request
# settings. Reasoning options often live in one of them rather than at the top level
# (Strands: ``params`` / ``additional_request_fields``; LangChain: ``model_kwargs`` /
# ``extra_body``). Only these names are ever descended into, and only one level deep.
NESTED_SOURCE_KEYS = (
    "params",
    "model_kwargs",
    "additional_request_fields",
    "additionalRequestFields",
    "extra_body",
    "extraBody",
)


def as_mapping(value: Any) -> Optional[Mapping]:
    """Return *value* if it is a mapping, else ``None``."""
    return value if isinstance(value, Mapping) else None


def flatten_sources(config_sources: Iterable[Any]) -> list[Mapping]:
    """Expand raw config objects into one ordered list of plain mappings.

    For each source we emit the source itself followed by its known nested sub-dicts
    (see ``NESTED_SOURCE_KEYS``). Non-mappings are dropped. Order matters: a later
    mapping wins over an earlier one when both define the same key, so callers should
    pass constructor config first and per-call config last.
    """
    flat: list[Mapping] = []
    for source in config_sources or ():
        mapping = as_mapping(source)
        if mapping is None:
            continue
        flat.append(mapping)
        for nested_key in NESTED_SOURCE_KEYS:
            nested = as_mapping(mapping.get(nested_key))
            if nested is not None:
                flat.append(nested)
    return flat


def pick_keys(sources: Iterable[Mapping], allowed_keys: Iterable[str]) -> Optional[dict[str, Any]]:
    """Copy *only* ``allowed_keys`` out of *sources*, values untouched.

    This is the single place where values leave a model config, which is why it takes
    an explicit allowlist: secrets such as ``api_key`` can never be picked up because
    they are never asked for. A key whose value is ``None`` counts as "not set".
    Returns ``None`` when nothing was found.
    """
    allowed = tuple(allowed_keys)
    found: dict[str, Any] = {}
    for source in sources:
        for key in allowed:
            value = source.get(key)
            if value is not None:
                found[key] = value
    return found or None
