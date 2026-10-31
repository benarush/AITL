from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

# What the type hint advertises. At runtime *any* value is accepted and converted
# with ``str()``, so converting richer values (UUID, Enum, dates, ...) the way you
# want them stored is up to the caller.
TagValue = Union[str, int, float, bool]

# Flat key/value labels, e.g. ``{"customer_id": 5, "topic": "billing"}``.
# ``Mapping`` (not ``dict``) so a ``dict[str, str]`` is accepted too: ``dict`` is
# invariant in its value type, ``Mapping`` is covariant.
Tags = Mapping[str, TagValue]


def check_tags(tags: Optional[Tags]) -> Dict[str, str]:
    """Return a copy of ``tags`` with every value converted to ``str`` (``{}`` for ``None``).

    Keys must be strings. Values go through ``str()``, except ``bool``, which
    becomes ``"true"`` / ``"false"`` (the same form the backend stores). Doing it
    here means the payload is always JSON-safe. The backend enforces the exact
    limits (count, lengths, allowed characters) and rejects invalid tags.
    """
    if tags is None:
        return {}
    if not isinstance(tags, Mapping):
        raise TypeError("tags must be a dict of key/value pairs, e.g. {'customer_id': 5}")
    checked: Dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str):
            raise TypeError(f"tag keys must be strings, got {key!r}")
        checked[key] = ("true" if value else "false") if isinstance(value, bool) else str(value)
    return checked
