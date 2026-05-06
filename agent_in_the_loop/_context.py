from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .callbacks.langchain_callback import DebugCallbackHandler

_current_callback: ContextVar[Optional["DebugCallbackHandler"]] = ContextVar(
    "_current_callback", default=None
)
