from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .callbacks.langchain_callback import _AgentGuardCallback

_current_callback: ContextVar[Optional["_AgentGuardCallback"]] = ContextVar(
    "_current_callback", default=None
)
