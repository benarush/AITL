from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .callbacks.langchain.langchain_callback import _LangchainAgentCallback

_current_callback: ContextVar[Optional["_LangchainAgentCallback"]] = ContextVar(
    "_current_callback", default=None
)
