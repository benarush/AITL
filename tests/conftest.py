from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from agent_in_the_loop._context import _current_callback
from agent_in_the_loop.callbacks.langchain_callback import DebugCallbackHandler


@pytest.fixture
def mock_http_ok() -> MagicMock:
    """A successful requests.post response (score=8)."""
    resp = MagicMock()
    resp.json.return_value = {"score": 8, "explanation": "looks good"}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def active_handler() -> DebugCallbackHandler:
    """A DebugCallbackHandler registered in _current_callback with a fake trace_id.

    Automatically cleans up the ContextVar after each test via reset().
    """
    handler = DebugCallbackHandler(agent_name="test-agent")
    handler.trace_id = uuid.uuid4()
    token = _current_callback.set(handler)
    yield handler
    _current_callback.reset(token)
