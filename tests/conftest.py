from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from trellar._context import _current_callback
from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback


@pytest.fixture
def mock_http_ok() -> MagicMock:
    """A successful requests.post response (score=8)."""
    resp = MagicMock()
    resp.json.return_value = {
        "score": 8,
        "explanation": "looks good",
        "decision_identifier": "decision-1",
        "should_stop_network": False,
    }
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture
def active_handler() -> _LangchainAgentCallback:
    """An _LangchainAgentCallback registered in _current_callback with a fake trace_id.

    Automatically cleans up the ContextVar after each test via reset().
    """
    handler = _LangchainAgentCallback(agent_name="test-agent")
    handler.trace_id = uuid.uuid4()
    token = _current_callback.set(handler)
    yield handler
    _current_callback.reset(token)


@pytest.fixture
def post_mock(monkeypatch, mock_http_ok):
    """Patch the HTTP call made by evaluate_confidence (score 8) and provide an API key.

    Also isolates the active-callback ContextVar, so tests that start a run (which
    registers the handler there) cannot leak it into other tests.
    """
    monkeypatch.setenv("TRELLAR_API_KEY", "test-key")
    token = _current_callback.set(None)
    with patch("trellar.agent_loop.requests.post", return_value=mock_http_ok) as post:
        yield post
    _current_callback.reset(token)
