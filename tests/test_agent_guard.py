from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from agent_in_the_loop import evaluate_confidence, get_agent_guard
from agent_in_the_loop._context import _current_callback
from agent_in_the_loop.callbacks.langchain_callback import _AgentGuardCallback


# ---------------------------------------------------------------------------
# get_agent_guard — factory behaviour
# ---------------------------------------------------------------------------

class TestGetAgentGuard:
    def test_returns_agent_guard_callback(self):
        guard = get_agent_guard("research-agent")
        assert isinstance(guard, _AgentGuardCallback)

    def test_stores_agent_name(self):
        guard = get_agent_guard("support-bot")
        assert guard.agent_name == "support-bot"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            get_agent_guard("")

    def test_raises_on_blank_string(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            get_agent_guard("   ")

    def test_each_call_returns_new_instance(self):
        a = get_agent_guard("my-agent")
        b = get_agent_guard("my-agent")
        assert a is not b


# ---------------------------------------------------------------------------
# _AgentGuardCallback — agent_name validation
# ---------------------------------------------------------------------------

class TestAgentGuardCallbackAgentName:
    def test_stores_agent_name(self):
        handler = _AgentGuardCallback(agent_name="my-graph")
        assert handler.agent_name == "my-graph"

    def test_agent_name_is_keyword_only(self):
        with pytest.raises(TypeError):
            _AgentGuardCallback("positional-name")  # type: ignore[call-arg]

    def test_raises_on_empty_name(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            _AgentGuardCallback(agent_name="")

    def test_error_message_is_descriptive(self):
        with pytest.raises(ValueError, match="uniquely identifies"):
            _AgentGuardCallback(agent_name="")


# ---------------------------------------------------------------------------
# evaluate_confidence — agent_name forwarded in payload
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceAgentName:
    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_agent_name_included_in_payload(self, mock_post, mock_http_ok, active_handler):
        mock_post.return_value = mock_http_ok

        evaluate_confidence(api_key="key")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["agent_name"] == "test-agent"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_agent_name_matches_guard(self, mock_post, mock_http_ok):
        mock_post.return_value = mock_http_ok

        guard = get_agent_guard("analytics-bot")
        guard.trace_id = uuid.uuid4()
        token = _current_callback.set(guard)
        try:
            evaluate_confidence(api_key="key")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["agent_name"] == "analytics-bot"
        finally:
            _current_callback.reset(token)

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_trace_id_also_in_payload(self, mock_post, mock_http_ok, active_handler):
        mock_post.return_value = mock_http_ok

        evaluate_confidence(api_key="key")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["trace_id"] == str(active_handler.trace_id)

    def test_raises_when_no_active_handler(self):
        token = _current_callback.set(None)
        try:
            with pytest.raises(ValueError, match="No active callback handler"):
                evaluate_confidence(api_key="key")
        finally:
            _current_callback.reset(token)
