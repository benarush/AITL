from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
import requests

from agent_in_the_loop import settings
from agent_in_the_loop.agent_loop import AgentLoopResult, evaluate_confidence
from agent_in_the_loop._context import _current_callback


def _successful_post(score: int = 8, explanation: str = "looks good"):
    """Return a mock requests.post response with a JSON body."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"score": score, "explanation": explanation}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# evaluate_confidence — request construction
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceRequest:
    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_posts_to_fixed_endpoint(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        url = mock_post.call_args.args[0]
        assert url == f"{settings.AITL_ENDPOINT}/agent-gateway/v1/agent-loop"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_sends_bearer_auth_header(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer key-123"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_falls_back_to_env_api_key(self, mock_post, active_handler, monkeypatch):
        mock_post.return_value = _successful_post()
        monkeypatch.setenv(settings.ENV_AITL_API_KEY, "env-key")

        evaluate_confidence()

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer env-key"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_sends_context_events_from_active_handler(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()
        active_handler.events.append({"event": "on_llm_start"})

        evaluate_confidence(api_key="key-123")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["context"] == active_handler.events

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_default_timeout_is_30_seconds(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        assert mock_post.call_args.kwargs["timeout"] == 30.0

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_custom_timeout_is_forwarded(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123", timeout=5.0)

        assert mock_post.call_args.kwargs["timeout"] == 5.0


# ---------------------------------------------------------------------------
# evaluate_confidence — validation errors
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceValidation:
    def test_raises_when_no_active_handler(self):
        token = _current_callback.set(None)
        try:
            with pytest.raises(ValueError, match="No active callback handler"):
                evaluate_confidence(api_key="key-123")
        finally:
            _current_callback.reset(token)

    def test_raises_when_trace_id_not_resolved(self):
        from agent_in_the_loop.callbacks.langchain_callback import _AgentGuardCallback

        handler = _AgentGuardCallback(agent_name="test-agent")
        token = _current_callback.set(handler)
        try:
            with pytest.raises(ValueError, match="trace_id could not be resolved"):
                evaluate_confidence(api_key="key-123")
        finally:
            _current_callback.reset(token)

    @patch("agent_in_the_loop.agent_loop.settings")
    def test_raises_when_no_api_key(self, mock_settings, active_handler):
        mock_settings.AITL_ENDPOINT = "https://trellar.io"
        mock_settings.get_env_api_key.return_value = None
        mock_settings.ENV_AITL_API_KEY = "AGENT_IN_THE_LOOP_API_KEY"

        with pytest.raises(ValueError, match="api_key must be provided"):
            evaluate_confidence()


# ---------------------------------------------------------------------------
# evaluate_confidence — response handling
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceResponse:
    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_returns_agent_loop_result(self, mock_post, active_handler):
        mock_post.return_value = _successful_post(score=3, explanation="meh")

        result = evaluate_confidence(api_key="key-123")

        assert isinstance(result, AgentLoopResult)
        assert result.score == 3
        assert result.explanation == "meh"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_raises_http_error_on_failure_response(self, mock_post, active_handler):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("boom")
        mock_post.return_value = resp

        with pytest.raises(requests.HTTPError):
            evaluate_confidence(api_key="key-123")
