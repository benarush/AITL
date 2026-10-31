from __future__ import annotations

import datetime
import json
import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
import requests

from trellar import settings
from trellar.agent_loop import (
    AgentLoopResult,
    evaluate_confidence,
    trellar_langchain_agent,
    trellar_langchain_single_call,
)
from trellar._context import _current_callback
from trellar.callbacks.langchain.single_langchain_callback import _LangchainSingleCallCallback


def _successful_post(
    score: int = 8,
    explanation: str = "looks good",
    decision_identifier: str = "decision-1",
    should_stop_network: bool = False,
):
    """Return a mock requests.post response with a JSON body."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "score": score,
        "explanation": explanation,
        "decision_identifier": decision_identifier,
        "should_stop_network": should_stop_network,
    }
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# evaluate_confidence — request construction
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceRequest:
    @patch("trellar.agent_loop.requests.post")
    def test_posts_to_fixed_endpoint(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        url = mock_post.call_args.args[0]
        assert url == f"{settings.DEFAULT_ENDPOINT.rstrip('/')}/agent-gateway/v1/agent-loop"

    @patch("trellar.agent_loop.requests.post")
    def test_sends_bearer_auth_header(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer key-123"

    @patch("trellar.agent_loop.requests.post")
    def test_falls_back_to_env_api_key(self, mock_post, active_handler, monkeypatch):
        mock_post.return_value = _successful_post()
        monkeypatch.setenv(settings.ENV_TRELLAR_API_KEY, "env-key")

        evaluate_confidence()

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer env-key"

    @patch("trellar.agent_loop.requests.post")
    def test_sends_context_events_from_active_handler(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()
        active_handler.events.append({"event": "on_llm_start"})

        evaluate_confidence(api_key="key-123")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["context"] == active_handler.events

    @patch("trellar.agent_loop.requests.post")
    def test_single_call_flag_false_for_graph_handler(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["single_call"] is False

    @patch("trellar.agent_loop.requests.post")
    def test_single_call_flag_true_for_single_call_handler(self, mock_post):
        mock_post.return_value = _successful_post()
        handler = _LangchainSingleCallCallback(agent_name="single-test")
        handler.trace_id = uuid.uuid4()
        token = _current_callback.set(handler)
        try:
            evaluate_confidence(api_key="key-123")
        finally:
            _current_callback.reset(token)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["single_call"] is True

    @patch("trellar.agent_loop.requests.post")
    def test_default_timeout_is_30_seconds(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        assert mock_post.call_args.kwargs["timeout"] == 30.0

    @patch("trellar.agent_loop.requests.post")
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
        from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback

        handler = _LangchainAgentCallback(agent_name="test-agent")
        token = _current_callback.set(handler)
        try:
            with pytest.raises(ValueError, match="trace_id could not be resolved"):
                evaluate_confidence(api_key="key-123")
        finally:
            _current_callback.reset(token)

    @patch("trellar.agent_loop.settings")
    def test_raises_when_no_api_key(self, mock_settings, active_handler):
        mock_settings.DEFAULT_ENDPOINT = "https://api.trellar.io/"
        mock_settings.get_env_api_key.return_value = None
        mock_settings.ENV_TRELLAR_API_KEY = "TRELLAR_API_KEY"

        with pytest.raises(ValueError, match="api_key must be provided"):
            evaluate_confidence()


# ---------------------------------------------------------------------------
# evaluate_confidence — response handling
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceResponse:
    @patch("trellar.agent_loop.requests.post")
    def test_returns_agent_loop_result(self, mock_post, active_handler):
        mock_post.return_value = _successful_post(score=3, explanation="meh")

        result = evaluate_confidence(api_key="key-123")

        assert isinstance(result, AgentLoopResult)
        assert result.score == 3
        assert result.explanation == "meh"

    @patch("trellar.agent_loop.requests.post")
    def test_raises_http_error_on_failure_response(self, mock_post, active_handler):
        resp = MagicMock()
        resp.raise_for_status.side_effect = requests.HTTPError("boom")
        mock_post.return_value = resp

        with pytest.raises(requests.HTTPError):
            evaluate_confidence(api_key="key-123")


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


class TestTags:
    @patch("trellar.agent_loop.requests.post")
    def test_payload_has_empty_tags_by_default(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123")

        assert mock_post.call_args.kwargs["json"]["tags"] == {}

    @patch("trellar.agent_loop.requests.post")
    def test_creation_tags_are_sent(self, mock_post):
        mock_post.return_value = _successful_post()
        handler = trellar_langchain_agent("tags-test", tags={"env": "prod"})
        handler.trace_id = uuid.uuid4()
        token = _current_callback.set(handler)
        try:
            evaluate_confidence(api_key="key-123")
        finally:
            _current_callback.reset(token)

        assert mock_post.call_args.kwargs["json"]["tags"] == {"env": "prod"}

    @patch("trellar.agent_loop.requests.post")
    def test_per_call_tags_are_merged_and_win(self, mock_post):
        mock_post.return_value = _successful_post()
        handler = trellar_langchain_agent("tags-test", tags={"env": "prod", "topic": "a"})
        handler.trace_id = uuid.uuid4()
        token = _current_callback.set(handler)
        try:
            evaluate_confidence(api_key="key-123", tags={"topic": "b", "customer_id": 5})
        finally:
            _current_callback.reset(token)

        assert mock_post.call_args.kwargs["json"]["tags"] == {"env": "prod", "topic": "b", "customer_id": "5"}
        assert handler.tags == {"env": "prod", "topic": "a"}  # creation tags are not mutated

    @pytest.mark.parametrize(
        "factory",
        [trellar_langchain_agent, trellar_langchain_single_call],
    )
    def test_factories_store_tags(self, factory):
        assert factory("tags-test", tags={"a": 1}).tags == {"a": "1"}

    @pytest.mark.parametrize("bad", ["x", ["a"], {1: "a"}])
    def test_invalid_tags_shape_is_rejected(self, bad):
        with pytest.raises(TypeError):
            trellar_langchain_agent("tags-test", tags=bad)

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("x", "x"),
            (5, "5"),
            (1.5, "1.5"),
            (True, "true"),  # same form the backend stores
            (False, "false"),
            (uuid.UUID(int=321), "00000000-0000-0000-0000-000000000141"),
            (Decimal("1.50"), "1.50"),
            (datetime.date(2026, 10, 31), "2026-10-31"),
            (None, "None"),
        ],
    )
    def test_every_value_is_converted_with_str(self, value, expected):
        tags = trellar_langchain_agent("tags-test", tags={"k": value}).tags
        assert tags == {"k": expected}
        json.dumps(tags)  # what requests does with the payload

    @patch("trellar.agent_loop.requests.post")
    def test_uuid_per_call_tag_is_sent_as_a_string(self, mock_post, active_handler):
        mock_post.return_value = _successful_post()

        evaluate_confidence(api_key="key-123", tags={"customer": uuid.UUID(int=321)})

        sent = mock_post.call_args.kwargs["json"]
        assert sent["tags"] == {"customer": "00000000-0000-0000-0000-000000000141"}
        json.dumps(sent["tags"])

    @patch("trellar.agent_loop.requests.post")
    def test_invalid_per_call_tags_shape_is_rejected(self, mock_post, active_handler):
        with pytest.raises(TypeError):
            evaluate_confidence(api_key="key-123", tags=["a"])
        mock_post.assert_not_called()
