from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from trellar import evaluate_confidence, trellar_langchain_agent
from trellar._context import _current_callback
from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback


# ---------------------------------------------------------------------------
# trellar_langchain_agent — factory behaviour
# ---------------------------------------------------------------------------

class TestTrellarLangchainAgent:
    def test_returns_langchain_agent_callback(self):
        trellar_agent = trellar_langchain_agent("research-agent")
        assert isinstance(trellar_agent, _LangchainAgentCallback)

    def test_stores_agent_name(self):
        trellar_agent = trellar_langchain_agent("support-bot")
        assert trellar_agent.agent_name == "support-bot"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            trellar_langchain_agent("")

    def test_raises_on_blank_string(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            trellar_langchain_agent("   ")

    def test_each_call_returns_new_instance(self):
        a = trellar_langchain_agent("my-agent")
        b = trellar_langchain_agent("my-agent")
        assert a is not b


# ---------------------------------------------------------------------------
# _LangchainAgentCallback — agent_name validation
# ---------------------------------------------------------------------------

class TestLangchainAgentCallbackAgentName:
    def test_stores_agent_name(self):
        handler = _LangchainAgentCallback(agent_name="my-graph")
        assert handler.agent_name == "my-graph"

    def test_agent_name_is_keyword_only(self):
        with pytest.raises(TypeError):
            _LangchainAgentCallback("positional-name")  # type: ignore[call-arg]

    def test_raises_on_empty_name(self):
        with pytest.raises(ValueError, match="agent_name is required"):
            _LangchainAgentCallback(agent_name="")

    def test_error_message_is_descriptive(self):
        with pytest.raises(ValueError, match="uniquely identifies"):
            _LangchainAgentCallback(agent_name="")


# ---------------------------------------------------------------------------
# evaluate_confidence — agent_name forwarded in payload
# ---------------------------------------------------------------------------

class TestEvaluateConfidenceAgentName:
    @patch("trellar.agent_loop.requests.post")
    def test_agent_name_included_in_payload(self, mock_post, mock_http_ok, active_handler):
        mock_post.return_value = mock_http_ok

        evaluate_confidence(api_key="key")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["agent_name"] == "test-agent"

    @patch("trellar.agent_loop.requests.post")
    def test_agent_name_matches_trellar_agent(self, mock_post, mock_http_ok):
        mock_post.return_value = mock_http_ok

        trellar_agent = trellar_langchain_agent("analytics-bot")
        trellar_agent.trace_id = uuid.uuid4()
        token = _current_callback.set(trellar_agent)
        try:
            evaluate_confidence(api_key="key")
            payload = mock_post.call_args.kwargs["json"]
            assert payload["agent_name"] == "analytics-bot"
        finally:
            _current_callback.reset(token)

    @patch("trellar.agent_loop.requests.post")
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


# ---------------------------------------------------------------------------
# Deprecated get_*_guard aliases
# ---------------------------------------------------------------------------

class TestDeprecatedGuardAliases:
    def test_alias_warns_and_forwards(self):
        from trellar.agent_loop import _deprecated_alias

        def trellar_fake_agent(agent_name, observability_mode=None):
            return (agent_name, observability_mode)

        alias = _deprecated_alias("get_fake_guard", trellar_fake_agent)
        assert alias.__name__ == "get_fake_guard"
        with pytest.warns(DeprecationWarning, match="trellar_fake_agent"):
            assert alias("my-agent", observability_mode="x") == ("my-agent", "x")

    @pytest.mark.parametrize(
        "old_name, new_name",
        [
            ("get_agent_guard", "trellar_langchain_agent"),
            ("get_single_call_guard", "trellar_langchain_single_call"),
            ("get_strands_guard", "trellar_strands_agent"),
            ("get_strands_single_call_guard", "trellar_strands_single_call"),
        ],
    )
    def test_old_name_points_to_new_function(self, old_name, new_name):
        import trellar

        with pytest.warns(DeprecationWarning, match=new_name):
            # Calling without args raises TypeError from the forwarded function,
            # after the warning has been emitted.
            with pytest.raises(TypeError):
                getattr(trellar, old_name)()

    def test_old_names_still_exported(self):
        import trellar

        for name in (
            "get_agent_guard",
            "get_single_call_guard",
            "get_strands_guard",
            "get_strands_single_call_guard",
        ):
            assert name in trellar.__all__
            assert callable(getattr(trellar, name))

    def test_get_agent_guard_returns_langchain_agent_callback(self):
        from trellar import get_agent_guard

        with pytest.warns(DeprecationWarning, match="trellar_langchain_agent"):
            handler = get_agent_guard("legacy-agent")
        assert isinstance(handler, _LangchainAgentCallback)
        assert handler.agent_name == "legacy-agent"
