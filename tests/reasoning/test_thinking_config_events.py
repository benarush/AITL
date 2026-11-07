"""``thinking_config`` on model-start events (LangChain + Strands)."""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage

from tests.backend_schema import OnChatModelStart, OnLlmStart
from trellar.callbacks.langchain.utils import _extract_thinking_config


def _serialized(model: str, class_id: list[str], **kwargs) -> dict:
    return {"id": class_id, "kwargs": {"model": model, **kwargs}}


class TestLangchainEvents:
    def test_chat_model_start_carries_constructor_config(self, handler, anthropic_config, anthropic_thinking):
        serialized = {"id": ["langchain_anthropic", "chat_models", "ChatAnthropic"], "kwargs": anthropic_config}

        handler.on_chat_model_start(serialized, [[HumanMessage(content="hi")]], run_id=uuid.uuid4())

        event = handler.events[-1]
        assert event["thinking_config"] == {"thinking": anthropic_thinking}
        assert "sk-super-secret" not in repr(event)

    def test_invocation_params_override_constructor_config(self, handler):
        serialized = _serialized("gpt-5", ["langchain_openai", "ChatOpenAI"], reasoning_effort="low")

        handler.on_chat_model_start(
            serialized, [[HumanMessage(content="hi")]], run_id=uuid.uuid4(),
            invocation_params={"reasoning_effort": "high"},
        )

        assert handler.events[-1]["thinking_config"] == {"reasoning_effort": "high"}

    def test_llm_start_carries_config(self, handler, gemini_config):
        serialized = {"id": ["langchain_google_genai", "ChatGoogleGenerativeAI"], "kwargs": gemini_config}

        handler.on_llm_start(serialized, ["prompt"], run_id=uuid.uuid4())

        assert handler.events[-1]["thinking_config"] == {"thinking_budget": 1024, "include_thoughts": True}

    def test_unset_gives_none(self, handler):
        handler.on_chat_model_start(
            {"kwargs": {"model": "gpt-4o"}}, [[HumanMessage(content="hi")]], run_id=uuid.uuid4()
        )

        assert handler.events[-1]["thinking_config"] is None

    def test_event_matches_backend_schema(self, handler, anthropic_config):
        serialized = {"id": ["ChatAnthropic"], "kwargs": anthropic_config}
        handler.on_chat_model_start(serialized, [[HumanMessage(content="hi")]], run_id=uuid.uuid4())
        handler.on_llm_start(serialized, ["p"], run_id=uuid.uuid4())

        OnChatModelStart.model_validate(handler.events[-2])
        OnLlmStart.model_validate(handler.events[-1])

    @pytest.mark.parametrize("serialized", [None, {}, {"kwargs": None}, {"id": None, "kwargs": "oops"}, "text"])
    def test_helper_never_raises_on_odd_serialized(self, serialized):
        assert _extract_thinking_config(serialized, None) is None
        assert _extract_thinking_config(serialized, {"reasoning_effort": "low"}) in (None, {"reasoning_effort": "low"})


class TestStrandsEvents:
    @pytest.fixture(autouse=True)
    def _strands(self):
        pytest.importorskip("strands")

    def _run(self, model):
        from strands import Agent

        from tests.strands_factories import events_of, text_turn
        from trellar import trellar_strands_agent

        trellar_agent = trellar_strands_agent("t")
        model.turns = [text_turn("hi")]
        Agent(name="calc", model=model, hooks=[trellar_agent], callback_handler=None)("hello")
        (event,) = events_of(trellar_agent, "on_chat_model_start")
        return event

    def test_params_thinking_is_recorded(self, anthropic_thinking):
        from tests.strands_factories import FakeModel

        model = FakeModel([], model_id="claude-sonnet-4-5")
        model.update_config(params={"thinking": anthropic_thinking}, api_key="sk-super-secret")

        event = self._run(model)

        assert event["thinking_config"] == {"thinking": anthropic_thinking}
        assert "sk-super-secret" not in repr(event)

    def test_unset_gives_none(self):
        from tests.strands_factories import FakeModel

        assert self._run(FakeModel([]))["thinking_config"] is None

    def test_get_config_raising_gives_none(self):
        from tests.strands_factories import FakeModel

        class Broken(FakeModel):
            def get_config(self):
                raise RuntimeError("no config")

        assert self._run(Broken([]))["thinking_config"] is None
