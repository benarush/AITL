"""Tests for ``input_tokens`` / ``output_tokens`` on ``on_llm_end`` events
(LangChain + Strands)."""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback

from tests.factories import make_ai_message, make_llm_result


@pytest.fixture
def handler() -> _LangchainAgentCallback:
    cb = _LangchainAgentCallback(agent_name="test-agent")
    cb.on_chain_start({"name": "graph"}, {"messages": []}, run_id=uuid.uuid4())
    return cb


def _llm_end(cb: _LangchainAgentCallback, result: LLMResult) -> dict:
    cb.on_llm_end(result, run_id=uuid.uuid4())
    return cb.events[-1]


def test_langchain_counts_from_usage_metadata(handler):
    message = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
    )
    event = _llm_end(handler, make_llm_result(message=message))

    assert event["input_tokens"] == 12
    assert event["output_tokens"] == 5


@pytest.mark.parametrize(
    "usage_key, usage",
    [
        ("token_usage", {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),  # OpenAI style
        ("usage", {"input_tokens": 7, "output_tokens": 3}),  # Anthropic style
    ],
)
def test_langchain_counts_fall_back_to_llm_output(handler, usage_key, usage):
    result = LLMResult(
        generations=[[ChatGeneration(message=make_ai_message("hi"))]],
        llm_output={usage_key: usage},
    )
    event = _llm_end(handler, result)

    assert (event["input_tokens"], event["output_tokens"]) == (7, 3)
    assert event["token_usage"] == usage  # the existing field is unchanged


def test_langchain_no_usage_gives_none(handler):
    event = _llm_end(handler, make_llm_result(message=make_ai_message("hi")))

    assert event["input_tokens"] is None
    assert event["output_tokens"] is None


def test_strands_counts_from_message_usage():
    pytest.importorskip("strands")
    from strands import Agent

    from trellar import trellar_strands_agent
    from tests.strands_factories import FakeModel, events_of, text_turn

    trellar_agent = trellar_strands_agent("t")
    # text_turn's metadata chunk reports inputTokens=1, outputTokens=1
    Agent(
        name="calc", model=FakeModel([text_turn("hi")]), hooks=[trellar_agent], callback_handler=None,
    )("hello")

    (event,) = events_of(trellar_agent, "on_llm_end")
    assert event["input_tokens"] == 1
    assert event["output_tokens"] == 1
