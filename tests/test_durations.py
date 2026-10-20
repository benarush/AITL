"""Tests for ``duration_seconds`` on model/tool end events (LangChain + Strands).

LangChain tests drive the callback directly with a fake clock so the expected
durations are exact. The Strands test runs a real Agent with a scripted model.
"""
from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import HumanMessage

from trellar.callbacks import _common
from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback

from tests.factories import make_ai_message, make_llm_result


class FakeClock:
    """Stands in for the ``time`` module inside trellar.callbacks._common."""

    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(_common, "time", fake)
    return fake


@pytest.fixture
def handler() -> _LangchainAgentCallback:
    """Callback with a started root run so the per-run state is initialised."""
    cb = _LangchainAgentCallback(agent_name="test-agent")
    cb.on_chain_start({"name": "graph"}, {"messages": []}, run_id=uuid.uuid4())
    return cb


def _event(cb: _LangchainAgentCallback, name: str, run_id: uuid.UUID) -> dict:
    return next(e for e in cb.events if e["event"] == name and e["run_id"] == str(run_id))


def test_langchain_tool_duration(handler, clock):
    run_id = uuid.uuid4()
    handler.on_tool_start({"name": "search"}, "q", run_id=run_id)
    clock.advance(2.5)
    handler.on_tool_end("result", run_id=run_id)

    assert "duration_seconds" not in _event(handler, "on_tool_start", run_id)
    assert _event(handler, "on_tool_end", run_id)["duration_seconds"] == 2.5
    assert handler._start_times == {}  # nothing left behind

    # The text context shown to the evaluator includes it too.
    assert "duration: 2.5s" in handler.build_context()


def test_langchain_model_duration_on_success_and_error(handler, clock):
    ok_id, err_id = uuid.uuid4(), uuid.uuid4()

    handler.on_chat_model_start({"name": "gpt"}, [[HumanMessage(content="hi")]], run_id=ok_id)
    clock.advance(1.25)
    handler.on_llm_end(make_llm_result(message=make_ai_message("hello")), run_id=ok_id)

    handler.on_chat_model_start({"name": "gpt"}, [[HumanMessage(content="hi")]], run_id=err_id)
    clock.advance(0.5)
    handler.on_llm_error(RuntimeError("boom"), run_id=err_id)

    assert _event(handler, "on_llm_end", ok_id)["duration_seconds"] == 1.25
    assert _event(handler, "on_llm_error", err_id)["duration_seconds"] == 0.5


def test_langchain_overlapping_calls_get_independent_durations(handler, clock):
    """Two parallel calls to the same tool, and a model call in between."""
    tool_a, tool_b, model = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    handler.on_tool_start({"name": "search"}, "a", run_id=tool_a)
    clock.advance(1)
    handler.on_tool_start({"name": "search"}, "b", run_id=tool_b)  # starts while A runs
    handler.on_chat_model_start({"name": "gpt"}, [[HumanMessage(content="hi")]], run_id=model)
    clock.advance(2)
    handler.on_tool_end("b-out", run_id=tool_b)  # B: 2s
    clock.advance(3)
    handler.on_llm_end(make_llm_result(message=make_ai_message("ok")), run_id=model)  # model: 5s
    clock.advance(4)
    handler.on_tool_end("a-out", run_id=tool_a)  # A: 10s

    assert _event(handler, "on_tool_end", tool_b)["duration_seconds"] == 2
    assert _event(handler, "on_llm_end", model)["duration_seconds"] == 5
    assert _event(handler, "on_tool_end", tool_a)["duration_seconds"] == 10
    assert handler._start_times == {}


def test_strands_model_and_tool_duration():
    pytest.importorskip("strands")
    from strands import Agent, tool

    from trellar import trellar_strands_agent
    from tests.strands_factories import FakeModel, events_of, text_turn, tool_turn

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    trellar_agent = trellar_strands_agent("t")
    agent = Agent(
        name="calc",
        model=FakeModel([tool_turn("add", {"a": 1, "b": 2}), text_turn("3")]),
        tools=[add],
        hooks=[trellar_agent],
        callback_handler=None,
    )
    agent("1+2?")

    for name in ("on_llm_end", "on_tool_end"):
        events = events_of(trellar_agent, name)
        assert events, name
        for e in events:
            assert isinstance(e["duration_seconds"], float) and e["duration_seconds"] >= 0
    for name in ("on_chat_model_start", "on_tool_start"):
        assert all("duration_seconds" not in e for e in events_of(trellar_agent, name))
    assert trellar_agent._start_times == {}
