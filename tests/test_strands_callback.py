"""Tests for the Strands trellar_agent, driven through real Strands Agents/Graphs with a
scripted fake model (no network). Every recorded payload is validated against a
mirror of the backend schema so a would-be HTTP 400 fails here instead."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("strands")

from mcp.types import Tool as MCPTool  # noqa: E402
from strands import Agent, tool  # noqa: E402
from strands.agent.agent_result import AgentResult  # noqa: E402
from strands.multiagent import GraphBuilder  # noqa: E402
from strands.telemetry.metrics import EventLoopMetrics  # noqa: E402
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool  # noqa: E402

from trellar import (  # noqa: E402
    ObservabilityMode,
    evaluate_confidence,
    trellar_strands_agent,
    trellar_strands_single_call,
)
from trellar._context import _current_callback  # noqa: E402
from trellar.callbacks.strands.utils import _llm_human_input  # noqa: E402

from tests.backend_schema import AgentLoopRequest  # noqa: E402
from tests.strands_factories import FakeModel, events_of, text_turn, tool_turn  # noqa: E402


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("tool exploded")


@pytest.fixture(autouse=True)
def _clean_context():
    token = _current_callback.set(None)
    yield
    _current_callback.reset(token)


@pytest.fixture
def mock_post():
    """Patch requests.post used by evaluate_confidence (score 8, no halt)."""
    resp = MagicMock()
    resp.json.return_value = {
        "score": 8, "explanation": "ok", "decision_identifier": "d-1", "should_stop_network": False,
    }
    with patch("trellar.agent_loop.requests.post", return_value=resp) as post:
        yield post


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("TRELLAR_API_KEY", "test-key")


def make_agent(trellar_agent, turns, name="calc", tools=None, system_prompt="You are a calculator."):
    return Agent(
        name=name, model=FakeModel(turns), tools=tools or [], hooks=[trellar_agent],
        system_prompt=system_prompt, callback_handler=None,
    )


def assert_backend_accepts(trellar_agent):
    """Validate events + available_tools exactly as the backend request model would."""
    AgentLoopRequest(
        context=trellar_agent.events,
        trace_id=trellar_agent.trace_id,
        agent_name=trellar_agent.agent_name,
        available_tools=[{"tools_hash": h, "tools": t} for h, t in trellar_agent.available_tools.items()],
    )


class FunctionNode:
    """Tiny non-LLM graph node (like the example's aitl_gate)."""

    def __init__(self, fn):
        self.name = self.id = "function-node"
        self.fn = fn

    async def invoke_async(self, prompt=None, **kwargs):
        self.fn()
        return AgentResult(
            stop_reason="end_turn", message={"role": "assistant", "content": [{"text": "done"}]},
            metrics=EventLoopMetrics(), state={},
        )

    def __call__(self, prompt=None, **kwargs):
        return asyncio.run(self.invoke_async(prompt))

    async def stream_async(self, prompt=None, **kwargs):
        yield {"result": await self.invoke_async(prompt)}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_agent_name_required():
    with pytest.raises(ValueError):
        trellar_strands_agent("  ")


# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------

class TestSingleAgent:
    def test_tool_run_event_sequence_and_links(self):
        trellar_agent = trellar_strands_agent("t")
        agent = make_agent(trellar_agent, [tool_turn("add", {"a": 1, "b": 2}), text_turn("3")], tools=[add])
        agent("what is 1+2?")

        assert [e["event"] for e in trellar_agent.events] == [
            "on_chain_start", "on_chat_model_start", "on_llm_end", "on_tool_start",
            "on_tool_end", "on_chat_model_start", "on_llm_end", "on_chain_end",
        ]
        assert [e["graph_order"] for e in trellar_agent.events] == list(range(1, 9))

        root = trellar_agent.events[0]
        assert root["node_name"] == "calc"
        assert root["parent_run_id"] is None
        assert trellar_agent.trace_id == root["run_id"]
        # llm and tool events hang off the agent's chain
        for e in trellar_agent.events[1:-1]:
            assert e["parent_run_id"] == root["run_id"]
        assert_backend_accepts(trellar_agent)

    def test_model_input_and_name(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [text_turn("hi")], system_prompt="Be brief.")("hello there")

        start = events_of(trellar_agent, "on_chat_model_start")[0]
        assert start["model"] == "fake-model"
        assert start["input"] == {"system": "Be brief.", "human": "hello there"}
        assert start["tools_hash"] is None

    def test_multi_turn_history_is_folded_into_human_input(self):
        trellar_agent = trellar_strands_agent("t")
        agent = make_agent(
            trellar_agent, [text_turn("Hello! How can I help?"), text_turn("KAN-5 details")],
            system_prompt="Be brief.",
        )
        agent("hello")
        agent("give me details about KAN-5")

        starts = events_of(trellar_agent, "on_chat_model_start")
        assert len(starts) == 1  # the trellar_agent resets per root run; this is the 2nd run's call
        assert set(starts[0]["input"]) == {"system", "human"}
        assert starts[0]["input"]["system"] == "Be brief."
        assert starts[0]["input"]["human"] == (
            "Human: hello\n"
            "AI LLM: Hello! How can I help?\n"
            "\n"
            "Current message - give me details about KAN-5"
        )

    def test_tool_events_and_response_folded_into_llm(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [tool_turn("add", {"a": 1, "b": 2}), text_turn("3")], tools=[add])("1+2?")

        tool_start = events_of(trellar_agent, "on_tool_start")[0]
        assert tool_start["tool"] == "add"
        assert tool_start["tool_description"] == "Add two numbers."
        assert tool_start["input"] == {"a": 1, "b": 2}
        assert tool_start["invoked_by_run_id"] == tool_start["parent_run_id"]

        tool_end = events_of(trellar_agent, "on_tool_end")[0]
        assert tool_end["output"] == "3"
        assert tool_end["is_mcp_tool"] is False
        assert tool_end["run_id"] == tool_start["run_id"]

        first_llm = events_of(trellar_agent, "on_llm_end")[0]["output"]["response"]
        assert 'TOOL CALL: add(args={"a": 1, "b": 2})' in first_llm
        assert "TOOL RESPONSE [add]: 3" in first_llm

    def test_available_tools_openai_shape_and_hash(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [tool_turn("add", {"a": 1, "b": 2}), text_turn("3")], tools=[add])("1+2?")

        starts = events_of(trellar_agent, "on_chat_model_start")
        assert len({s["tools_hash"] for s in starts}) == 1  # same toolset -> one entry
        tools_hash = starts[0]["tools_hash"]
        assert list(trellar_agent.available_tools) == [tools_hash]

        schema = trellar_agent.available_tools[tools_hash][0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert schema["function"]["description"] == "Add two numbers."
        assert "a" in schema["function"]["parameters"]["properties"]

    def test_tool_failure_still_produces_valid_payload(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [tool_turn("boom", {}), text_turn("sorry")], tools=[boom])("go")

        errors = events_of(trellar_agent, "on_tool_error")
        assert len(errors) == 1 and "tool exploded" in errors[0]["error"]
        assert_backend_accepts(trellar_agent)

    def test_model_failure_records_error_and_releases_context(self):
        trellar_agent = trellar_strands_agent("t")
        agent = make_agent(trellar_agent, [RuntimeError("model down")])
        with pytest.raises(Exception):
            agent("go")

        err = events_of(trellar_agent, "on_llm_error")[0]
        assert "model down" in err["error"]
        assert _current_callback.get() is None
        assert_backend_accepts(trellar_agent)

    def test_reuse_resets_events(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [text_turn("one")])("first")
        first_trace = trellar_agent.trace_id
        make_agent(trellar_agent, [text_turn("two")])("second")

        assert trellar_agent.trace_id != first_trace
        assert all(e["trace_id"] == trellar_agent.trace_id for e in trellar_agent.events)
        assert len(events_of(trellar_agent, "on_chain_start")) == 1

    def test_build_context_narrative(self):
        trellar_agent = trellar_strands_agent("t")
        make_agent(trellar_agent, [tool_turn("add", {"a": 1, "b": 2}), text_turn("3")], tools=[add])("1+2?")

        context = trellar_agent.build_context()
        assert f"Trace ID: {trellar_agent.trace_id}" in context
        assert "tool: add" in context
        assert "human: 1+2?" in context


# ---------------------------------------------------------------------------
# LLM human input (history folding)
# ---------------------------------------------------------------------------

def _msg(role, *blocks):
    return {"role": role, "content": list(blocks)}


class TestLlmHumanInput:
    def test_no_messages_returns_none(self):
        assert _llm_human_input([]) is None
        assert _llm_human_input(None) is None

    def test_single_turn_is_plain_text(self):
        assert _llm_human_input([_msg("user", {"text": "hi"})]) == "hi"

    def test_multi_turn_history(self):
        messages = [
            _msg("user", {"text": "hello"}),
            _msg("assistant", {"text": "Hello! How can I help?"}),
            _msg("user", {"text": "turn 2"}),
            _msg("assistant", {"text": "reply 2"}),
            _msg("user", {"text": "turn 3"}),
        ]
        assert _llm_human_input(messages) == (
            "Human: hello\n"
            "AI LLM: Hello! How can I help?\n"
            "Human: turn 2\n"
            "AI LLM: reply 2\n"
            "\n"
            "Current message - turn 3"
        )

    def test_tool_use_and_tool_result_messages_are_skipped(self):
        messages = [
            _msg("user", {"text": "turn 1"}),
            _msg("assistant", {"toolUse": {"toolUseId": "1", "name": "add", "input": {}}}),
            _msg("user", {"toolResult": {"toolUseId": "1", "content": [{"text": "3"}]}}),
            _msg("assistant", {"text": "reply 1"}),
            _msg("user", {"text": "turn 2"}),
        ]
        assert _llm_human_input(messages) == (
            "Human: turn 1\nAI LLM: reply 1\n\nCurrent message - turn 2"
        )

    def test_messages_after_last_user_turn_are_ignored(self):
        messages = [
            _msg("user", {"text": "turn 1"}),
            _msg("assistant", {"toolUse": {"toolUseId": "1", "name": "add", "input": {}}}),
            _msg("user", {"toolResult": {"toolUseId": "1", "content": [{"text": "3"}]}}),
        ]
        assert _llm_human_input(messages) == "turn 1"


# ---------------------------------------------------------------------------
# MCP detection
# ---------------------------------------------------------------------------

def test_mcp_tool_is_flagged():
    mcp_tool = MCPTool(
        name="create_ticket", description="Create a ticket",
        inputSchema={"type": "object", "properties": {"symbols": {"type": "string"}}},
    )
    client = MagicMock()
    client.call_tool_async = AsyncMock(
        return_value={"status": "success", "toolUseId": "tu-1", "content": [{"text": "T-123"}]}
    )
    trellar_agent = trellar_strands_agent("t")
    agent = make_agent(
        trellar_agent, [tool_turn("create_ticket", {"symbols": "GM"}), text_turn("ok")],
        tools=[MCPAgentTool(mcp_tool, client), add],
    )
    agent("buy")

    end = events_of(trellar_agent, "on_tool_end")[0]
    assert end["is_mcp_tool"] is True
    assert end["output"] == "T-123"
    assert_backend_accepts(trellar_agent)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(trellar_agent, gate_fn):
    a1 = make_agent(trellar_agent, [tool_turn("add", {"a": 1, "b": 2}), text_turn("3")], name="a1", tools=[add])
    a2 = make_agent(trellar_agent, [text_turn("report")], name="a2")
    builder = GraphBuilder()
    builder.add_node(a1, "a1")
    builder.add_node(FunctionNode(gate_fn), "gate")
    builder.add_node(a2, "a2")
    builder.add_edge("a1", "gate")
    builder.add_edge("gate", "a2")
    builder.set_entry_point("a1")
    builder.set_hook_providers([trellar_agent])
    return builder.build()


class TestGraph:
    def test_hierarchy_graph_node_agent_llm(self):
        trellar_agent = trellar_strands_agent("t")
        build_graph(trellar_agent, lambda: None)("go")

        by_run = {e["run_id"]: e for e in events_of(trellar_agent, "on_chain_start")}
        root = trellar_agent.events[0]
        assert root["parent_run_id"] is None and root["node_type"] == "chain"

        # one root, one chain per graph node, one chain per agent run
        names = [e["node_name"] for e in events_of(trellar_agent, "on_chain_start")]
        assert names.count("a1") == 2 and names.count("a2") == 2 and names.count("gate") == 1

        # every llm/tool event's parent is a chain named after its agent
        llm = events_of(trellar_agent, "on_chat_model_start")[0]
        assert by_run[llm["parent_run_id"]]["node_name"] == "a1"
        tool_start = events_of(trellar_agent, "on_tool_start")[0]
        assert by_run[tool_start["parent_run_id"]]["node_name"] == "a1"

        # agent chain is nested under its graph-node chain, which is under the graph root
        agent_chain = by_run[llm["parent_run_id"]]
        node_chain = by_run[agent_chain["parent_run_id"]]
        assert node_chain["parent_run_id"] == root["run_id"]
        assert_backend_accepts(trellar_agent)

    def test_only_one_trace_and_root_end_last(self):
        trellar_agent = trellar_strands_agent("t")
        build_graph(trellar_agent, lambda: None)("go")

        assert {e["trace_id"] for e in trellar_agent.events} == {trellar_agent.trace_id}
        assert trellar_agent.events[-1]["event"] == "on_chain_end"
        assert trellar_agent.events[-1]["run_id"] == trellar_agent.trace_id

    def test_evaluate_confidence_from_inside_a_node(self, mock_post):
        trellar_agent = trellar_strands_agent("net-name")
        results = []
        build_graph(trellar_agent, lambda: results.append(evaluate_confidence()))("go")

        assert results[0].score == 8
        payload = mock_post.call_args.kwargs["json"]
        assert payload["agent_name"] == "net-name"
        assert payload["trace_id"] == trellar_agent.trace_id
        assert payload["observability_call"] is False
        # the payload sent over HTTP must satisfy the backend schema
        AgentLoopRequest(**payload)
        assert _current_callback.get() is None  # released after the run

    def test_agent_as_tool_nests_under_tool_call(self):
        trellar_agent = trellar_strands_agent("t")
        inner = make_agent(trellar_agent, [text_turn("inner answer")], name="inner")

        @tool
        def ask_inner(question: str) -> str:
            """Ask the inner agent."""
            return str(inner(question))

        outer = make_agent(trellar_agent, [tool_turn("ask_inner", {"question": "q"}), text_turn("done")],
                           name="outer", tools=[ask_inner])
        outer("go")

        tool_start = events_of(trellar_agent, "on_tool_start")[0]
        inner_chain = next(e for e in events_of(trellar_agent, "on_chain_start") if e["node_name"] == "inner")
        assert inner_chain["parent_run_id"] == tool_start["run_id"]
        roots = [e for e in events_of(trellar_agent, "on_chain_start") if e["parent_run_id"] is None]
        assert [r["node_name"] for r in roots] == ["outer"]  # inner is not a second root
        assert_backend_accepts(trellar_agent)


class TestAutoBind:
    """Registering on the Graph alone binds every node's Agent; double registration is a no-op."""

    @staticmethod
    def bare_agent(name, hooks=None):
        return Agent(
            # two turns: the same Agent may be run twice (the fake model errors when out of turns)
            name=name, model=FakeModel([text_turn("ok"), text_turn("ok")]),
            hooks=hooks or [], callback_handler=None,
        )

    @staticmethod
    def graph_of(trellar_agent, *agents):
        builder = GraphBuilder()
        for a in agents:
            builder.add_node(a, a.name)
        for first, second in zip(agents, agents[1:]):
            builder.add_edge(first.name, second.name)
        builder.set_entry_point(agents[0].name)
        builder.set_hook_providers([trellar_agent])
        return builder.build()

    @staticmethod
    def counts(trellar_agent):
        """Number of recorded events per event type (duplicates would inflate these)."""
        names = [e["event"] for e in trellar_agent.events]
        return {n: names.count(n) for n in set(names)}

    def test_graph_hook_alone_records_agent_llm_events(self):
        trellar_agent = trellar_strands_agent("t")
        self.graph_of(trellar_agent, self.bare_agent("a1"), self.bare_agent("a2"))("go")

        assert self.counts(trellar_agent)["on_chat_model_start"] == 2  # one per agent
        assert_backend_accepts(trellar_agent)

    def test_explicit_hook_plus_graph_hook_records_once(self):
        explicit, auto = trellar_strands_agent("t"), trellar_strands_agent("t")
        self.graph_of(explicit, self.bare_agent("a1", hooks=[explicit]), self.bare_agent("a2", hooks=[explicit]))("go")
        self.graph_of(auto, self.bare_agent("a1"), self.bare_agent("a2"))("go")

        # identical to the auto-bound run: nothing recorded twice
        assert self.counts(explicit) == self.counts(auto)

    def test_graph_run_twice_does_not_duplicate(self):
        trellar_agent = trellar_strands_agent("t")
        graph = self.graph_of(trellar_agent, self.bare_agent("a1"), self.bare_agent("a2"))
        graph("go")
        first_counts, first_trace = self.counts(trellar_agent), trellar_agent.trace_id
        graph("again")

        assert self.counts(trellar_agent) == first_counts
        assert trellar_agent.trace_id != first_trace

    def test_agent_in_two_graphs_records_once_per_run(self):
        trellar_agent = trellar_strands_agent("t")
        shared = self.bare_agent("shared")
        self.graph_of(trellar_agent, shared)("go")
        self.graph_of(trellar_agent, shared)("again")  # second graph binds the same Agent again

        assert self.counts(trellar_agent)["on_chat_model_start"] == 1  # this run only, not doubled

    def test_non_agent_node_is_skipped_without_error(self):
        trellar_agent = trellar_strands_agent("t")
        builder = GraphBuilder()
        builder.add_node(FunctionNode(lambda: None), "gate")
        builder.set_entry_point("gate")
        builder.set_hook_providers([trellar_agent])
        builder.build()("go")

        assert [e["node_name"] for e in events_of(trellar_agent, "on_chain_start")] == [
            trellar_agent.events[0]["node_name"], "gate",
        ]


# ---------------------------------------------------------------------------
# Conditional routing (router -> jira | gate -> reporter), like the Jira example
# ---------------------------------------------------------------------------

@tool
def route_to_jira_agent() -> str:
    """Route to the jira agent."""
    return "route:jira"


@tool
def route_to_send_email_agent() -> str:
    """Route to the email path."""
    return "route:send_email"


@tool
def flag_email_report_request() -> str:
    """Flag that an email report was requested."""
    return "email_report_requested"


def build_routing_graph(trellar_agent, route_tool, jira_flags_email):
    router = make_agent(trellar_agent, [tool_turn(route_tool, {}), text_turn("routed")], name="router_agent",
                        tools=[route_to_jira_agent, route_to_send_email_agent])
    jira_turns = ([tool_turn("flag_email_report_request", {}), text_turn("preparing report")]
                  if jira_flags_email else [text_turn("3 open issues")])
    jira = make_agent(trellar_agent, jira_turns, name="jira_react_agent", tools=[flag_email_report_request])
    reporter = make_agent(trellar_agent, [text_turn("Dear user, ...")], name="reporter_agent")

    called = lambda s, node: set(s.results[node].result.metrics.tool_metrics)
    email_only = lambda s: "route_to_send_email_agent" in called(s, "router_agent")

    builder = GraphBuilder()
    builder.add_node(router, "router_agent")
    builder.add_node(jira, "jira_react_agent")
    builder.add_node(FunctionNode(lambda: evaluate_confidence()), "aitl_email_gate")
    builder.add_node(reporter, "reporter_agent")
    builder.add_edge("router_agent", "jira_react_agent", condition=lambda s: not email_only(s))
    builder.add_edge("router_agent", "aitl_email_gate", condition=email_only)
    builder.add_edge("jira_react_agent", "aitl_email_gate",
                     condition=lambda s: "flag_email_report_request" in called(s, "jira_react_agent"))
    builder.add_edge("aitl_email_gate", "reporter_agent")
    builder.set_entry_point("router_agent")
    builder.set_hook_providers([trellar_agent])
    return builder.build()


def agent_chain_names(trellar_agent):
    """Names of chains that are agents (graph + node chains share a name pair, so dedupe)."""
    return {e["node_name"] for e in events_of(trellar_agent, "on_chain_start")}


class TestConditionalRouting:
    def test_email_only_skips_jira(self, mock_post):
        trellar_agent = trellar_strands_agent("t")
        build_routing_graph(trellar_agent, "route_to_send_email_agent", jira_flags_email=False)("email me")

        names = agent_chain_names(trellar_agent)
        assert {"router_agent", "aitl_email_gate", "reporter_agent"} <= names
        assert "jira_react_agent" not in names
        assert mock_post.call_count == 1  # the gate evaluated once
        assert_backend_accepts(trellar_agent)

    def test_jira_without_email_skips_gate_and_reporter(self, mock_post):
        trellar_agent = trellar_strands_agent("t")
        build_routing_graph(trellar_agent, "route_to_jira_agent", jira_flags_email=False)("how many issues?")

        names = agent_chain_names(trellar_agent)
        assert "jira_react_agent" in names
        assert not ({"aitl_email_gate", "reporter_agent"} & names)
        mock_post.assert_not_called()
        assert_backend_accepts(trellar_agent)

    def test_jira_with_email_flag_goes_through_gate(self, mock_post):
        trellar_agent = trellar_strands_agent("t")
        build_routing_graph(trellar_agent, "route_to_jira_agent", jira_flags_email=True)("issues, then email me")

        names = agent_chain_names(trellar_agent)
        assert {"router_agent", "jira_react_agent", "aitl_email_gate", "reporter_agent"} <= names
        order = [e["node_name"] for e in events_of(trellar_agent, "on_chain_start")]
        assert order.index("jira_react_agent") < order.index("aitl_email_gate") < order.index("reporter_agent")
        assert mock_post.call_count == 1
        assert_backend_accepts(trellar_agent)


# ---------------------------------------------------------------------------
# ObservabilityMode
# ---------------------------------------------------------------------------

class TestObservabilityMode:
    def test_none_never_auto_evaluates(self, mock_post):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.NONE)
        make_agent(trellar_agent, [text_turn("hi")])("go")
        mock_post.assert_not_called()

    def test_always_evaluates_at_root_end(self, mock_post):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.ALWAYS)
        build_graph(trellar_agent, lambda: None)("go")

        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        assert payload["observability_call"] is True
        # the final root chain_end is already recorded when the call is made
        assert payload["context"][-1]["event"] == "on_chain_end"
        AgentLoopRequest(**payload)

    def test_if_not_evaluated_skips_after_manual_call(self, mock_post):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.IF_NOT_EVALUATED)
        build_graph(trellar_agent, lambda: evaluate_confidence())("go")
        assert mock_post.call_count == 1  # only the manual call

    def test_if_not_evaluated_runs_when_no_manual_call(self, mock_post):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.IF_NOT_EVALUATED)
        build_graph(trellar_agent, lambda: None)("go")
        assert mock_post.call_count == 1

    def test_auto_evaluate_failure_is_swallowed(self):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.ALWAYS)
        with patch("trellar.agent_loop.requests.post", side_effect=RuntimeError("backend down")):
            result = make_agent(trellar_agent, [text_turn("hi")])("go")
        assert "hi" in str(result)


# ---------------------------------------------------------------------------
# Single call (one Agent, no Graph)
# ---------------------------------------------------------------------------

class TestSingleCall:
    def test_payload_flag_and_backend_schema(self, mock_post):
        trellar_agent = trellar_strands_single_call("t", ObservabilityMode.ALWAYS)
        make_agent(trellar_agent, [text_turn("hi")])("go")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["single_call"] is True
        AgentLoopRequest(**payload)

    def test_regular_trellar_agent_is_not_single_call(self, mock_post):
        trellar_agent = trellar_strands_agent("t", ObservabilityMode.ALWAYS)
        make_agent(trellar_agent, [text_turn("hi")])("go")
        assert mock_post.call_args.kwargs["json"]["single_call"] is False

    def test_result_is_stored_on_trellar_agent(self, mock_post):
        trellar_agent = trellar_strands_single_call("t", ObservabilityMode.ALWAYS)
        make_agent(trellar_agent, [text_turn("hi")])("go")

        assert trellar_agent.trellar_evaluate_result.score == 8
        assert trellar_agent.trellar_evaluate_error is None
        assert _current_callback.get() is None  # released after the call

    def test_backend_error_is_stored_not_raised(self):
        trellar_agent = trellar_strands_single_call("t", ObservabilityMode.ALWAYS)
        with patch("trellar.agent_loop.requests.post", side_effect=RuntimeError("backend down")):
            make_agent(trellar_agent, [text_turn("hi")])("go")

        assert trellar_agent.trellar_evaluate_result is None
        assert isinstance(trellar_agent.trellar_evaluate_error, RuntimeError)

    def test_second_call_resets_result(self, mock_post):
        trellar_agent = trellar_strands_single_call("t", ObservabilityMode.ALWAYS)
        make_agent(trellar_agent, [text_turn("hi")])("go")
        first = trellar_agent.trellar_evaluate_result

        # Second run fails: the old result must not leak through.
        with patch("trellar.agent_loop.requests.post", side_effect=RuntimeError("down")):
            make_agent(trellar_agent, [text_turn("hi")])("go again")
        assert first is not None
        assert trellar_agent.trellar_evaluate_result is None

    def test_none_mode_does_not_evaluate(self, mock_post):
        trellar_agent = trellar_strands_single_call("t")
        make_agent(trellar_agent, [text_turn("hi")])("go")

        mock_post.assert_not_called()
        assert trellar_agent.trellar_evaluate_result is None
