from __future__ import annotations

import json
import uuid
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from trellar import ObservabilityMode
from trellar._context import _current_callback
from trellar.callbacks.langchain_callback import _AgentGuardCallback

from tests.factories import FakeGeneration, FakeMessage, FakeResponse, make_ai_message, make_llm_result


class _FakeTriage(BaseModel):
    """Mirrors the shape of a `with_structured_output()` return value."""

    severity: str
    affected_service: str


# ---------------------------------------------------------------------------
# on_chain_start — must not crash on non-dict inputs
# ---------------------------------------------------------------------------

class TestOnChainStartNonDictInputs:
    def test_list_of_messages_input_does_not_raise(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start({}, [HumanMessage(content="hi")], run_id=run_id, parent_run_id=uuid.uuid4())

        event = active_handler.events[-1]
        assert event["event"] == "on_chain_start"
        assert event["inputs"] == ["HUMAN MESSAGE: hi"]
        json.dumps(event)  # must not raise

    def test_bare_ai_message_input_does_not_raise(self, active_handler):
        """Reproduces the `PydanticOutputParser` sub-chain from
        `with_structured_output()`, whose raw input is a single AIMessage."""
        run_id = uuid.uuid4()
        ai_message = AIMessage(content="", tool_calls=[
            {"name": "check_service_health", "args": {"service_name": "checkout"}, "id": "call_1"}
        ])
        active_handler.on_chain_start({}, ai_message, run_id=run_id, parent_run_id=uuid.uuid4())

        event = active_handler.events[-1]
        assert event["event"] == "on_chain_start"
        # The backend requires `inputs` to always be a list — a bare message
        # must be wrapped, never sent as a raw string.
        assert isinstance(event["inputs"], list)
        json.dumps(event)  # must not raise

    def test_inputs_field_is_always_a_list(self, active_handler):
        """The backend's schema requires `on_chain_start.inputs` to be a list
        no matter what shape the raw LangChain input takes."""
        for raw_input in (
            {"messages": [HumanMessage(content="hi")]},
            [HumanMessage(content="hi")],
            AIMessage(content="hi"),
            "some raw string input",
            {"no_messages_key": True},
        ):
            active_handler.on_chain_start({}, raw_input, run_id=uuid.uuid4(), parent_run_id=uuid.uuid4())
            assert isinstance(active_handler.events[-1]["inputs"], list)

    def test_dict_input_with_messages_still_uses_serialize_messages(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start(
            {}, {"messages": [HumanMessage(content="hello")]}, run_id=run_id, parent_run_id=uuid.uuid4()
        )

        event = active_handler.events[-1]
        assert event["inputs"] == ["HUMAN MESSAGE: hello"]


# ---------------------------------------------------------------------------
# on_chain_end — must not crash / leak non-JSON-serializable objects
# ---------------------------------------------------------------------------

class TestOnChainEndNonDictOutputs:
    def test_pydantic_model_output_does_not_raise(self, active_handler):
        """Reproduces the reported crash: a `with_structured_output()` sub-chain's
        `outputs` is the raw pydantic model, not a dict."""
        run_id = uuid.uuid4()
        triage = _FakeTriage(severity="high", affected_service="checkout-service")

        active_handler.on_chain_end(triage, run_id=run_id, parent_run_id=uuid.uuid4())

        event = active_handler.events[-1]
        assert event["event"] == "on_chain_end"
        json.dumps(event)  # must not raise
        assert event["outputs"] == {"severity": "high", "affected_service": "checkout-service"}

    def test_dict_output_with_messages_still_uses_serialize_messages(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_end(
            {"messages": [AIMessage(content="done")]}, run_id=run_id, parent_run_id=uuid.uuid4()
        )

        event = active_handler.events[-1]
        assert event["outputs"]["messages"] == ["AI MESSAGE: done"]


# ---------------------------------------------------------------------------
# Full round-trip — the exact check that would have caught the reported crash
# ---------------------------------------------------------------------------

class TestEventsAreAlwaysJsonSerializable:
    def test_full_tool_calling_sequence_is_json_serializable(self, active_handler):
        trace_run = uuid.uuid4()
        llm_run = uuid.uuid4()
        tool_run = uuid.uuid4()
        structured_run = uuid.uuid4()

        active_handler.on_chain_start({}, {"messages": []}, run_id=trace_run, parent_run_id=None)

        # AIMessage requesting a tool call — empty content, populated tool_calls.
        ai_message = AIMessage(
            content="",
            tool_calls=[{"name": "check_service_health", "args": {"service_name": "checkout"}, "id": "call_1"}],
        )
        active_handler.on_chain_start({}, ai_message, run_id=llm_run, parent_run_id=trace_run)

        active_handler.on_tool_start(
            {"name": "check_service_health"}, "{}", run_id=tool_run, parent_run_id=llm_run,
            inputs={"service_name": "checkout"},
        )
        active_handler.on_tool_end("cpu=92%", run_id=tool_run, parent_run_id=llm_run)

        # Structured-output sub-chain returning a raw pydantic model.
        triage = _FakeTriage(severity="high", affected_service="checkout-service")
        active_handler.on_chain_end(triage, run_id=structured_run, parent_run_id=trace_run)

        active_handler.on_chain_end({"messages": [ai_message]}, run_id=trace_run, parent_run_id=None)

        # This is exactly what evaluate_confidence() would try to send.
        json.dumps({"context": active_handler.events, "trace_id": str(active_handler.trace_id)})


# ---------------------------------------------------------------------------
# _to_jsonable — static coercion helper
# ---------------------------------------------------------------------------

class TestToJsonable:
    def test_primitives_pass_through(self):
        for value in (None, "str", 1, 1.5, True, False):
            assert _AgentGuardCallback._to_jsonable(value) == value

    def test_message_like_object_uses_serialize_message_obj(self):
        msg = AIMessage(content="hi")
        assert _AgentGuardCallback._to_jsonable(msg) == "AI MESSAGE: hi"

    def test_pydantic_model_uses_model_dump(self):
        model = _FakeTriage(severity="high", affected_service="checkout")
        assert _AgentGuardCallback._to_jsonable(model) == {
            "severity": "high",
            "affected_service": "checkout",
        }

    def test_nested_dict_and_list_are_recursively_coerced(self):
        value = {"a": [1, 2, {"b": AIMessage(content="hi")}]}
        result = _AgentGuardCallback._to_jsonable(value)
        assert result == {"a": [1, 2, {"b": "AI MESSAGE: hi"}]}

    def test_tuple_is_converted_to_list(self):
        assert _AgentGuardCallback._to_jsonable((1, 2, 3)) == [1, 2, 3]

    def test_unserializable_object_falls_back_to_str(self):
        class Unserializable:
            def __repr__(self):
                return "<Unserializable>"

        assert _AgentGuardCallback._to_jsonable(Unserializable()) == "<Unserializable>"


# ---------------------------------------------------------------------------
# _serialize_message_obj / _serialize_messages — static message formatting
# ---------------------------------------------------------------------------

class TestSerializeMessageObjAndMessages:
    @pytest.mark.parametrize(
        "message,expected_prefix",
        [
            (AIMessage(content="hi"), "AI MESSAGE"),
            (HumanMessage(content="hi"), "HUMAN MESSAGE"),
            (SystemMessage(content="hi"), "SYSTEM MESSAGE"),
            (ToolMessage(content="hi", tool_call_id="1"), "TOOL MESSAGE"),
        ],
    )
    def test_known_message_type_prefixes(self, message, expected_prefix):
        assert _AgentGuardCallback._serialize_message_obj(message) == f"{expected_prefix}: hi"

    def test_unknown_message_type_falls_back_to_generic_prefix(self):
        class FakeMsg:
            type = "weird"
            content = "hello"

        assert _AgentGuardCallback._serialize_message_obj(FakeMsg()) == "MESSAGE: hello"

    def test_non_message_falls_back_to_str(self):
        assert _AgentGuardCallback._serialize_message_obj("plain string") == "plain string"
        assert _AgentGuardCallback._serialize_message_obj(42) == "42"

    def test_non_str_content_is_json_encoded(self):
        msg = AIMessage(content=[{"type": "text", "text": "hi"}])
        result = _AgentGuardCallback._serialize_message_obj(msg)
        assert result.startswith("AI MESSAGE: ")
        assert json.loads(result[len("AI MESSAGE: "):]) == [{"type": "text", "text": "hi"}]

    def test_serialize_messages_returns_list_of_labeled_strings(self):
        messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        result = _AgentGuardCallback._serialize_messages(messages)
        assert result == ["SYSTEM MESSAGE: sys", "HUMAN MESSAGE: hi"]

    def test_serialize_messages_handles_non_message_items(self):
        assert _AgentGuardCallback._serialize_messages(["raw", 1]) == ["raw", "1"]


# ---------------------------------------------------------------------------
# on_llm_start / on_chat_model_start
# ---------------------------------------------------------------------------

class TestOnLlmStart:
    def test_records_model_and_joined_prompts(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_llm_start(
            {"kwargs": {"model": "gpt-4o"}}, ["prompt one", "prompt two"],
            run_id=run_id, parent_run_id=None,
        )
        event = active_handler.events[-1]
        assert event["event"] == "on_llm_start"
        assert event["model"] == "gpt-4o"
        assert event["input"] == {"system": None, "human": "prompt one\nprompt two"}

    def test_registers_node_as_llm_type(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_llm_start(
            {"kwargs": {"model": "gpt-4o"}}, ["hi"], run_id=run_id, parent_run_id=None
        )
        assert active_handler._run_registry[str(run_id)] == {"name": "gpt-4o", "type": "llm"}


class TestOnChatModelStart:
    def test_extracts_system_and_human_from_first_batch(self, active_handler):
        run_id = uuid.uuid4()
        messages = [[SystemMessage(content="sys"), HumanMessage(content="hi")]]
        active_handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, messages, run_id=run_id, parent_run_id=None
        )
        event = active_handler.events[-1]
        assert event["model"] == "gemini-2.5-flash"
        assert event["input"] == {"system": "sys", "human": "hi"}

    def test_only_first_message_batch_is_used(self, active_handler):
        run_id = uuid.uuid4()
        messages = [
            [HumanMessage(content="batch one")],
            [HumanMessage(content="batch two, should be ignored")],
        ]
        active_handler.on_chat_model_start(
            {"kwargs": {"model": "m"}}, messages, run_id=run_id, parent_run_id=None
        )
        assert active_handler.events[-1]["input"]["human"] == "batch one"

    def test_empty_messages_list_does_not_raise(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chat_model_start(
            {"kwargs": {"model": "m"}}, [], run_id=run_id, parent_run_id=None
        )
        assert active_handler.events[-1]["input"] == {"system": None, "human": None}

    def test_registers_node_as_llm_type(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash"}}, [[HumanMessage(content="hi")]],
            run_id=run_id, parent_run_id=None,
        )
        assert active_handler._run_registry[str(run_id)]["type"] == "llm"


# ---------------------------------------------------------------------------
# on_llm_end — text extraction, tool-call folding, token usage
# ---------------------------------------------------------------------------

class TestOnLlmEnd:
    def test_chat_generation_message_content_happy_path(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(message=make_ai_message(content="hello there"))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "hello there"

    def test_plain_generation_text_fallback(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(text="plain llm text")
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "plain llm text"

    def test_message_with_none_content_falls_back_to_generation_text(self, active_handler):
        run_id = uuid.uuid4()
        gen = FakeGeneration(message=FakeMessage(content=None), text="fallback from .text")
        response = FakeResponse(generations=[[gen]])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "fallback from .text"

    def test_gemini_multimodal_list_content_is_json_encoded(self, active_handler):
        run_id = uuid.uuid4()
        content = [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]
        result = make_llm_result(message=make_ai_message(content=content))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        response_text = active_handler.events[-1]["output"]["response"]
        assert json.loads(response_text) == content

    def test_generation_text_field_as_list_is_json_encoded(self, active_handler):
        """`.text` can itself be a list (Gemini multimodal fallback path)."""
        run_id = uuid.uuid4()
        gen = FakeGeneration(message=None, text=[{"type": "text", "text": "part"}])
        response = FakeResponse(generations=[[gen]])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        response_text = active_handler.events[-1]["output"]["response"]
        assert json.loads(response_text) == [{"type": "text", "text": "part"}]

    def test_phoenix_dict_generation_with_message_dict(self, active_handler):
        run_id = uuid.uuid4()
        gen_dict = {"message": {"content": "phoenix content", "tool_calls": []}}
        response = FakeResponse(generations=[[gen_dict]])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "phoenix content"

    def test_phoenix_dict_generation_with_plain_text_key(self, active_handler):
        """The `gen.get("text", "")` fallback only triggers when `message` is
        present but is *not* itself a dict (e.g. a bare string) -- when
        `message` is absent entirely, `msg_dict` defaults to `{}` (still a
        dict), so the "message dict" branch runs instead, extracting a `None`
        content."""
        run_id = uuid.uuid4()
        gen_dict = {"message": "not-a-dict-value", "text": "phoenix plain text"}
        response = FakeResponse(generations=[[gen_dict]])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "phoenix plain text"

    def test_empty_generations_list_does_not_raise(self, active_handler):
        run_id = uuid.uuid4()
        response = FakeResponse(generations=[])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == ""

    def test_empty_first_batch_does_not_raise(self, active_handler):
        run_id = uuid.uuid4()
        response = FakeResponse(generations=[[]])
        active_handler.on_llm_end(response, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == ""

    def test_token_usage_from_token_usage_key(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(message=make_ai_message(content="hi"), token_usage={"total_tokens": 42})
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["token_usage"] == {"total_tokens": 42}

    def test_token_usage_from_usage_key_fallback(self, active_handler):
        run_id = uuid.uuid4()
        result = LLMResult(
            generations=[[ChatGeneration(message=make_ai_message(content="hi"))]],
            llm_output={"usage": {"total_tokens": 7}},
        )
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["token_usage"] == {"total_tokens": 7}

    def test_no_token_usage_is_none(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(message=make_ai_message(content="hi"))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["token_usage"] is None

    def test_tool_calls_folded_into_response_text(self, active_handler):
        run_id = uuid.uuid4()
        tool_calls = [{"name": "check_stock", "args": {"chip": "AX-9200"}, "id": "call_1"}]
        result = make_llm_result(message=make_ai_message(content="Checking stock...", tool_calls=tool_calls))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        response_text = active_handler.events[-1]["output"]["response"]
        assert "Checking stock..." in response_text
        assert "TOOL CALL: check_stock(args=" in response_text
        assert '"chip": "AX-9200"' in response_text

    def test_tool_calls_with_empty_response_text_still_records_call_lines(self, active_handler):
        run_id = uuid.uuid4()
        tool_calls = [{"name": "f", "args": {}, "id": "1"}]
        result = make_llm_result(message=make_ai_message(content="", tool_calls=tool_calls))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["output"]["response"] == "TOOL CALL: f(args={})"

    def test_multiple_tool_calls_each_get_a_call_line(self, active_handler):
        run_id = uuid.uuid4()
        tool_calls = [
            {"name": "tool_a", "args": {"x": 1}, "id": "1"},
            {"name": "tool_b", "args": {"y": 2}, "id": "2"},
        ]
        result = make_llm_result(message=make_ai_message(content="", tool_calls=tool_calls))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        response_text = active_handler.events[-1]["output"]["response"]
        assert "TOOL CALL: tool_a" in response_text
        assert "TOOL CALL: tool_b" in response_text

    def test_tool_calls_register_pending_entry(self, active_handler):
        run_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        tool_calls = [{"name": "f", "args": {}, "id": "1"}]
        result = make_llm_result(message=make_ai_message(content="", tool_calls=tool_calls))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=parent_id)

        assert len(active_handler._pending_llm_tool_calls) == 1
        pending = active_handler._pending_llm_tool_calls[0]
        assert pending["remaining_tools"] == ["f"]
        assert pending["parent_run_id"] == str(parent_id)
        assert pending["event"] is active_handler.events[-1]

    def test_no_tool_calls_means_no_pending_entry(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(message=make_ai_message(content="hi"))
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        assert active_handler._pending_llm_tool_calls == []


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------

class TestErrorEvents:
    def test_on_llm_error_records_error_string(self, active_handler):
        active_handler.on_llm_error(ValueError("llm boom"), run_id=uuid.uuid4(), parent_run_id=None)
        event = active_handler.events[-1]
        assert event["event"] == "on_llm_error"
        assert event["error"] == "llm boom"

    def test_on_tool_error_records_error_string(self, active_handler):
        active_handler.on_tool_error(ConnectionError("tool boom"), run_id=uuid.uuid4(), parent_run_id=None)
        event = active_handler.events[-1]
        assert event["event"] == "on_tool_error"
        assert event["error"] == "tool boom"

    def test_on_chain_error_records_error_string(self, active_handler):
        active_handler.on_chain_error(RuntimeError("chain boom"), run_id=uuid.uuid4(), parent_run_id=uuid.uuid4())
        event = active_handler.events[-1]
        assert event["event"] == "on_chain_error"
        assert event["error"] == "chain boom"


# ---------------------------------------------------------------------------
# on_tool_start
# ---------------------------------------------------------------------------

class TestOnToolStart:
    def test_records_tool_name_and_description(self, active_handler):
        run_id = uuid.uuid4()
        parent_id = uuid.uuid4()
        active_handler.on_tool_start(
            {"name": "check_inventory_stock", "description": "Check ERP stock."},
            '{"chip_model": "AX-9200"}', run_id=run_id, parent_run_id=parent_id,
            inputs={"chip_model": "AX-9200"},
        )
        event = active_handler.events[-1]
        assert event["tool"] == "check_inventory_stock"
        assert event["tool_description"] == "Check ERP stock."
        assert event["input"] == {"raw": '{"chip_model": "AX-9200"}', "parsed": {"chip_model": "AX-9200"}}
        assert event["invoked_by_run_id"] == str(parent_id)

    def test_registers_node_as_tool_type(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "my_tool"}, "{}", run_id=run_id, parent_run_id=None)
        assert active_handler._run_registry[str(run_id)] == {"name": "my_tool", "type": "tool"}

    def test_missing_parsed_inputs_defaults_to_none(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "my_tool"}, "raw input", run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["input"]["parsed"] is None

    def test_no_parent_run_id_gives_none_invoked_by(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "my_tool"}, "{}", run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["invoked_by_run_id"] is None


# ---------------------------------------------------------------------------
# on_tool_end / _attach_tool_response_to_llm
# ---------------------------------------------------------------------------

class TestOnToolEndAttachment:
    def test_single_pending_call_matching_parent_gets_response_appended(self, active_handler):
        parent_id = uuid.uuid4()
        llm_run = uuid.uuid4()
        tool_run = uuid.uuid4()

        result = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": "check_stock", "args": {}, "id": "1"}])
        )
        active_handler.on_llm_end(result, run_id=llm_run, parent_run_id=parent_id)
        llm_event = active_handler.events[-1]

        active_handler.on_tool_start({"name": "check_stock"}, "{}", run_id=tool_run, parent_run_id=parent_id)
        active_handler.on_tool_end("stock: 8400 units", run_id=tool_run, parent_run_id=parent_id)

        assert "TOOL RESPONSE [check_stock]: stock: 8400 units" in llm_event["output"]["response"]
        assert active_handler._pending_llm_tool_calls == []

    def test_multiple_tool_calls_only_removed_once_all_resolved(self, active_handler):
        parent_id = uuid.uuid4()
        llm_run = uuid.uuid4()
        tool_calls = [{"name": "tool_a", "args": {}, "id": "1"}, {"name": "tool_b", "args": {}, "id": "2"}]
        result = make_llm_result(message=make_ai_message(content="", tool_calls=tool_calls))
        active_handler.on_llm_end(result, run_id=llm_run, parent_run_id=parent_id)
        llm_event = active_handler.events[-1]

        tool_a_run = uuid.uuid4()
        active_handler.on_tool_start({"name": "tool_a"}, "{}", run_id=tool_a_run, parent_run_id=parent_id)
        active_handler.on_tool_end("result a", run_id=tool_a_run, parent_run_id=parent_id)

        assert len(active_handler._pending_llm_tool_calls) == 1  # tool_b still pending
        assert "TOOL RESPONSE [tool_a]: result a" in llm_event["output"]["response"]

        tool_b_run = uuid.uuid4()
        active_handler.on_tool_start({"name": "tool_b"}, "{}", run_id=tool_b_run, parent_run_id=parent_id)
        active_handler.on_tool_end("result b", run_id=tool_b_run, parent_run_id=parent_id)

        assert active_handler._pending_llm_tool_calls == []
        assert "TOOL RESPONSE [tool_b]: result b" in llm_event["output"]["response"]

    def test_two_concurrent_branches_different_tools_do_not_cross_wire(self, active_handler):
        """Mirrors Inventory vs Logistics running in the same LangGraph
        superstep: each branch is its own chain (distinct parent_run_id)
        calling a different tool. Even with interleaved firing order, each
        tool's response must attach to its own branch's LLM event only."""
        branch_a_parent = uuid.uuid4()
        branch_b_parent = uuid.uuid4()
        llm_a = uuid.uuid4()
        llm_b = uuid.uuid4()
        tool_a = uuid.uuid4()
        tool_b = uuid.uuid4()

        result_a = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": "check_inventory_stock", "args": {}, "id": "1"}])
        )
        result_b = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": "find_alternate_routes", "args": {}, "id": "2"}])
        )

        # Both LLM turns finish before either tool starts.
        active_handler.on_llm_end(result_a, run_id=llm_a, parent_run_id=branch_a_parent)
        event_a = active_handler.events[-1]
        active_handler.on_llm_end(result_b, run_id=llm_b, parent_run_id=branch_b_parent)
        event_b = active_handler.events[-1]

        active_handler.on_tool_start({"name": "check_inventory_stock"}, "{}", run_id=tool_a, parent_run_id=branch_a_parent)
        active_handler.on_tool_start({"name": "find_alternate_routes"}, "{}", run_id=tool_b, parent_run_id=branch_b_parent)

        # Finish B's tool first, then A's -- out of order relative to start.
        active_handler.on_tool_end("route info", run_id=tool_b, parent_run_id=branch_b_parent)
        active_handler.on_tool_end("stock info", run_id=tool_a, parent_run_id=branch_a_parent)

        assert "TOOL RESPONSE [find_alternate_routes]: route info" in event_b["output"]["response"]
        assert "route info" not in event_a["output"]["response"]
        assert "TOOL RESPONSE [check_inventory_stock]: stock info" in event_a["output"]["response"]
        assert "stock info" not in event_b["output"]["response"]

    def test_ambiguous_same_tool_name_different_parents_uses_most_recent_fallback(self, active_handler):
        """Documents current behavior: when a tool_end's parent_run_id doesn't
        exactly match any pending entry (e.g. the tool runs under a shared
        ToolNode run_id, different from either LLM's parent), the match falls
        back to *any* pending entry still awaiting that tool name -- the most
        recently added one wins. If two concurrent branches call a tool with
        the same name, this can attach the response to the wrong branch's LLM
        event. This is the callback-side analogue of the cross-wiring
        documented in trellar-parallel-execution-bug-investigation.md."""
        branch_a_parent = uuid.uuid4()
        branch_b_parent = uuid.uuid4()
        llm_a = uuid.uuid4()
        llm_b = uuid.uuid4()

        shared_tool_name = "shared_tool"
        result_a = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": shared_tool_name, "args": {}, "id": "1"}])
        )
        result_b = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": shared_tool_name, "args": {}, "id": "2"}])
        )

        active_handler.on_llm_end(result_a, run_id=llm_a, parent_run_id=branch_a_parent)
        event_a = active_handler.events[-1]
        active_handler.on_llm_end(result_b, run_id=llm_b, parent_run_id=branch_b_parent)
        event_b = active_handler.events[-1]

        # Tool executes under a *different* parent than either LLM call --
        # no exact parent_run_id match exists.
        tool_run = uuid.uuid4()
        unrelated_parent = uuid.uuid4()
        active_handler.on_tool_start({"name": shared_tool_name}, "{}", run_id=tool_run, parent_run_id=unrelated_parent)
        active_handler.on_tool_end("ambiguous result", run_id=tool_run, parent_run_id=unrelated_parent)

        attached_to_a = "ambiguous result" in event_a["output"]["response"]
        attached_to_b = "ambiguous result" in event_b["output"]["response"]
        assert attached_to_a != attached_to_b, "response should attach to exactly one branch"
        assert attached_to_b, (
            "current implementation walks _pending_llm_tool_calls in reverse "
            "(most recently added first) when no exact parent_run_id match exists"
        )

    def test_no_pending_calls_is_a_noop(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "orphan_tool"}, "{}", run_id=run_id, parent_run_id=None)
        active_handler.on_tool_end("some output", run_id=run_id, parent_run_id=None)
        assert active_handler.events[-1]["event"] == "on_tool_end"

    def test_unmatched_tool_name_leaves_pending_calls_untouched(self, active_handler):
        parent_id = uuid.uuid4()
        llm_run = uuid.uuid4()
        result = make_llm_result(
            message=make_ai_message(content="", tool_calls=[{"name": "expected_tool", "args": {}, "id": "1"}])
        )
        active_handler.on_llm_end(result, run_id=llm_run, parent_run_id=parent_id)

        other_tool_run = uuid.uuid4()
        active_handler.on_tool_start({"name": "unrelated_tool"}, "{}", run_id=other_tool_run, parent_run_id=parent_id)
        active_handler.on_tool_end("output", run_id=other_tool_run, parent_run_id=parent_id)

        assert len(active_handler._pending_llm_tool_calls) == 1
        assert active_handler._pending_llm_tool_calls[0]["remaining_tools"] == ["expected_tool"]


# ---------------------------------------------------------------------------
# on_chain_start — root-run reset behavior
# ---------------------------------------------------------------------------

class TestOnChainStartRootReset:
    def test_root_call_sets_trace_id(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        assert active_handler.trace_id == run_id

    def test_root_call_registers_as_current_callback(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        assert _current_callback.get() is active_handler

    def test_non_root_call_does_not_change_trace_id(self, active_handler):
        original_trace_id = active_handler.trace_id
        active_handler.on_chain_start({"name": "sub_chain"}, {}, run_id=uuid.uuid4(), parent_run_id=uuid.uuid4())
        assert active_handler.trace_id == original_trace_id

    def test_reusing_handler_across_two_top_level_invocations_does_not_leak_events(self):
        handler = _AgentGuardCallback(agent_name="reusable-agent")

        # First "graph.invoke()".
        run_1 = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_1, parent_run_id=None)
        handler.on_chain_end({"result": "first run output"}, run_id=run_1, parent_run_id=None)
        first_run_events = list(handler.events)
        assert len(first_run_events) == 2

        # Second "graph.invoke()" reusing the same instance.
        run_2 = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_2, parent_run_id=None)

        assert handler.trace_id == run_2
        assert handler.events != first_run_events
        assert all(e["trace_id"] == str(run_2) for e in handler.events)
        assert not any(
            e.get("outputs", {}).get("result") == "first run output" for e in handler.events
        )

    def test_reset_clears_run_registry_and_pending_tool_calls(self, active_handler):
        active_handler._run_registry["stale"] = {"name": "x", "type": "llm"}
        active_handler._pending_llm_tool_calls.append({"event": {}, "parent_run_id": None, "remaining_tools": ["x"]})
        active_handler._evaluated = True

        new_run_id = uuid.uuid4()
        active_handler.on_chain_start({"name": "graph"}, {}, run_id=new_run_id, parent_run_id=None)

        assert "stale" not in active_handler._run_registry
        assert active_handler._pending_llm_tool_calls == []
        assert active_handler._evaluated is False
        assert str(new_run_id) in active_handler._run_registry


# ---------------------------------------------------------------------------
# on_chain_start — langgraph_step metadata
# ---------------------------------------------------------------------------

class TestOnChainStartLanggraphStep:
    def test_langgraph_step_surfaced_from_metadata(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start(
            {"name": "inventory_agent"}, {}, run_id=run_id, parent_run_id=uuid.uuid4(),
            metadata={"langgraph_step": 3},
        )
        assert active_handler.events[-1]["langgraph_step"] == 3

    def test_missing_metadata_gives_none(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start(
            {"name": "inventory_agent"}, {}, run_id=run_id, parent_run_id=uuid.uuid4()
        )
        assert active_handler.events[-1]["langgraph_step"] is None

    def test_metadata_without_langgraph_step_key_gives_none(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start(
            {"name": "n"}, {}, run_id=run_id, parent_run_id=uuid.uuid4(), metadata={"other": "value"},
        )
        assert active_handler.events[-1]["langgraph_step"] is None

    def test_chain_name_falls_back_to_kwargs_name(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start(None, {}, run_id=run_id, parent_run_id=uuid.uuid4(), name="kwarg-name")
        assert active_handler.events[-1]["node_name"] == "kwarg-name"


# ---------------------------------------------------------------------------
# on_chain_end — auto-evaluate trigger (_maybe_auto_evaluate)
# ---------------------------------------------------------------------------

class TestOnChainEndAutoEvaluate:
    def test_none_mode_never_calls_evaluate_confidence(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.NONE)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)
        mock_eval.assert_not_called()

    def test_always_mode_calls_evaluate_confidence(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)
        mock_eval.assert_called_once_with(_observability_call=True)

    def test_always_mode_calls_even_if_already_evaluated(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        handler._evaluated = True
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)
        mock_eval.assert_called_once()

    def test_if_not_evaluated_mode_calls_when_not_yet_evaluated(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.IF_NOT_EVALUATED)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)
        mock_eval.assert_called_once()

    def test_if_not_evaluated_mode_skips_when_already_evaluated(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.IF_NOT_EVALUATED)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        handler._evaluated = True
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)
        mock_eval.assert_not_called()

    def test_non_root_chain_end_never_triggers_auto_eval(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
        root_run = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=root_run, parent_run_id=None)
        sub_run = uuid.uuid4()
        handler.on_chain_start({"name": "sub"}, {}, run_id=sub_run, parent_run_id=root_run)
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            handler.on_chain_end({}, run_id=sub_run, parent_run_id=root_run)
        mock_eval.assert_not_called()

    def test_exception_from_evaluate_confidence_is_caught_and_logged(self):
        handler = _AgentGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
        run_id = uuid.uuid4()
        handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        with patch("trellar.agent_loop.evaluate_confidence", side_effect=RuntimeError("backend down")):
            # Must not raise.
            handler.on_chain_end({}, run_id=run_id, parent_run_id=None)


# ---------------------------------------------------------------------------
# build_context — narrative rendering
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_header_includes_trace_id_and_step_count(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start({"name": "graph"}, {}, run_id=run_id, parent_run_id=None)
        context = active_handler.build_context()
        assert f"Trace ID: {run_id}" in context
        assert "Total steps: 1" in context

    def test_llm_start_renders_system_and_human(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chat_model_start(
            {"kwargs": {"model": "gpt-4o"}},
            [[SystemMessage(content="sys prompt"), HumanMessage(content="human prompt")]],
            run_id=run_id, parent_run_id=None,
        )
        context = active_handler.build_context()
        assert "system: sys prompt" in context
        assert "human: human prompt" in context

    def test_llm_end_renders_response_and_token_usage(self, active_handler):
        run_id = uuid.uuid4()
        result = make_llm_result(message=make_ai_message(content="the answer"), token_usage={"total_tokens": 10})
        active_handler.on_llm_end(result, run_id=run_id, parent_run_id=None)
        context = active_handler.build_context()
        assert "response: the answer" in context
        assert "token_usage:" in context
        assert '"total_tokens": 10' in context

    def test_tool_start_and_end_rendered(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "my_tool"}, "{}", run_id=run_id, parent_run_id=None, inputs={"x": 1})
        active_handler.on_tool_end("tool output", run_id=run_id, parent_run_id=None)
        context = active_handler.build_context()
        assert "tool: my_tool" in context
        assert "output: " in context
        assert "tool output" in context

    def test_chain_start_and_end_rendered(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_chain_start({"name": "my_chain"}, {"a": 1}, run_id=run_id, parent_run_id=uuid.uuid4())
        active_handler.on_chain_end({"b": 2}, run_id=run_id, parent_run_id=uuid.uuid4())
        context = active_handler.build_context()
        assert "inputs:" in context
        assert "outputs:" in context

    def test_error_events_rendered(self, active_handler):
        active_handler.on_llm_error(ValueError("oops"), run_id=uuid.uuid4(), parent_run_id=None)
        context = active_handler.build_context()
        assert "error: oops" in context

    def test_node_name_and_type_included_in_header_when_present(self, active_handler):
        run_id = uuid.uuid4()
        active_handler.on_tool_start({"name": "my_tool"}, "{}", run_id=run_id, parent_run_id=None)
        active_handler.on_tool_end("out", run_id=run_id, parent_run_id=None)
        context = active_handler.build_context()
        assert "(tool: my_tool)" in context

    def test_missing_node_name_omits_type_suffix(self, active_handler):
        active_handler.on_llm_error(ValueError("oops"), run_id=uuid.uuid4(), parent_run_id=None)
        context = active_handler.build_context()
        for line in context.splitlines():
            if line.startswith("[Step") and "on_llm_error" in line:
                assert "(" not in line
