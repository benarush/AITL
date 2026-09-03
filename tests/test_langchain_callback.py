from __future__ import annotations

import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from trellar.callbacks.langchain_callback import _AgentGuardCallback


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
