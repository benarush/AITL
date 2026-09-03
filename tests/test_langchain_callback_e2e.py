"""End-to-end simulations of realistic multi-agent LangGraph runs against
_AgentGuardCallback, mirroring the topologies in:

  - orchestrations_examples/our_lab_with_aviran/l3_healthcare_prior_authorization/healthcare_prior_authorization.py
  - orchestrations_examples/our_lab_with_aviran/curcurent_network/supply_chain_network.py

Real LLM calls require network/API keys, so "e2e" here means: drive
_AgentGuardCallback through the exact sequence of callback invocations
LangGraph/LangChain would fire for these topologies (using fake
AIMessage/LLMResult objects via ScriptedRun), then assert on the resulting
event list, build_context() output, and auto-eval side effects as a whole --
not just individual method calls.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from trellar import ObservabilityMode
from trellar.callbacks.langchain_callback import _AgentGuardCallback

from tests.factories import ScriptedRun, make_ai_message, make_llm_result


class _FakePatientIntake(BaseModel):
    patient_id: str
    procedure_code: str
    diagnosis_code: str


class _FakeClinicalDecision(BaseModel):
    recommendation: str
    reasoning: str


# ---------------------------------------------------------------------------
# Sequential pipeline (healthcare-prior-authorization-style)
# ---------------------------------------------------------------------------

class TestSequentialPipelineE2E:
    def test_full_sequential_run_with_structured_output_and_tool_error(self):
        handler = _AgentGuardCallback(agent_name="healthcare-e2e", observability_mode=ObservabilityMode.ALWAYS)
        script = ScriptedRun(handler)

        root = script.chain_start(name="LangGraph", parent=None, inputs={"request": "auth request"})

        # --- intake_agent: structured-output LLM call ---
        intake = script.chain_start(name="intake_agent", parent=root)
        intake_llm = script.chat_model_start(
            model="gemini-2.5-flash-lite", parent=intake,
            messages=[[SystemMessage(content="Extract patient info."), HumanMessage(content="raw request")]],
        )
        script.llm_end(intake_llm, parent=intake, result=make_llm_result(message=make_ai_message(content="")))
        intake_output = _FakePatientIntake(patient_id="PT-ERR", procedure_code="29881", diagnosis_code="M23.50")
        script.chain_end(intake, parent=root, outputs=intake_output)

        # --- eligibility_check: tool call that raises, handled gracefully by the node ---
        eligibility = script.chain_start(name="eligibility_check", parent=root)
        eligibility_tool = script.tool_start(
            name="call_payer_eligibility_api", parent=eligibility, parsed_inputs={"patient_id": "PT-ERR"}
        )
        script.tool_error(eligibility_tool, parent=eligibility, error=ConnectionError("payer eligibility API timed out"))
        script.chain_end(
            eligibility, parent=root,
            outputs={"eligibility_result": {"status": "unknown", "error": "payer eligibility API timed out"}},
        )

        # --- clinical_criteria_agent: tool call + another structured-output LLM call ---
        clinical = script.chain_start(name="clinical_criteria_agent", parent=root)
        clinical_tool = script.tool_start(
            name="get_clinical_guidelines", parent=clinical, parsed_inputs={"procedure_code": "29881"}
        )
        script.tool_end(clinical_tool, parent=clinical, output="Knee arthroscopy guideline text.")
        clinical_llm = script.chat_model_start(
            model="gemini-2.5-flash-lite", parent=clinical,
            messages=[[SystemMessage(content="Recommend approve/deny/escalate."), HumanMessage(content="...")]],
        )
        script.llm_end(clinical_llm, parent=clinical, result=make_llm_result(message=make_ai_message(content="")))
        clinical_output = _FakeClinicalDecision(recommendation="deny", reasoning="No conservative treatment documented.")
        script.chain_end(clinical, parent=root, outputs=clinical_output)

        # --- decision_gate -> human_review (a "deny" is always escalated, never auto-denied) ---
        gate = script.chain_start(name="decision_gate", parent=root)
        script.chain_end(gate, parent=root, outputs={"requires_human_review": True})

        human_review = script.chain_start(name="human_review", parent=root)
        human_review_llm = script.chat_model_start(
            model="gemini-2.5-flash-lite", parent=human_review,
            messages=[[SystemMessage(content="Explain the escalation.")]],
        )
        script.llm_end(
            human_review_llm, parent=human_review,
            result=make_llm_result(message=make_ai_message(content="Escalating for human review.")),
        )
        script.chain_end(human_review, parent=root, outputs={"outcome": "ESCALATED FOR HUMAN REVIEW"})

        # --- audit_logger: runs unconditionally, no LLM call ---
        audit = script.chain_start(name="audit_logger", parent=root)
        script.chain_end(audit, parent=root, outputs={"audit_log": [{"audit_id": "abc"}]})

        # --- notifier: LLM call with no system message ---
        notifier = script.chain_start(name="notifier", parent=root)
        notifier_llm = script.chat_model_start(
            model="gemini-2.5-flash-lite", parent=notifier,
            messages=[[HumanMessage(content="Write a status message.")]],
        )
        script.llm_end(
            notifier_llm, parent=notifier,
            result=make_llm_result(message=make_ai_message(content="Your request is under review.")),
        )
        script.chain_end(notifier, parent=root, outputs={"outcome": "final outcome"})

        # --- root chain end: must auto-trigger evaluate_confidence exactly once ---
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            script.chain_end(root, parent=None, outputs={"outcome": "final outcome"})
        mock_eval.assert_called_once_with(_observability_call=True)

        # The whole run must be JSON-serializable end to end -- the exact
        # payload evaluate_confidence() would send.
        json.dumps({"context": handler.events, "trace_id": str(handler.trace_id)})

        context = handler.build_context()
        assert "on_tool_error" in context
        assert "payer eligibility API timed out" in context
        assert f"Total steps: {len(handler.events)}" in context

        # Structured-output pydantic outputs must have been coerced to plain dicts.
        chain_end_events = [e for e in handler.events if e["event"] == "on_chain_end"]
        intake_end = next(e for e in chain_end_events if e["run_id"] == str(intake))
        assert intake_end["outputs"] == {
            "patient_id": "PT-ERR", "procedure_code": "29881", "diagnosis_code": "M23.50",
        }
        clinical_end = next(e for e in chain_end_events if e["run_id"] == str(clinical))
        assert clinical_end["outputs"] == {
            "recommendation": "deny", "reasoning": "No conservative treatment documented.",
        }


# ---------------------------------------------------------------------------
# Fan-out/fan-in parallel pipeline (supply-chain-disruption-style)
# ---------------------------------------------------------------------------

class TestFanOutFanInParallelE2E:
    def test_concurrent_branches_do_not_cross_wire_tool_responses(self):
        """Mirrors supply_chain_network.py's Orchestrator -> [Inventory,
        Logistics, Procurement] -> Synthesize fan-out/fan-in shape -- the
        exact topology implicated in
        trellar-parallel-execution-bug-investigation.md. Each branch calls a
        *different* tool (as the real network does), fired in interleaved
        order to simulate real concurrent scheduling. This proves the
        callback's own event bookkeeping (self.events / parent_run_id
        linkage) is not the source of the cross-wiring the investigation
        found -- the corruption must be entirely on the backend side."""
        handler = _AgentGuardCallback(agent_name="supply-chain-e2e")
        script = ScriptedRun(handler)

        root = script.chain_start(name="LangGraph", parent=None, inputs={"incident_id": "INC-1"})

        dispatch = script.chain_start(name="orchestrator_dispatch", parent=root, metadata={"langgraph_step": 1})
        dispatch_llm = script.chat_model_start(
            model="gemini-2.5-flash", parent=dispatch, messages=[[HumanMessage(content="dispatch")]]
        )
        script.llm_end(dispatch_llm, parent=dispatch, result=make_llm_result(message=make_ai_message(content="Dispatched to teams.")))
        script.chain_end(dispatch, parent=root, outputs={"dispatch_text": "Dispatched to teams."})

        # Three concurrent branches, all under the same langgraph_step (fan-out).
        inventory = script.chain_start(name="inventory_agent", parent=root, metadata={"langgraph_step": 2})
        logistics = script.chain_start(name="logistics_agent", parent=root, metadata={"langgraph_step": 2})
        procurement = script.chain_start(name="procurement_agent", parent=root, metadata={"langgraph_step": 2})

        inventory_llm = script.chat_model_start(model="gemini-2.5-flash", parent=inventory, messages=[[HumanMessage(content="check stock")]])
        logistics_llm = script.chat_model_start(model="gemini-2.5-flash", parent=logistics, messages=[[HumanMessage(content="check routes")]])
        procurement_llm = script.chat_model_start(model="gemini-2.5-flash", parent=procurement, messages=[[HumanMessage(content="check vendors")]])

        inventory_result = make_llm_result(message=make_ai_message(
            content="", tool_calls=[{"name": "check_inventory_stock", "args": {"chip_model": "AX-9200"}, "id": "1"}]
        ))
        logistics_result = make_llm_result(message=make_ai_message(
            content="", tool_calls=[{"name": "find_alternate_routes", "args": {"origin_port": "LB"}, "id": "2"}]
        ))
        procurement_result = make_llm_result(message=make_ai_message(
            content="", tool_calls=[{"name": "contact_backup_supplier", "args": {"chip_model": "AX-9200"}, "id": "3"}]
        ))

        # Interleave the LLM-end events across branches, as real concurrent
        # execution would (no guaranteed ordering between branches).
        script.llm_end(inventory_llm, parent=inventory, result=inventory_result)
        script.llm_end(logistics_llm, parent=logistics, result=logistics_result)
        script.llm_end(procurement_llm, parent=procurement, result=procurement_result)

        inventory_tool = script.tool_start(name="check_inventory_stock", parent=inventory, parsed_inputs={"chip_model": "AX-9200"})
        logistics_tool = script.tool_start(name="find_alternate_routes", parent=logistics, parsed_inputs={"origin_port": "LB"})
        procurement_tool = script.tool_start(name="contact_backup_supplier", parent=procurement, parsed_inputs={"chip_model": "AX-9200"})

        # Finish out of start-order to simulate real scheduling jitter.
        script.tool_end(procurement_tool, parent=procurement, output={"vendor_name": "SiliconBridge Ltd.", "available_units": 8400})
        script.tool_end(inventory_tool, parent=inventory, output={"days_until_depletion": 5})
        script.tool_end(logistics_tool, parent=logistics, output={"secondary_port": "Port of Oakland"})

        script.chain_end(inventory, parent=root, outputs={"inventory_text": "5 days of stock left."})
        script.chain_end(logistics, parent=root, outputs={"logistics_text": "Oakland available."})
        script.chain_end(procurement, parent=root, outputs={"procurement_text": "SiliconBridge available."})

        # Fan-in
        synthesize = script.chain_start(name="orchestrator_synthesize", parent=root, metadata={"langgraph_step": 3})
        script.chain_end(synthesize, parent=root, outputs={"consolidated_proposal_text": "Plan ready."})

        script.chain_end(root, parent=None, outputs={"decision": "recovery_confirmed"})

        # --- Assertions: no cross-wiring between concurrent branches ---
        llm_events_by_run = {e["run_id"]: e for e in handler.events if e["event"] == "on_llm_end"}
        inventory_event = llm_events_by_run[str(inventory_llm)]
        logistics_event = llm_events_by_run[str(logistics_llm)]
        procurement_event = llm_events_by_run[str(procurement_llm)]

        assert "TOOL RESPONSE [check_inventory_stock]" in inventory_event["output"]["response"]
        assert "days_until_depletion" in inventory_event["output"]["response"]
        assert "Port of Oakland" not in inventory_event["output"]["response"]
        assert "SiliconBridge" not in inventory_event["output"]["response"]

        assert "TOOL RESPONSE [find_alternate_routes]" in logistics_event["output"]["response"]
        assert "Port of Oakland" in logistics_event["output"]["response"]
        assert "days_until_depletion" not in logistics_event["output"]["response"]
        assert "SiliconBridge" not in logistics_event["output"]["response"]

        assert "TOOL RESPONSE [contact_backup_supplier]" in procurement_event["output"]["response"]
        assert "SiliconBridge" in procurement_event["output"]["response"]
        assert "days_until_depletion" not in procurement_event["output"]["response"]
        assert "Port of Oakland" not in procurement_event["output"]["response"]

        assert handler._pending_llm_tool_calls == []

        # langgraph_step correctly marks the three branches as concurrent.
        branch_starts = [
            e for e in handler.events
            if e["event"] == "on_chain_start"
            and e["node_name"] in ("inventory_agent", "logistics_agent", "procurement_agent")
        ]
        assert {e["langgraph_step"] for e in branch_starts} == {2}

        json.dumps({"context": handler.events, "trace_id": str(handler.trace_id)})


# ---------------------------------------------------------------------------
# Multi-invocation reuse
# ---------------------------------------------------------------------------

class TestMultiInvocationReuseE2E:
    def test_second_invocation_does_not_leak_first_run_state(self):
        handler = _AgentGuardCallback(agent_name="reuse-e2e")
        script = ScriptedRun(handler)

        # --- First graph.invoke() ---
        root_1 = script.chain_start(name="LangGraph", parent=None)
        llm_1 = script.chat_model_start(model="m", parent=root_1, messages=[[HumanMessage(content="first run")]])
        script.llm_end(llm_1, parent=root_1, result=make_llm_result(message=make_ai_message(content="first run response")))
        script.chain_end(root_1, parent=None, outputs={"outcome": "first run outcome"})

        first_run_trace_id = handler.trace_id
        first_run_event_count = len(handler.events)
        assert first_run_event_count > 0

        # Simulate a manual evaluate_confidence() call having succeeded during run 1.
        handler._evaluated = True

        # --- Second graph.invoke(), same handler instance ---
        root_2 = script.chain_start(name="LangGraph", parent=None)

        assert handler.trace_id == root_2
        assert handler.trace_id != first_run_trace_id
        assert len(handler.events) == 1  # only this run's on_chain_start so far
        assert not any("first run" in json.dumps(e) for e in handler.events)
        assert handler._evaluated is False  # reset even though it was set True during run 1

        llm_2 = script.chat_model_start(model="m", parent=root_2, messages=[[HumanMessage(content="second run")]])
        script.llm_end(llm_2, parent=root_2, result=make_llm_result(message=make_ai_message(content="second run response")))
        script.chain_end(root_2, parent=None, outputs={"outcome": "second run outcome"})

        context = handler.build_context()
        assert "first run" not in context
        assert "second run response" in context


# ---------------------------------------------------------------------------
# Tool error mid-run alongside a healthy sibling branch
# ---------------------------------------------------------------------------

class TestToolErrorMidRunE2E:
    def test_branch_tool_error_alongside_sibling_success(self):
        handler = _AgentGuardCallback(agent_name="tool-error-e2e")
        script = ScriptedRun(handler)

        root = script.chain_start(name="LangGraph", parent=None)

        failing_branch = script.chain_start(name="flaky_agent", parent=root)
        healthy_branch = script.chain_start(name="healthy_agent", parent=root)

        failing_tool = script.tool_start(name="flaky_tool", parent=failing_branch)
        healthy_tool = script.tool_start(name="healthy_tool", parent=healthy_branch)

        script.tool_error(failing_tool, parent=failing_branch, error=ConnectionError("upstream timeout"))
        script.tool_end(healthy_tool, parent=healthy_branch, output="all good")

        script.chain_end(failing_branch, parent=root, outputs={"status": "degraded"})
        script.chain_end(healthy_branch, parent=root, outputs={"status": "ok"})

        script.chain_end(root, parent=None, outputs={"decision": "completed_with_warning"})

        # The run must still end coherently -- root on_chain_end is the last event.
        assert handler.events[-1]["event"] == "on_chain_end"
        assert handler.events[-1]["parent_run_id"] is None

        error_events = [e for e in handler.events if e["event"] == "on_tool_error"]
        assert len(error_events) == 1
        assert error_events[0]["error"] == "upstream timeout"

        # build_context() must render the error without raising, and the
        # sibling branch's success must still be present.
        context = handler.build_context()
        assert "on_tool_error" in context
        assert "upstream timeout" in context
        assert "all good" in context

        json.dumps({"context": handler.events, "trace_id": str(handler.trace_id)})
