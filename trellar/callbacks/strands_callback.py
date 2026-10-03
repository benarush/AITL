"""Strands Agents guard: turns Strands hook events into the same event payload
the LangChain callback produces (chain / llm / tool events).

Mapping:
    Graph/Swarm run -> root chain      Graph node   -> chain (node_id)
    Agent invoke    -> chain (agent.name)
    Model call      -> on_chat_model_start / on_llm_end
    Tool call       -> on_tool_start / on_tool_end
"""
from __future__ import annotations

import logging
import threading
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.tools.mcp.mcp_agent_tool import MCPAgentTool

from .._context import _current_callback
from ..agent_loop import ObservabilityMode
from ._common import build_context, compact_json, hash_tools, to_jsonable

logger = logging.getLogger(__name__)

# Innermost open run id. Contextvars are copied into Strands' threads/tasks,
# so parallel graph branches and nested agents each see the right parent.
_current_run: ContextVar[Optional[str]] = ContextVar("trellar_strands_current_run", default=None)


def _text_of_blocks(blocks: Any) -> str:
    """Join the text / json content blocks of a Strands message or tool result."""
    parts: list[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(compact_json(block["json"]))
    return "\n".join(parts)


def _last_user_text(messages: Any) -> Optional[str]:
    """Text of the last user message (tool-result-only user turns are skipped)."""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _text_of_blocks(msg.get("content"))
            if text:
                return text
    return None


def _model_name(agent: Any) -> str:
    """Real model id (e.g. 'gemini-2.5-flash-lite'); class name as fallback."""
    try:
        config = agent.model.get_config() or {}
        name = config.get("model_id") or config.get("model")
    except Exception:
        name = None
    return str(name or type(agent.model).__name__)


def _openai_tools(agent: Any) -> list[dict[str, Any]]:
    """Agent's tools in the OpenAI function shape the backend expects."""
    tools = []
    for spec in agent.tool_registry.get_all_tools_config().values():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": (spec.get("inputSchema") or {}).get("json", {}),
                },
            }
        )
    return to_jsonable(tools)


class _StrandsGuardCallback(HookProvider):
    """Hook provider that records a Strands run for the Trellar backend.

    Not public API; use :func:`trellar.get_strands_guard`. Register it on every
    Agent (``Agent(hooks=[guard])``) and on the Graph/Swarm
    (``GraphBuilder.set_hook_providers([guard])``).
    """

    def __init__(
        self,
        *,
        agent_name: str,
        observability_mode: ObservabilityMode = ObservabilityMode.NONE,
    ) -> None:
        if not agent_name or not agent_name.strip():
            raise ValueError(
                "agent_name is required. It uniquely identifies this agent network in the "
                "Trellar backend. Use a stable, descriptive name such as 'research-agent'."
            )
        self.agent_name = agent_name
        self.observability_mode = ObservabilityMode(observability_mode)
        self._lock = threading.Lock()
        self._reset(None)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _reset(self, root_run_id: Optional[str]) -> None:
        """Clear all per-run state; called when a new root run starts."""
        self.trace_id: Optional[str] = root_run_id
        self.events: list[dict[str, Any]] = []
        self.available_tools: dict[str, list[Any]] = {}
        self._step = 0
        self._evaluated = False
        self._root_run_id = root_run_id
        self._parents: dict[str, Optional[str]] = {}  # chain/tool run_id -> parent run_id
        self._agent_runs: dict[int, str] = {}  # id(agent) -> open invocation run_id
        self._model_runs: dict[int, str] = {}  # id(agent) -> open model-call run_id
        self._tool_runs: dict[str, str] = {}  # toolUseId -> open tool run_id
        self._tool_llm_event: dict[str, dict[str, Any]] = {}  # toolUseId -> LLM event that asked for it
        self._multi_run: Optional[str] = None  # open Graph/Swarm run_id
        self._node_runs: dict[tuple[int, str], str] = {}  # (id(graph), node_id) -> open node run_id

    def _record(
        self,
        event: str,
        run_id: str,
        parent_run_id: Optional[str],
        node_name: Optional[str],
        node_type: str,
        **data: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._step += 1
            record = to_jsonable(
                {
                    "event": event,
                    "graph_order": self._step,
                    "trace_id": self.trace_id,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                    "node_name": node_name,
                    "node_type": node_type,
                    **data,
                }
            )
            self.events.append(record)
        return record

    # ------------------------------------------------------------------
    # Chain helpers (graph, graph node, agent invocation)
    # ------------------------------------------------------------------

    def _start_chain(self, name: Optional[str], inputs: list[Any]) -> str:
        run_id = str(uuid.uuid4())
        if self._root_run_id is None:
            # First run: this is the root. Reset state and become the active guard.
            self._reset(run_id)
            parent = None
            _current_callback.set(self)
        else:
            parent = _current_run.get() or self._root_run_id
        self._parents[run_id] = parent
        _current_run.set(run_id)
        self._record("on_chain_start", run_id, parent, name, "chain", inputs=inputs)
        return run_id

    def _end_chain(self, run_id: str, name: Optional[str], outputs: Any) -> None:
        parent = self._parents.get(run_id)
        self._record("on_chain_end", run_id, parent, name, "chain", outputs=outputs)
        _current_run.set(parent)
        if run_id == self._root_run_id:
            self._finish_root()

    def _finish_root(self) -> None:
        self._maybe_auto_evaluate()
        # Release the slot so the next run starts clean; a new root re-resets state.
        if _current_callback.get() is self:
            _current_callback.set(None)
        self._root_run_id = None

    def _maybe_auto_evaluate(self) -> None:
        """Auto-call evaluate_confidence() per observability_mode; never raises."""
        if self.observability_mode is ObservabilityMode.NONE:
            return
        if self.observability_mode is ObservabilityMode.IF_NOT_EVALUATED and self._evaluated:
            return
        from ..agent_loop import evaluate_confidence

        try:
            evaluate_confidence(_observability_call=True)
        except Exception:
            logger.warning("Auto-triggered evaluate_confidence() failed", exc_info=True)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeMultiAgentInvocationEvent, self._on_multi_start)
        registry.add_callback(AfterMultiAgentInvocationEvent, self._on_multi_end)
        registry.add_callback(BeforeNodeCallEvent, self._on_node_start)
        registry.add_callback(AfterNodeCallEvent, self._on_node_end)
        registry.add_callback(BeforeInvocationEvent, self._on_agent_start)
        registry.add_callback(AfterInvocationEvent, self._on_agent_end)
        registry.add_callback(BeforeModelCallEvent, self._on_model_start)
        registry.add_callback(AfterModelCallEvent, self._on_model_end)
        registry.add_callback(BeforeToolCallEvent, self._on_tool_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool_end)

    # ------------------------------------------------------------------
    # Graph / Swarm events
    # ------------------------------------------------------------------

    @staticmethod
    def _multi_name(source: Any) -> str:
        return str(getattr(source, "id", None) or type(source).__name__)

    def _on_multi_start(self, event: BeforeMultiAgentInvocationEvent) -> None:
        task = getattr(getattr(event.source, "state", None), "task", "")
        self._multi_run = self._start_chain(self._multi_name(event.source), [str(task or "")])

    def _on_multi_end(self, event: AfterMultiAgentInvocationEvent) -> None:
        if self._multi_run:
            self._end_chain(self._multi_run, self._multi_name(event.source), {})
            self._multi_run = None

    def _on_node_start(self, event: BeforeNodeCallEvent) -> None:
        self._node_runs[(id(event.source), event.node_id)] = self._start_chain(event.node_id, [])

    def _on_node_end(self, event: AfterNodeCallEvent) -> None:
        run_id = self._node_runs.pop((id(event.source), event.node_id), None)
        if run_id is None:
            return
        results = getattr(getattr(event.source, "state", None), "results", {}) or {}
        self._end_chain(run_id, event.node_id, str(results.get(event.node_id, "")))

    # ------------------------------------------------------------------
    # Agent invocation events
    # ------------------------------------------------------------------

    def _on_agent_start(self, event: BeforeInvocationEvent) -> None:
        agent = event.agent
        prompt = _last_user_text(event.messages or agent.messages) or ""
        self._agent_runs[id(agent)] = self._start_chain(agent.name, [prompt])

    def _on_agent_end(self, event: AfterInvocationEvent) -> None:
        agent = event.agent
        run_id = self._agent_runs.pop(id(agent), None)
        if run_id is None:
            return
        self._end_chain(run_id, agent.name, str(event.result) if event.result else "")

    # ------------------------------------------------------------------
    # Model events
    # ------------------------------------------------------------------

    def _on_model_start(self, event: BeforeModelCallEvent) -> None:
        agent = event.agent
        run_id = str(uuid.uuid4())
        self._model_runs[id(agent)] = run_id

        tools_hash: Optional[str] = None
        tools = _openai_tools(agent)
        if tools:
            tools_hash = hash_tools(tools)
            self.available_tools[tools_hash] = tools

        model = _model_name(agent)
        self._parents[run_id] = self._agent_runs.get(id(agent), self._root_run_id)
        self._record(
            "on_chat_model_start",
            run_id,
            self._parents[run_id],
            model,
            "llm",
            model=model,
            input={"system": agent.system_prompt, "human": _last_user_text(agent.messages)},
            tools_hash=tools_hash,
        )

    def _on_model_end(self, event: AfterModelCallEvent) -> None:
        agent = event.agent
        run_id = self._model_runs.pop(id(agent), None)
        if run_id is None:
            return
        parent = self._parents.get(run_id)
        model = _model_name(agent)

        if event.exception is not None or event.stop_response is None:
            self._record("on_llm_error", run_id, parent, model, "llm", error=str(event.exception))
            return

        # Fold tool calls into the response text (same shape as the LangChain callback);
        # tool results are appended later by _on_tool_end.
        content = event.stop_response.message.get("content", [])
        parts = [_text_of_blocks(content)] if _text_of_blocks(content) else []
        tool_uses = [b["toolUse"] for b in content if isinstance(b, dict) and "toolUse" in b]
        for tu in tool_uses:
            parts.append(f"TOOL CALL: {tu['name']}(args={compact_json(tu.get('input'))})")

        record = self._record(
            "on_llm_end", run_id, parent, model, "llm",
            output={"response": "\n".join(parts)},
            token_usage=None,
        )
        for tu in tool_uses:
            self._tool_llm_event[tu["toolUseId"]] = record

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def _on_tool_start(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use
        run_id = str(uuid.uuid4())
        parent = self._agent_runs.get(id(event.agent), self._root_run_id)
        self._tool_runs[tool_use["toolUseId"]] = run_id
        self._parents[run_id] = parent
        _current_run.set(run_id)  # agents-as-tools nest under this tool call

        tool_input = tool_use.get("input")
        description = event.selected_tool.tool_spec.get("description") if event.selected_tool else None
        self._record(
            "on_tool_start", run_id, parent, tool_use["name"], "tool",
            tool=tool_use["name"],
            tool_description=description,
            input=tool_input if isinstance(tool_input, dict) else {"raw": tool_input},
            invoked_by_run_id=parent,
        )

    def _on_tool_end(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use
        run_id = self._tool_runs.pop(tool_use["toolUseId"], None)
        if run_id is None:
            return
        parent = self._parents.get(run_id)
        name = tool_use["name"]
        _current_run.set(parent)

        if event.exception is not None:
            self._record("on_tool_error", run_id, parent, name, "tool", error=str(event.exception))
            return

        output = _text_of_blocks(event.result.get("content"))
        self._record(
            "on_tool_end", run_id, parent, name, "tool",
            output=output,
            is_mcp_tool=isinstance(event.selected_tool, MCPAgentTool),
        )

        # Attach the tool's output to the LLM event that requested it.
        llm_event = self._tool_llm_event.pop(tool_use["toolUseId"], None)
        if llm_event is not None:
            llm_event["output"]["response"] += f"\nTOOL RESPONSE [{name}]: {output}"

    # ------------------------------------------------------------------
    # Context serialization
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        """Serialize collected events into a step-by-step string for the backend."""
        return build_context(self.events, self.trace_id)
