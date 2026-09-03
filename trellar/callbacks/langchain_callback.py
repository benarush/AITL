import json
import logging
import uuid
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .._context import _current_callback
from ..agent_loop import ObservabilityMode

logger = logging.getLogger(__name__)


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a LangChain BaseMessage to a plain dict (role + content + extras)."""
    if not (hasattr(msg, "type") and hasattr(msg, "content")):
        return {"raw": str(msg)}

    result: dict[str, Any] = {"role": msg.type, "content": msg.content}

    additional = getattr(msg, "additional_kwargs", {})
    if additional:
        # Capture tool_calls, function_call, etc.
        result["additional_kwargs"] = additional

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def _content_to_str(content: Any) -> str:
    """Convert a message content value to a plain string.

    LangChain message content can be a str, a list of dicts (multimodal),
    or any other JSON-serializable value for structured outputs.
    """
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _extract_llm_input(messages: list[Any]) -> dict[str, Optional[str]]:
    """Extract structured system/human fields from a list of LangChain messages.

    Returns a dict with:
    - ``system``: content of the first SystemMessage, or ``None``
    - ``human``: content of the last HumanMessage, or ``None``

    For multi-turn conversation histories the *last* human turn is used as the
    active prompt because that is what the LLM is responding to.

    Handles both LangChain ``BaseMessage`` objects (standard) and plain dicts
    (e.g. when Phoenix auto-instrumentation serialises messages before passing
    them to the callback).  Also accepts ``"user"`` as a synonym for ``"human"``
    to cover OpenAI-style role names.
    """
    system: Optional[str] = None
    human: Optional[str] = None

    for msg in messages:
        if hasattr(msg, "type") and hasattr(msg, "content"):
            # Standard LangChain BaseMessage object
            role = str(msg.type).lower()
            content = _content_to_str(msg.content)
        elif isinstance(msg, dict):
            # Serialised dict — may use "role" (OpenAI/Phoenix) or "type" (LangChain)
            role = str(msg.get("role") or msg.get("type") or "").lower()
            content = _content_to_str(msg.get("content") or "")
        else:
            continue

        if role == "system" and system is None:
            system = content
        elif role in ("human", "user"):
            # "user" is the OpenAI/Phoenix style; keep overwriting so the last wins.
            human = content

    return {"system": system, "human": human}


def _compact_json(value: Any) -> str:
    """Render *value* as compact JSON, falling back to repr on failure."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def _extract_model_name(serialized: dict[str, Any]) -> Optional[str]:
    """
    Pull the real model identifier out of a serialized LLM dict.

    LangChain puts the *class* name in ``serialized["name"]`` (e.g. "ChatOpenAI")
    but the actual model string (e.g. "gpt-4o") lives inside ``kwargs``.
    """
    kwargs = serialized.get("kwargs", {})
    name = kwargs.get("model_name") or kwargs.get("model")
    if not name:
        # Fall back to the class name so we always have something.
        name = serialized.get("name")
    return name


class _AgentGuardCallback(BaseCallbackHandler):
    """Internal LangChain callback handler that tracks agent lifecycle events.

    This class is not part of the public API. Use :func:`get_agent_guard` to
    obtain an instance.

    Accumulates all graph events into ``self.events`` as a list of dicts.
    State is reset at the start of each top-level ``graph.invoke()`` call
    (detected via ``parent_run_id is None``), so reusing one instance across
    multiple invocations does not leak prior-run events into later payloads.
    Each dict contains:

    * ``event``        – callback name (e.g. ``on_chat_model_start``)
    * ``graph_order``  – monotonically increasing step counter across the whole run
    * ``trace_id``     – root run_id (graph-level trace)
    * ``run_id``       – this specific run
    * ``parent_run_id``– direct parent run (``None`` for the root)
    * ``node_name``    – human-readable name of the node/chain/tool/model
    * ``node_type``    – ``"llm"``, ``"tool"``, or ``"chain"``
    * extra payload fields depending on the event type
    """

    # Maps the LangChain message `type` attribute to a human-readable prefix.
    _MESSAGE_PREFIXES: dict[str, str] = {
        "ai": "AI MESSAGE",
        "human": "HUMAN MESSAGE",
        "system": "SYSTEM MESSAGE",
        "tool": "TOOL MESSAGE",
        "function": "FUNCTION MESSAGE",
        "chat": "CHAT MESSAGE",
    }

    @staticmethod
    def _serialize_message_obj(msg: Any) -> str:
        """Serialize a single LangChain message object into a labeled plain string.

        Mirrors the logic in ``_serialize_messages`` but for a single value.
        Falls back to ``str()`` for anything that is not a recognised message object.
        """
        if hasattr(msg, "type") and hasattr(msg, "content"):
            prefix = _AgentGuardCallback._MESSAGE_PREFIXES.get(msg.type.lower(), "MESSAGE")
            content = msg.content
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except (TypeError, ValueError):
                    content = str(content)
            return f"{prefix}: {content}"
        return str(msg)

    @staticmethod
    def _serialize_messages(messages: list[Any]) -> list[str]:
        """Convert a list of LangChain message objects into labeled plain strings.

        Each item is formatted as ``"<TYPE PREFIX>: <content>"`` so that the
        result is JSON-serializable and unambiguous about which role produced
        the content.  Unknown message types fall back to ``"MESSAGE: ..."`` and
        anything that is not a recognised message object is coerced via ``str()``.
        """
        result: list[str] = []
        for msg in messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                prefix = _AgentGuardCallback._MESSAGE_PREFIXES.get(
                    msg.type.lower(), "MESSAGE"
                )
                content = msg.content
                if not isinstance(content, str):
                    # content can be a list of dicts (e.g. multimodal messages)
                    try:
                        content = json.dumps(content)
                    except (TypeError, ValueError):
                        content = str(content)
                result.append(f"{prefix}: {content}")
            else:
                result.append(str(msg))
        return result

    def __init__(
        self,
        *,
        agent_name: str,
        observability_mode: ObservabilityMode = ObservabilityMode.NONE,
    ) -> None:
        if not agent_name or not agent_name.strip():
            raise ValueError(
                "agent_name is required. It uniquely identifies this agent graph in the "
                "Trellar backend and is used to track its network profile across runs. "
                "Use a stable, descriptive name such as 'research-agent' or 'support-bot'."
            )
        super().__init__()
        self.agent_name: str = agent_name
        self.observability_mode: ObservabilityMode = ObservabilityMode(observability_mode)
        self.trace_id: Optional[uuid.UUID] = None
        self.events: list[dict[str, Any]] = []
        self._step: int = 0
        # Whether evaluate_confidence() has already succeeded during this run;
        # reset per top-level invocation alongside the other run state below.
        self._evaluated: bool = False
        # run_id (str) -> {"name": str, "type": str}
        self._run_registry: dict[str, dict[str, Any]] = {}
        # LLM events that requested tool calls and are still awaiting the
        # corresponding tool outputs. Each entry:
        #   {"event": <recorded event dict>, "parent_run_id": str | None,
        #    "remaining_tools": [tool names...]}
        self._pending_llm_tool_calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _register(
        self,
        run_id: uuid.UUID,
        name: Optional[str],
        node_type: str,
    ) -> None:
        self._run_registry[str(run_id)] = {"name": name, "type": node_type}

    def _record(
        self,
        event: str,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **data: Any,
    ) -> None:
        node_info = self._run_registry.get(str(run_id), {})
        self.events.append(
            self._to_jsonable(
                {
                    "event": event,
                    "graph_order": self._next_step(),
                    "trace_id": str(self.trace_id),
                    "run_id": str(run_id),
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "node_name": node_info.get("name"),
                    "node_type": node_info.get("type"),
                    **data,
                }
            )
        )

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        """Recursively coerce *value* into something ``json.dumps`` can handle.

        LangChain/LangGraph hand callbacks all sorts of raw objects that are
        not JSON-safe out of the box — most notably pydantic models (e.g. the
        return value of ``with_structured_output``) and message objects. This
        makes sure nothing appended to ``self.events`` can ever break the
        ``evaluate_confidence()`` HTTP call downstream.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "type") and hasattr(value, "content"):
            return _AgentGuardCallback._serialize_message_obj(value)
        if hasattr(value, "model_dump"):
            return _AgentGuardCallback._to_jsonable(value.model_dump(mode="json"))
        if isinstance(value, dict):
            return {k: _AgentGuardCallback._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_AgentGuardCallback._to_jsonable(v) for v in value]
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        model = _extract_model_name(serialized)
        self._register(run_id, model, "llm")
        self._record(
            "on_llm_start",
            run_id,
            parent_run_id,
            model=model,
            input={"system": None, "human": "\n".join(prompts)},
        )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        model = _extract_model_name(serialized)
        self._register(run_id, model, "llm")

        # messages is list[list[BaseMessage]] — one inner list per prompt batch item.
        # Use the first batch to extract system/human fields.
        llm_input = _extract_llm_input(messages[0] if messages else [])
        self._record(
            "on_chat_model_start",
            run_id,
            parent_run_id,
            model=model,
            input=llm_input,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Pull the text from the first generation of the first batch.
        # Try multiple attributes in priority order to handle:
        #   - ChatGeneration (.message.content) — standard LangChain chat models
        #   - Generation (.text) — plain (non-chat) LLMs
        #   - Gemini multimodal content (list of parts — serialise to JSON)
        #   - Dicts produced by Phoenix instrumentation wrapping
        response_text: str = ""
        tool_calls: list[Any] = []
        if response.generations:
            first_batch = response.generations[0]
            if first_batch:
                gen = first_batch[0]

                # 1. Try .message.content (ChatGeneration)
                message = getattr(gen, "message", None)
                if message is not None:
                    tool_calls = getattr(message, "tool_calls", None) or []
                    content = getattr(message, "content", None)
                    if content is not None:
                        response_text = _content_to_str(content)
                    else:
                        # Content is None — fall through to .text
                        message = None

                # 2. Try .text (plain Generation or ChatGeneration fallback)
                if not message:
                    raw_text = getattr(gen, "text", None)
                    if raw_text is not None:
                        # .text can be a list when Gemini returns multimodal content
                        response_text = _content_to_str(raw_text) if not isinstance(raw_text, str) else (raw_text or "")

                # 3. Treat gen itself as a dict (Phoenix serialisation edge case)
                if not response_text and isinstance(gen, dict):
                    msg_dict = gen.get("message") or {}
                    if isinstance(msg_dict, dict):
                        response_text = _content_to_str(msg_dict.get("content"))
                        if not tool_calls:
                            tool_calls = msg_dict.get("tool_calls") or []
                    else:
                        response_text = _content_to_str(gen.get("text", ""))

        token_usage = (response.llm_output or {}).get("token_usage") or (
            response.llm_output or {}
        ).get("usage")

        # Fold tool calls into the response text so the payload keeps its
        # original shape ({"response": <str>}). The matching tool results are
        # appended retroactively by on_tool_end once each tool finishes.
        if tool_calls:
            parts = [response_text] if response_text else []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parts.append(
                        f"TOOL CALL: {tc.get('name')}(args={_compact_json(tc.get('args'))})"
                    )
            response_text = "\n".join(parts)

        self._record(
            "on_llm_end",
            run_id,
            parent_run_id,
            output={"response": response_text},
            token_usage=token_usage,
        )

        if tool_calls:
            # _record appends a jsonable copy; keep a reference to that copy so
            # on_tool_end can enrich it in place before the payload is built.
            self._pending_llm_tool_calls.append(
                {
                    "event": self.events[-1],
                    "parent_run_id": str(parent_run_id) if parent_run_id else None,
                    "remaining_tools": [
                        tc.get("name") for tc in tool_calls if isinstance(tc, dict)
                    ],
                }
            )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("on_llm_error", run_id, parent_run_id, error=str(error))

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name")
        tool_description = serialized.get("description")
        self._register(run_id, tool_name, "tool")

        # ``inputs`` kwarg carries the parsed argument dict when available.
        parsed_inputs = kwargs.get("inputs")

        self._record(
            "on_tool_start",
            run_id,
            parent_run_id,
            tool=tool_name,
            tool_description=tool_description,
            input={
                "raw": input_str,
                "parsed": parsed_inputs,
            },
            # parent_run_id already in the record; surface it explicitly
            # so callers can link this tool call back to the LLM that invoked it.
            invoked_by_run_id=str(parent_run_id) if parent_run_id else None,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        serialized_output = self._serialize_message_obj(output)
        self._record("on_tool_end", run_id, parent_run_id, output=serialized_output)
        tool_name = self._run_registry.get(str(run_id), {}).get("name")
        self._attach_tool_response_to_llm(tool_name, serialized_output, parent_run_id)

    def _attach_tool_response_to_llm(
        self,
        tool_name: Optional[str],
        serialized_output: str,
        parent_run_id: Optional[uuid.UUID],
    ) -> None:
        """Retroactively attach a finished tool's output to the LLM event that
        requested it, so the LLM's recorded output is never just an empty string.

        Matching strategy (most recent entries first):
        1. A pending LLM event sharing the same ``parent_run_id`` (the LLM and
           the tool live in the same chain/node — direct-invocation pattern).
        2. Any pending LLM event still awaiting this tool name (covers
           ToolNode/react-agent graphs where the tool runs under a different
           parent chain).
        """
        if not tool_name or not self._pending_llm_tool_calls:
            return

        parent_id = str(parent_run_id) if parent_run_id else None
        match: Optional[dict[str, Any]] = None
        for entry in reversed(self._pending_llm_tool_calls):
            if tool_name not in entry["remaining_tools"]:
                continue
            if entry["parent_run_id"] == parent_id:
                match = entry
                break
            if match is None:
                match = entry

        if match is None:
            return

        match["event"]["output"]["response"] += (
            f"\nTOOL RESPONSE [{tool_name}]: {serialized_output}"
        )
        match["remaining_tools"].remove(tool_name)
        if not match["remaining_tools"]:
            self._pending_llm_tool_calls.remove(match)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("on_tool_error", run_id, parent_run_id, error=str(error))

    # ------------------------------------------------------------------
    # Chain / Graph node events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            # Root invocation — this run_id is the graph-level trace ID.
            self.trace_id = run_id
            # Guard against reused instances: if this handler is passed into more
            # than one top-level graph.invoke() (sequentially), drop state from
            # the previous run instead of letting it accumulate unbounded and
            # leaking into this run's evaluate_confidence() payload.
            self.events = []
            self._step = 0
            self._run_registry = {}
            self._pending_llm_tool_calls = []
            self._evaluated = False
            # Self-register so evaluate_confidence() can pick us up automatically.
            _current_callback.set(self)

        chain_name = (serialized or {}).get("name") or kwargs.get("name")

        # `inputs` is usually the graph/node state dict, but LangChain also
        # fires this event for internal sub-runnables (e.g. ToolNode) whose
        # raw input is a bare list of messages or a single message object.
        # The backend's schema requires `inputs` to always be a list, so
        # every branch below normalizes to one instead of a bare string/dict.
        if isinstance(inputs, dict) and "messages" in inputs:
            safe_inputs = self._serialize_messages(inputs["messages"])
        elif isinstance(inputs, list):
            safe_inputs = self._serialize_messages(inputs)
        else:
            safe_inputs = [self._to_jsonable(inputs)]

        # LangGraph stamps its own superstep number onto the RunnableConfig
        # metadata for each Pregel node task (see langgraph/pregel/_algo.py).
        # Nodes sharing the same langgraph_step ran in the same superstep —
        # i.e. in parallel — which lets the backend distinguish true fan-out
        # branches from a sequential chain. None for non-LangGraph callers.
        metadata = kwargs.get("metadata") or {}
        langgraph_step = metadata.get("langgraph_step")

        self._register(run_id, chain_name, "chain")
        self._record(
            "on_chain_start",
            run_id,
            parent_run_id,
            inputs=safe_inputs,
            langgraph_step=langgraph_step,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        # See on_chain_start — `outputs` is not always a dict (e.g. the raw
        # pydantic model returned by a `with_structured_output` sub-chain).
        if isinstance(outputs, dict) and "messages" in outputs:
            outputs = {**outputs, "messages": self._serialize_messages(outputs["messages"])}
        self._record("on_chain_end", run_id, parent_run_id, outputs=self._to_jsonable(outputs))
        # Do NOT clear _current_callback here. ContextVar is already scoped per
        # asyncio Task / thread, so it never leaks across concurrent runs.
        # Clearing it before graph.invoke() returns would make evaluate_confidence()
        # fail when called after the graph completes.
        if parent_run_id is None:
            # Root run ending — the whole graph flow has reached its end.
            self._maybe_auto_evaluate()

    def _maybe_auto_evaluate(self) -> None:
        """Auto-trigger evaluate_confidence() per self.observability_mode.

        Errors are caught and logged, never raised, so a passive observability
        call can never crash the graph.
        """
        if self.observability_mode is ObservabilityMode.NONE:
            return
        if self.observability_mode is ObservabilityMode.IF_NOT_EVALUATED and self._evaluated:
            return
        from ..agent_loop import evaluate_confidence
        try:
            evaluate_confidence(_observability_call=True)
        except Exception:
            logger.warning("Auto-triggered evaluate_confidence() failed", exc_info=True)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("on_chain_error", run_id, parent_run_id, error=str(error))

    # ------------------------------------------------------------------
    # Context serialization
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        """Serialize collected events into a structured string for the Trellar backend.

        Produces a numbered, step-by-step narrative of the full agent run
        (LLM calls, tool invocations, chain boundaries) suitable as the
        ``context`` field of the evaluate_confidence request.
        """
        lines: list[str] = [
            f"=== Agent Run Context ===",
            f"Trace ID: {self.trace_id}",
            f"Total steps: {len(self.events)}",
            "",
        ]

        for event in self.events:
            step = event.get("graph_order", "?")
            event_name = event.get("event", "unknown")
            node_name = event.get("node_name") or ""
            node_type = event.get("node_type") or ""

            header = f"[Step {step}] {event_name}"
            if node_name:
                header += f"  ({node_type}: {node_name})"
            lines.append(header)

            # Per-event payload rendering
            if event_name in ("on_llm_start", "on_chat_model_start"):
                inp = event.get("input", {})
                if inp.get("system"):
                    lines.append(f"  system: {inp['system']}")
                if inp.get("human"):
                    lines.append(f"  human: {inp['human']}")

            elif event_name == "on_llm_end":
                out = event.get("output", {})
                usage = event.get("token_usage")
                lines.append(f"  response: {out.get('response', '')}")
                if usage:
                    lines.append(f"  token_usage: {_compact_json(usage)}")

            elif event_name == "on_tool_start":
                lines.append(f"  tool: {event.get('tool', '')}")
                lines.append(f"  input: {_compact_json(event.get('input', {}))}")

            elif event_name == "on_tool_end":
                lines.append(f"  output: {_compact_json(event.get('output', ''))}")

            elif event_name == "on_chain_start":
                lines.append(f"  inputs: {_compact_json(event.get('inputs', {}))}")

            elif event_name == "on_chain_end":
                lines.append(f"  outputs: {_compact_json(event.get('outputs', {}))}")

            elif event_name in ("on_llm_error", "on_tool_error", "on_chain_error"):
                lines.append(f"  error: {event.get('error', '')}")

            lines.append("")  # blank line between steps

        return "\n".join(lines)
