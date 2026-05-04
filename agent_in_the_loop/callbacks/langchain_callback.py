import uuid
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


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


class DebugCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that tracks lifecycle events.

    Accumulates all graph events into ``self.events`` as a list of dicts.
    Each dict contains:

    * ``event``        – callback name (e.g. ``on_chat_model_start``)
    * ``graph_order``  – monotonically increasing step counter across the whole run
    * ``trace_id``     – root run_id (graph-level trace)
    * ``run_id``       – this specific run
    * ``parent_run_id``– direct parent run (``None`` for the root)
    * ``node_name``    – human-readable name of the node/chain/tool/model
    * ``node_type``    – ``"llm"``, ``"tool"``, or ``"chain"``
    * extra payload fields depending on the event type

    Usage::

        handler = DebugCallbackHandler()
        graph.invoke(input, config={"callbacks": [handler]})
        print(handler.events)   # full trace of the run
    """

    def __init__(self) -> None:
        super().__init__()
        self.trace_id: Optional[uuid.UUID] = None
        self.events: list[dict[str, Any]] = []
        self._step: int = 0
        # run_id (str) -> {"name": str, "type": str}
        self._run_registry: dict[str, dict[str, Any]] = {}

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
            input={"prompts": prompts},
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
        serialized_messages = [
            [_serialize_message(m) for m in batch] for batch in messages
        ]
        self._record(
            "on_chat_model_start",
            run_id,
            parent_run_id,
            model=model,
            input={"messages": serialized_messages},
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Extract generated texts cleanly instead of dumping the whole dict.
        output_texts: list[list[str]] = []
        for batch in response.generations:
            texts = []
            for gen in batch:
                # ChatGeneration has .message; Generation has .text
                message = getattr(gen, "message", None)
                if message is not None:
                    texts.append(_serialize_message(message))
                else:
                    texts.append(gen.text)
            output_texts.append(texts)

        token_usage = (response.llm_output or {}).get("token_usage") or (
            response.llm_output or {}
        ).get("usage")

        self._record(
            "on_llm_end",
            run_id,
            parent_run_id,
            output={"generations": output_texts},
            token_usage=token_usage,
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
        self._record("on_tool_end", run_id, parent_run_id, output=output)

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

        chain_name = serialized.get("name") or kwargs.get("name")
        self._register(run_id, chain_name, "chain")
        self._record(
            "on_chain_start",
            run_id,
            parent_run_id,
            inputs=inputs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("on_chain_end", run_id, parent_run_id, outputs=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._record("on_chain_error", run_id, parent_run_id, error=str(error))
