import json
import uuid
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


def _pretty(obj: Any) -> str:
    """Best-effort pretty-print of any object."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


class DebugCallbackHandler(BaseCallbackHandler):
    """Prints every LangChain callback event with its full payload.

    Attach this handler to the LLM or the graph to observe exactly what
    LangChain sends at each lifecycle event — useful for understanding
    what data would be available in a real AITL integration.
    """

    def __init__(self):
        super().__init__()
        self._call_count = 0

    def _header(self, event: str, run_id: uuid.UUID) -> None:
        self._call_count += 1
        print(f"\n{'='*70}")
        print(f"[CALLBACK #{self._call_count}]  EVENT: {event}")
        print(f"  run_id: {run_id}")
        print(f"{'='*70}")

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
        self._header("on_llm_start", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  serialized    :\n{_pretty(serialized)}")
        print(f"  prompts       :\n{_pretty(prompts)}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_chat_model_start", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  serialized    :\n{_pretty(serialized)}")
        for i, msg_batch in enumerate(messages):
            print(f"  messages[{i}]:")
            for msg in msg_batch:
                print(f"    type={type(msg).__name__}  content={_pretty(msg.content)}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_llm_end", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  llm_output    :\n{_pretty(response.llm_output)}")
        for i, gen_list in enumerate(response.generations):
            for j, gen in enumerate(gen_list):
                print(f"  generations[{i}][{j}]:")
                print(f"    text  : {_pretty(getattr(gen, 'text', None))}")
                print(f"    type  : {type(gen).__name__}")
                if hasattr(gen, "message"):
                    print(f"    message.content : {_pretty(gen.message.content)}")
                    print(f"    message.type    : {gen.message.type}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_llm_error", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  error         : {error}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

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
        self._header("on_tool_start", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  serialized    :\n{_pretty(serialized)}")
        print(f"  input_str     : {input_str}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_tool_end", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  output        : {_pretty(output)}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_tool_error", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  error         : {error}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    # ------------------------------------------------------------------
    # Chain / Graph events
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
        self._header("on_chain_start", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  serialized    :\n{_pretty(serialized)}")
        print(f"  inputs        :\n{_pretty(inputs)}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_chain_end", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  outputs       :\n{_pretty(outputs)}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._header("on_chain_error", run_id)
        print(f"  parent_run_id : {parent_run_id}")
        print(f"  error         : {error}")
        print(f"  extra kwargs  :\n{_pretty(kwargs)}")
