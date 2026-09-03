"""Shared test builders for LangChain-object fakes and scripted callback event
sequences. Not a test module itself -- imported by the other test files.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult


def make_ai_message(content: Any = "", tool_calls: Optional[list] = None) -> AIMessage:
    """Build an AIMessage, optionally carrying tool_calls."""
    return AIMessage(content=content, tool_calls=tool_calls or [])


def make_chat_generation(
    content: Any = "",
    tool_calls: Optional[list] = None,
    message: Optional[AIMessage] = None,
) -> ChatGeneration:
    """Build a ChatGeneration wrapping an AIMessage (the standard chat-model shape)."""
    msg = message if message is not None else make_ai_message(content=content, tool_calls=tool_calls)
    return ChatGeneration(message=msg)


def make_llm_result(
    *,
    message: Optional[AIMessage] = None,
    text: Optional[str] = None,
    token_usage: Optional[dict] = None,
    generation: Optional[Any] = None,
) -> LLMResult:
    """Build an LLMResult with a single generation in a single batch.

    Pass ``message`` for the standard ChatGeneration path, ``text`` for a plain
    Generation, or a pre-built ``generation`` object for edge cases.
    """
    if generation is not None:
        gen = generation
    elif message is not None:
        gen = ChatGeneration(message=message)
    else:
        gen = Generation(text=text or "")

    llm_output = {"token_usage": token_usage} if token_usage else None
    return LLMResult(generations=[[gen]], llm_output=llm_output)


class FakeMessage:
    """Duck-typed stand-in for a LangChain BaseMessage.

    Used for edge cases the real pydantic message classes reject outright
    (e.g. ``content=None``), since ``on_llm_end`` only ever reaches these
    objects via ``getattr``, never ``isinstance``.
    """

    def __init__(self, content: Any = None, tool_calls: Optional[list] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeGeneration:
    """Duck-typed stand-in for a Generation/ChatGeneration.

    Bypasses pydantic validation so edge cases (``content=None``, a ``.text``
    that is itself a list, etc.) can be constructed directly.
    """

    def __init__(self, message: Optional[Any] = None, text: Optional[Any] = None):
        self.message = message
        self.text = text


class FakeResponse:
    """Duck-typed stand-in for LLMResult.

    ``on_llm_end`` only ever accesses ``.generations`` and ``.llm_output`` via
    attribute access, so a plain object lets us exercise edge cases (empty
    generations, dict-shaped Phoenix generations) the real pydantic
    ``LLMResult`` would reject.
    """

    def __init__(self, generations: Optional[list] = None, llm_output: Optional[dict] = None):
        self.generations = generations if generations is not None else []
        self.llm_output = llm_output


class ScriptedRun:
    """Fires an ordered sequence of callback events against a handler.

    Keeps e2e scenario setup readable without hand-rolling ``uuid``
    bookkeeping for every run. Usage::

        script = ScriptedRun(handler)
        root = script.chain_start(name="graph", parent=None)
        llm = script.chat_model_start(model="gpt-4o", parent=root, messages=[[...]])
        script.llm_end(llm, parent=root, result=make_llm_result(message=make_ai_message("hi")))
    """

    def __init__(self, handler):
        self.handler = handler

    @staticmethod
    def new_id() -> uuid.UUID:
        return uuid.uuid4()

    def chain_start(
        self,
        *,
        name: str,
        parent: Optional[uuid.UUID] = None,
        inputs: Any = None,
        metadata: Optional[dict] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        run_id = run_id or self.new_id()
        self.handler.on_chain_start(
            {"name": name},
            inputs if inputs is not None else {},
            run_id=run_id,
            parent_run_id=parent,
            metadata=metadata or {},
        )
        return run_id

    def chain_end(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID] = None, outputs: Any = None) -> None:
        self.handler.on_chain_end(
            outputs if outputs is not None else {}, run_id=run_id, parent_run_id=parent
        )

    def chain_error(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID] = None, error: BaseException) -> None:
        self.handler.on_chain_error(error, run_id=run_id, parent_run_id=parent)

    def chat_model_start(
        self,
        *,
        model: str,
        parent: Optional[uuid.UUID],
        messages: list,
        run_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        run_id = run_id or self.new_id()
        self.handler.on_chat_model_start(
            {"kwargs": {"model": model}}, messages, run_id=run_id, parent_run_id=parent
        )
        return run_id

    def llm_end(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID], result: LLMResult) -> None:
        self.handler.on_llm_end(result, run_id=run_id, parent_run_id=parent)

    def llm_error(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID], error: BaseException) -> None:
        self.handler.on_llm_error(error, run_id=run_id, parent_run_id=parent)

    def tool_start(
        self,
        *,
        name: str,
        parent: Optional[uuid.UUID],
        input_str: str = "{}",
        parsed_inputs: Optional[dict] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> uuid.UUID:
        run_id = run_id or self.new_id()
        self.handler.on_tool_start(
            {"name": name, "description": f"{name} tool"},
            input_str,
            run_id=run_id,
            parent_run_id=parent,
            inputs=parsed_inputs,
        )
        return run_id

    def tool_end(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID], output: Any) -> None:
        self.handler.on_tool_end(output, run_id=run_id, parent_run_id=parent)

    def tool_error(self, run_id: uuid.UUID, *, parent: Optional[uuid.UUID], error: BaseException) -> None:
        self.handler.on_tool_error(error, run_id=run_id, parent_run_id=parent)
