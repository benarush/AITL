"""Local mirror of the backend's agent-loop request schema
(aitl_fastapi/src/schemas/agent_loop/). Lets tests catch a payload that the
real server would reject with HTTP 400/422, without needing the server.

Keep in sync with the backend if its schema changes.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class _Base(BaseModel):
    event: str
    graph_order: int
    node_name: Optional[str]
    node_type: Optional[str]
    run_id: str
    parent_run_id: Optional[str]
    trace_id: str
    langgraph_step: Optional[int] = None


class LLMInput(BaseModel):
    system: Optional[str] = None
    human: Optional[str] = None


class LLMOutput(BaseModel):
    response: Optional[str] = None


class OnChainStart(_Base):
    event: Literal["on_chain_start"]
    inputs: list[Any]


class OnChainEnd(_Base):
    event: Literal["on_chain_end"]
    outputs: Any


class OnChatModelStart(_Base):
    event: Literal["on_chat_model_start"]
    input: LLMInput
    model: str
    tools_hash: Optional[str] = None


class OnLlmStart(_Base):
    event: Literal["on_llm_start"]
    input: LLMInput
    model: Optional[str] = None


class OnLlmEnd(_Base):
    event: Literal["on_llm_end"]
    output: LLMOutput
    token_usage: Optional[Any] = None


class OnToolStart(_Base):
    event: Literal["on_tool_start"]
    input: dict
    tool: Optional[str]
    tool_description: Optional[str]
    invoked_by_run_id: Optional[str]


class OnToolEnd(_Base):
    event: Literal["on_tool_end"]
    output: Any
    is_mcp_tool: bool = False


class OnError(_Base):
    event: Literal["on_llm_error", "on_tool_error", "on_chain_error"]
    error: str


AgentEvent = Annotated[
    Union[
        OnChainStart, OnChainEnd, OnChatModelStart, OnLlmStart,
        OnLlmEnd, OnToolStart, OnToolEnd, OnError,
    ],
    Field(discriminator="event"),
]


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: dict = Field(default_factory=dict)


class ToolSchema(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction


class AvailableTools(BaseModel):
    tools_hash: str
    tools: list[ToolSchema]


class AgentLoopRequest(BaseModel):
    context: list[AgentEvent]
    trace_id: str
    agent_name: Optional[str] = None
    observability_call: bool = False
    available_tools: list[AvailableTools] = Field(default_factory=list)
