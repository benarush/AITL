"""Shared fixtures for the reasoning-extraction tests.

Each ``*_config`` fixture is a model config shaped like what the real SDK hands the
callbacks (LangChain ``serialized["kwargs"]``, Strands ``get_config()``), always
polluted with secret-looking fields so the tests also prove they never leak.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from trellar.callbacks.langchain.langchain_callback import _LangchainAgentCallback

SECRET_FIELDS = {
    "api_key": "sk-super-secret",
    "openai_api_key": {"lc": 1, "type": "secret", "id": ["OPENAI_API_KEY"]},
    "headers": {"Authorization": "Bearer top-secret"},
}


@pytest.fixture
def secret_fields() -> dict[str, Any]:
    """Fields that must never show up in an extraction result."""
    return dict(SECRET_FIELDS)


@pytest.fixture
def anthropic_thinking() -> dict[str, Any]:
    return {"type": "enabled", "budget_tokens": 2000}


@pytest.fixture
def anthropic_config(secret_fields, anthropic_thinking) -> dict[str, Any]:
    return {"model": "claude-sonnet-4-5", "thinking": anthropic_thinking, **secret_fields}


@pytest.fixture
def gemini_config(secret_fields) -> dict[str, Any]:
    return {
        "model": "gemini-2.5-flash",
        "thinking_budget": 1024,
        "include_thoughts": True,
        **secret_fields,
    }


@pytest.fixture
def openai_config(secret_fields) -> dict[str, Any]:
    return {"model_name": "o3-mini", "reasoning_effort": "high", **secret_fields}


@pytest.fixture
def strands_gemini_config(secret_fields) -> dict[str, Any]:
    """Strands keeps Gemini's thinking inside ``params`` as a nested ``thinking_config``."""
    return {
        "model_id": "gemini-2.5-flash",
        "params": {"temperature": 0.2, "thinking_config": {"thinking_budget": 512}},
        **secret_fields,
    }


@pytest.fixture
def bedrock_claude_config(secret_fields, anthropic_thinking) -> dict[str, Any]:
    return {
        "model_id": "anthropic.claude-sonnet-4-5",
        "additional_request_fields": {"thinking": anthropic_thinking},
        **secret_fields,
    }


@pytest.fixture
def handler() -> _LangchainAgentCallback:
    """A LangChain callback with a started root chain, ready to record model events."""
    cb = _LangchainAgentCallback(agent_name="test-agent")
    cb.on_chain_start({"name": "graph"}, {"messages": []}, run_id=uuid.uuid4())
    return cb
