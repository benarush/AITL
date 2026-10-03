"""Scripted fake Strands model + helpers, so tests drive real Agents/Graphs
without any network or API key."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from strands.models.model import Model


def text_turn(text: str) -> list[dict]:
    """Stream chunks for a plain text answer."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": text}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}},
    ]


def tool_turn(name: str, args: dict, tool_use_id: str = "tu-1") -> list[dict]:
    """Stream chunks for a model turn that asks to call one tool."""
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}},
    ]


class FakeModel(Model):
    """Replays one scripted turn per model call; raises if a turn is an Exception."""

    def __init__(self, turns: list[Any], model_id: str = "fake-model") -> None:
        self.turns = list(turns)
        self.config = {"model_id": model_id}

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    def get_config(self) -> Any:
        return self.config

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:  # not used
        raise NotImplementedError

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[dict]:
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        for chunk in turn:
            yield chunk


def events_of(guard, name: Optional[str] = None) -> list[dict]:
    """Recorded events, optionally filtered by event name."""
    return [e for e in guard.events if name is None or e["event"] == name]
