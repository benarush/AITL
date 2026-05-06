from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import requests
from opentelemetry.sdk.trace import SpanProcessor

from . import settings
from ._context import _current_callback

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, Span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level trace capture -- set by TraceIdCapture SpanProcessor
# ---------------------------------------------------------------------------
_current_trace_id: ContextVar[Optional[str]] = ContextVar('_current_trace_id', default=None)


class TraceIdCapture(SpanProcessor):
    """Lightweight SpanProcessor that remembers the latest trace_id.

    Usage::

        from agent_in_the_loop import TraceIdCapture

        trace_capture = TraceIdCapture()
        tracer_provider.add_span_processor(trace_capture)

        # Later – evaluate_confidence() auto-detects the trace_id.
    """

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        tid = span.get_span_context().trace_id  # type: ignore[union-attr]
        _current_trace_id.set(format(tid, '032x'))

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 0) -> bool:
        return True


@dataclass(frozen=True)
class AgentLoopResult:
    explanation: str
    score: int


def evaluate_confidence(
    extra_context: Optional[str] = None,
    *,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> AgentLoopResult:
    """Call the Agent-in-the-Loop backend to get a confidence score.

    Both ``context`` and ``trace_id`` are resolved automatically when a
    ``DebugCallbackHandler`` was used during the graph run — no manual
    wiring needed::

        handler = DebugCallbackHandler()
        graph.invoke(input, config={"callbacks": [handler]})
        result = evaluate_confidence()   # fully seamless

    Explicit values always take precedence over auto-detected ones.

    Args:
        extra_context:  Conversation + graph flow with tool-calling results.
                  Auto-built from the active ``DebugCallbackHandler``'s
                  collected events when omitted.
        endpoint: Base URL of the AITL backend.
                  Defaults to ``AGENT_IN_THE_LOOP_ENDPOINT`` env var.
        api_key:  Bearer token for authentication.
                  Defaults to ``AGENT_IN_THE_LOOP_API_KEY`` env var.
        timeout:  HTTP request timeout in seconds.

    Returns:
        AgentLoopResult with ``explanation`` and ``score`` (1-10).

    Raises:
        requests.HTTPError: On non-2xx responses.
        ValueError: When context, trace_id, or api_key cannot be resolved.
    """
    callback = _current_callback.get()

    if callback is None:
        raise ValueError(
            "context could not be resolved. Either pass it explicitly or "
            "use a DebugCallbackHandler so it can be built automatically."
        )

    # Resolve trace_id: explicit → callback → TraceIdCapture ContextVar
    callback_trace_id = str(callback.trace_id) if (callback and callback.trace_id) else None
    resolved_trace_id = callback_trace_id or _current_trace_id.get()

    if not resolved_trace_id:
        raise ValueError(
            "trace_id could not be resolved. Either pass it explicitly, use a "
            "DebugCallbackHandler, or register a TraceIdCapture processor on "
            "your tracer provider."
        )

    graph_context = callback.events

    base_url = (endpoint or settings.get_env_endpoint()).rstrip("/")
    key = api_key or settings.get_env_api_key()
    if not key:
        raise ValueError(
            "api_key must be provided or set via "
            f"{settings.ENV_AITL_API_KEY} environment variable."
        )

    url = f"{base_url}/agent-gateway/v1/agent-loop"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {"context": graph_context, "trace_id": resolved_trace_id}

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    return AgentLoopResult(
        explanation=data["explanation"],
        score=data["score"],
    )
