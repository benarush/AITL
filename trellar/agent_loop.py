from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

import requests

from . import settings
from ._context import _current_callback

if TYPE_CHECKING:
    from .callbacks.langchain_callback import _AgentGuardCallback
    from .callbacks.single_call_callback import _SingleCallGuardCallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentLoopResult:
    explanation: str
    score: int
    decision_identifier: str
    should_stop_network: bool


class ObservabilityMode(str, Enum):
    """Controls whether the guard auto-triggers ``evaluate_confidence()`` when
    the graph's root run finishes (i.e. ``graph.invoke()`` is about to return).

    * ``ALWAYS``           — always auto-call at the end of the run.
    * ``IF_NOT_EVALUATED``  — auto-call at the end only if
      ``evaluate_confidence()`` was not already successfully called anywhere
      during the run.
    * ``NONE``             — never auto-call (default; current behavior).

    Errors raised by an auto-triggered call are caught and logged, never
    propagated out of ``graph.invoke()``.
    """

    ALWAYS = "always"
    IF_NOT_EVALUATED = "if_not_evaluated"
    NONE = "none"


class NetworkHaltedError(Exception):
    """Raised when the Trellar backend signals the agent network must stop."""

    def __init__(self, explanation: str, score: int, decision_identifier: str):
        self.explanation = explanation
        self.score = score
        self.decision_identifier = decision_identifier
        super().__init__(
            f"Trellar halted the agent network (decision_identifier={decision_identifier}): {explanation}"
        )


def get_agent_guard(
    agent_name: str,
    observability_mode: ObservabilityMode = ObservabilityMode.NONE,
) -> "_AgentGuardCallback":
    """Create a callback handler that identifies this graph to the Trellar backend.

    ``agent_name`` must be a stable, unique name for this agent graph within
    your repository (e.g. ``'research-agent'``, ``'support-bot'``). The backend
    uses it to look up and maintain the graph's network profile across runs.
    Different graphs in the same repo must use different names.

    Usage::

        guard = get_agent_guard("research-agent")

        def confidence_gate(state):
            result = evaluate_confidence()
            ...

        graph.invoke(input, config={"callbacks": [guard]})

    ``evaluate_confidence()`` must be called from inside a graph node, while
    the run is still in progress — not after ``graph.invoke()`` returns. The
    callback handler is released as soon as the root run ends, so a call made
    after ``invoke()`` returns will raise ``ValueError``.

    Args:
        agent_name: Unique, stable name for this agent graph.
        observability_mode: Controls whether ``evaluate_confidence()`` is
            auto-triggered when the graph run finishes. See
            :class:`ObservabilityMode`. Defaults to ``ObservabilityMode.NONE``
            (no auto-trigger, current behavior).

    Returns:
        An internal callback handler bound to the given agent name.
    """
    from .callbacks.langchain_callback import _AgentGuardCallback
    return _AgentGuardCallback(agent_name=agent_name, observability_mode=observability_mode)


def get_single_call_guard(
    agent_name: str,
    observability_mode: ObservabilityMode = ObservabilityMode.NONE,
) -> "_SingleCallGuardCallback":
    """Create a callback handler for a single bare LLM call (no LangGraph/chain wrapper).

    Use this instead of :func:`get_agent_guard` when you are calling a chat
    model directly (e.g. ``llm.invoke(...)``) rather than invoking a graph or
    an agent built with ``create_react_agent`` (which is itself a compiled
    graph, and already works with :func:`get_agent_guard`).

    A bare ``llm.invoke()`` call has no node to call ``evaluate_confidence()``
    from mid-run, and the callback handler is released as soon as the call
    finishes — so a manual call is never supported here. Use
    ``observability_mode=ObservabilityMode.ALWAYS`` (or ``IF_NOT_EVALUATED``)
    to auto-trigger the evaluation, then read the result off the guard::

        guard = get_single_call_guard("single-llm-call", ObservabilityMode.ALWAYS)
        llm.invoke(messages, config={"callbacks": [guard]})
        result = guard.trellar_evaluate_result

    Args:
        agent_name: Unique, stable name for this agent within your repository.
        observability_mode: Controls whether ``evaluate_confidence()`` is
            auto-triggered when the LLM call finishes. See
            :class:`ObservabilityMode`. Defaults to ``ObservabilityMode.NONE``
            (no auto-trigger).

    Returns:
        An internal callback handler bound to the given agent name. Exposes
        ``trellar_evaluate_result``/``trellar_evaluate_error`` for reading the outcome of an
        auto-triggered evaluation.
    """
    from .callbacks.single_call_callback import _SingleCallGuardCallback
    return _SingleCallGuardCallback(agent_name=agent_name, observability_mode=observability_mode)


def evaluate_confidence(
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
    _observability_call: bool = False,
) -> AgentLoopResult:
    """Call the Trellar backend to get a confidence score.

    ``context``, ``trace_id``, and ``agent_name`` are all resolved automatically
    from the active guard created by :func:`get_agent_guard` — no manual wiring needed.
    Must be called from inside a graph node while the run is still in progress,
    not after ``graph.invoke()`` returns::

        guard = get_agent_guard("research-agent")

        def confidence_gate(state):
            result = evaluate_confidence()
            ...

        graph.invoke(input, config={"callbacks": [guard]})

    Args:
        api_key:  Bearer token for authentication.
                  Defaults to the ``TRELLAR_API_KEY`` env var.
        timeout:  HTTP request timeout in seconds (default 30).
        _observability_call: Internal — set by the guard's auto-trigger
                  (see ``ObservabilityMode``) to mark the request as
                  automatic rather than a manual call. Not for external use.

    Returns:
        :class:`AgentLoopResult` with ``explanation`` and ``score`` (1–10).

    Raises:
        requests.HTTPError: On non-2xx responses.
        ValueError: When the callback handler, trace_id, or api_key cannot be resolved.
        NetworkHaltedError: When the backend signals that the agent network must stop.
    """
    callback = _current_callback.get()

    if callback is None:
        raise ValueError(
            "No active callback handler found. Use get_agent_guard() to create one "
            "and pass it to graph.invoke() before calling evaluate_confidence()."
        )

    if not callback.trace_id:
        raise ValueError(
            "trace_id could not be resolved. Make sure get_agent_guard() is passed to "
            "graph.invoke() before calling evaluate_confidence()."
        )
    resolved_trace_id = str(callback.trace_id)

    base_url = settings.DEFAULT_ENDPOINT.rstrip("/")
    key = api_key or settings.get_env_api_key()
    if not key:
        raise ValueError(
            "api_key must be provided or set via "
            f"{settings.ENV_TRELLAR_API_KEY} environment variable."
        )

    url = f"{base_url}/agent-gateway/v1/agent-loop"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "context": callback.events,
        "trace_id": resolved_trace_id,
        "agent_name": callback.agent_name,
        "observability_call": _observability_call,
        "single_call": getattr(callback, "is_single_call", False),
        # One entry per distinct model bound during this run, not per LLM call —
        # see _AgentGuardCallback.available_tools.
        "available_tools": [
            {"model": model, "tools": tools}
            for model, tools in callback.available_tools.items()
        ],
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    result = AgentLoopResult(
        explanation=data["explanation"],
        score=data["score"],
        decision_identifier=data["decision_identifier"],
        should_stop_network=data["should_stop_network"],
    )
    callback._evaluated = True

    if result.should_stop_network and not _observability_call :
        raise NetworkHaltedError(
            explanation=result.explanation,
            score=result.score,
            decision_identifier=result.decision_identifier,
        )

    return result
