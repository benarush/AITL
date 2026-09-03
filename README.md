# trellar

[![PyPI version](https://img.shields.io/pypi/v/trellar.svg)](https://pypi.org/project/trellar/)
[![Python](https://img.shields.io/pypi/pyversions/trellar.svg)](https://pypi.org/project/trellar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/AITL/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/AITL/actions/workflows/ci.yml)

A lightweight Python client for the **Trellar** confidence evaluation API. Attach a callback to your LangChain / LangGraph run, then call `evaluate_confidence()` when you want a score. Context, trace ID, and agent name are picked up automatically — no manual wiring.

This library cannot be used without an API key from [trellar.io](https://trellar.io).

---

## Create an account and API key

[trellar.io](https://trellar.io) is the only place that issues API keys for this library. Create an account there, then generate an API key from the dashboard. Without that key, `evaluate_confidence()` cannot authenticate and the client will not work.

Then pass the key into the SDK (see [Environment Variables](#environment-variables)):

- `evaluate_confidence(api_key="...")`, or
- `TRELLAR_API_KEY` in the environment

---

## Installation

```bash
pip install "trellar[langchain]"
```

The `langchain` extra is required because agent runs are captured via a LangChain callback handler (`get_agent_guard`). Requires Python 3.9+.

---

## Quick Start

```python
from trellar import get_agent_guard, evaluate_confidence

# agent_name must be a stable, unique name for this agent graph — the
# backend uses it to track the graph's network profile across runs.
guard = get_agent_guard("research-agent")

graph.invoke(inputs, config={"callbacks": [guard]})

# context, trace_id, and agent_name are picked up from the guard
result = evaluate_confidence()
print(result.score)        # int, 1-10
print(result.explanation)  # str, human-readable reasoning
```

---

## Where to call `evaluate_confidence`

Call it from a graph node (or after `invoke()`), at the point in the run you want scored. The payload is the events captured **so far** — later nodes are not included.

There are two ways to use the result:

### 1. Gate — validate before the graph continues

Put the call on an edge you do not want the graph to cross until Trellar has scored the run. Use `result.score` / `result.explanation` to decide whether to proceed or stop.

```python
def confidence_gate(state):
    result = evaluate_confidence()
    if result.score < 7:
        return {**state, "halt": True, "reason": result.explanation}
    return {**state, "halt": False}
```

Wire that node in front of the next step, and only continue when the score is acceptable.

### 2. Observe — send a validation, do not restrict the graph

Put the call anywhere you want a score recorded (a node, or after `invoke()`). Store or log `result` if you want it; do not branch on it. The graph continues either way.

```python
def report_confidence(state):
    result = evaluate_confidence()
    return {**state, "confidence_score": result.score, "confidence_explanation": result.explanation}
```

---

## Environment Variables

The SDK always talks to the managed Trellar backend at `https://api.trellar.io` — this is fixed and cannot be overridden via an environment variable or function argument.

The API key itself is created only at [trellar.io](https://trellar.io). Once you have it, you can pass it to `evaluate_confidence(api_key=...)` or set it as an environment variable so you do not pass it on every call:

| Variable | Description | Default |
|---|---|---|
| `TRELLAR_API_KEY` | Bearer token for authentication | *(required)* |

```bash
export TRELLAR_API_KEY=your-api-key
```

```python
result = evaluate_confidence()  # api_key read from the env var
```

---

## API Reference

### `get_agent_guard`

```python
get_agent_guard(
    agent_name: str,
    observability_mode: ObservabilityMode = ObservabilityMode.NONE,
) -> BaseCallbackHandler
```

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` | Stable, unique name identifying this agent graph (e.g. `"research-agent"`) |
| `observability_mode` | `ObservabilityMode` | Controls whether `evaluate_confidence()` is auto-triggered when the graph run finishes. Default `ObservabilityMode.NONE` (no auto-trigger). |

Returns a LangChain callback handler bound to `agent_name`. Pass it to `graph.invoke(..., config={"callbacks": [guard]})`.

**Raises:**
- `ValueError` — if `agent_name` is empty or blank

#### `ObservabilityMode`

Controls whether the guard automatically calls `evaluate_confidence()` for you when the graph run finishes (the root `graph.invoke()` call completes), so you don't have to add a manual call yourself.

| Value | Behavior |
|---|---|
| `ObservabilityMode.NONE` | Never auto-call. Default; identical to not passing `observability_mode` at all. |
| `ObservabilityMode.ALWAYS` | Always call `evaluate_confidence()` when the run finishes. |
| `ObservabilityMode.IF_NOT_EVALUATED` | Call `evaluate_confidence()` when the run finishes only if it was not already successfully called earlier in the run (e.g. from a gate node). |

```python
from trellar import get_agent_guard, ObservabilityMode

guard = get_agent_guard("research-agent", ObservabilityMode.IF_NOT_EVALUATED)
graph.invoke(inputs, config={"callbacks": [guard]})
# evaluate_confidence() has already run automatically if no node called it.
```

Auto-triggered calls never raise: any error (missing API key, HTTP error, `NetworkHaltedError`, etc.) is caught and logged instead of propagating out of `graph.invoke()`. A manual call to `evaluate_confidence()` still raises normally.

Requests triggered this way are marked in the payload sent to the backend with `observability_call: true` (`false` for a normal, manually-invoked call), so the backend can distinguish automatic observability calls from explicit ones.

### `evaluate_confidence`

```python
evaluate_confidence(
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> AgentLoopResult
```

| Parameter | Type | Description |
|---|---|---|
| `api_key` | `str \| None` | Bearer token. Falls back to `TRELLAR_API_KEY` |
| `timeout` | `float` | HTTP request timeout in seconds (default `30.0`) |

`context`, `trace_id`, and `agent_name` are resolved automatically from the active guard created by `get_agent_guard` — there is no way to pass them manually. Requests always go to `https://api.trellar.io`; callers cannot redirect them.

**Raises:**
- `ValueError` — if no active guard is found, its `trace_id` cannot be resolved, or `api_key` is missing
- `requests.HTTPError` — on non-2xx HTTP responses

### `AgentLoopResult`

A frozen dataclass with two fields:

| Field | Type | Description |
|---|---|---|
| `score` | `int` | Confidence score from 1 (low) to 10 (high) |
| `explanation` | `str` | Human-readable explanation of the score |


---
## License

MIT — see [LICENSE](LICENSE) for details.
