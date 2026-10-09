# trellar

[![PyPI version](https://img.shields.io/pypi/v/trellar.svg)](https://pypi.org/project/trellar/)
[![Python](https://img.shields.io/pypi/pyversions/trellar.svg)](https://pypi.org/project/trellar/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/AITL/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/AITL/actions/workflows/ci.yml)

The Python SDK for **Trellar**, the trust layer for autonomous multi-agent systems. Trellar is an AI governance platform that sits above your existing agent frameworks (LangChain, LangGraph, Strands Agents and more) and gives your team visibility and control over what your agents do in production, so you can extend their autonomy with confidence.

Connecting takes one callback on your agent run. Trellar monitors the run outside the execution path, so it adds no latency to your agents, and it evaluates the run against what each agent is meant to do. Trellar can evaluate each run automatically when it finishes. For finer control, call `evaluate_confidence()` wherever you want a score, either to gate the next step or just to record it. Context, trace ID, and agent name are picked up automatically — no manual wiring.

This library cannot be used without an API key from [trellar.io](https://trellar.io).

---

## Create an account and API key

[trellar.io](https://trellar.io) is the only place that issues API keys for this library. Create an account there, then generate an API key from the dashboard. Without that key, `evaluate_confidence()` cannot authenticate and the client will not work.

Then pass the key into the SDK (see [Environment Variables](#environment-variables)):

- `evaluate_confidence(api_key="...")`, or
- `TRELLAR_API_KEY` in the environment

---

## Installation

Install `trellar` with the extra for the framework you use:

```bash
pip install "trellar[langchain]"   # LangChain / LangGraph (Python 3.9+)
pip install "trellar[strands]"     # Strands Agents (Python 3.10+)
pip install "trellar[langchain,strands]"   # both (Python 3.10+)
```

An extra is required because agent runs are captured through the framework's own hooks: a LangChain callback handler (`trellar_langchain_agent`) or a Strands hook provider (`trellar_strands_agent`). The base package requires Python 3.9+; the `strands` extra requires Python 3.10+.

---

## Quick Start

Start by observing your agent. Set your API key, add the Trellar callback to your run, and Trellar evaluates the run automatically when it finishes. You don't need to call `evaluate_confidence()` yourself.

```bash
export TRELLAR_API_KEY=your-api-key
```

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from trellar import trellar_langchain_agent, ObservabilityMode

@tool
def search_docs(query: str) -> str:
    """Search the internal docs."""
    return "The office opens at 9am."

graph = create_react_agent(model, tools=[search_docs])

# agent_name must be a stable, unique name for this agent graph — the
# backend uses it to track the graph's network profile across runs.
trellar_agent = trellar_langchain_agent("research-agent", ObservabilityMode.ALWAYS)

# The run is captured and evaluated when invoke() finishes.
# Context, trace_id, and agent_name are picked up automatically.
graph.invoke(
    {"messages": [("user", "What time does the office open?")]},
    config={"callbacks": [trellar_agent]},
)
```

The run is sent to Trellar when `invoke()` returns, and the result appears in your dashboard at [trellar.io](https://trellar.io).

> **Evaluation needs some context.** Trellar scores the run from the events it captured, so the run should include at least one agent call or tool call. A run with nothing in it gives the evaluation too little to work with.

Want to read the score in your code, or stop the graph when it's low? See [Where to call `evaluate_confidence`](#where-to-call-evaluate_confidence).

---

## Strands Agents

```bash
pip install "trellar[strands]"   # requires Python 3.10+
```

```python
from strands import Agent
from strands.multiagent import GraphBuilder
from trellar import trellar_strands_agent, evaluate_confidence

trellar_agent = trellar_strands_agent("research-agent")

# Give every agent a stable name.
searcher = Agent(name="searcher")
reporter = Agent(name="reporter")

# Register the Trellar agent on the Graph/Swarm: the whole run is one trace, and every
# node's Agent is bound automatically (no need for `hooks=[trellar_agent]` on each one).
builder = GraphBuilder()
builder.add_node(searcher, "searcher")
builder.add_node(reporter, "reporter")
builder.add_edge("searcher", "reporter")
builder.set_entry_point("searcher")
builder.set_hook_providers([trellar_agent])
graph = builder.build()

graph("Find me something to report on")
```

A standalone Agent (no Graph/Swarm) still needs `Agent(hooks=[trellar_agent])`. Passing it explicitly to a graph's Agents as well is harmless: registering twice is a no-op. Once bound, an Agent keeps the hook (Strands cannot remove it).

Call `evaluate_confidence()` from inside the run (a graph node or a tool), exactly as with LangChain. `ObservabilityMode` works the same way. See `orchestrations_examples/our_lab_with_aviran/car_stocks_buy_mcp_celery.py/car_stocks_buy_from_remote_provider_orchestration_strands.py` for a full example.

### `trellar_strands_single_call`

For one Agent called once (no Graph/Swarm). The run is over when the call returns, so use `ObservabilityMode.ALWAYS` (or `IF_NOT_EVALUATED`) and read the result off the Trellar agent:

```python
from strands import Agent
from trellar import trellar_strands_single_call, ObservabilityMode

trellar_agent = trellar_strands_single_call("faq-agent", ObservabilityMode.ALWAYS)
agent = Agent(name="faq_agent", hooks=[trellar_agent])
agent("What time does the office open?")

result = trellar_agent.trellar_evaluate_result   # AgentLoopResult, or None
error = trellar_agent.trellar_evaluate_error     # the exception, if the auto-triggered call failed
```

Requests are marked `single_call: true` in the payload. Not covered: `agent.structured_output()` and calling a Strands `Model` directly — Strands fires no model-call hooks for them. See `orchestrations_examples/our_lab_with_aviran/single_llm_agent/simple_llm_call_strands.py`.

---

## Where to call `evaluate_confidence`

Call it from a graph node, at the point in the run you want scored, while the run is still in progress — the callback handler is released as soon as the root run ends, so calling it after `invoke()` returns raises `ValueError`. The payload is the events captured **so far** — later nodes are not included.

There are two ways to get a score:

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

### 2. Observe — record a score, do not restrict the graph

If you only want the run scored and recorded, you don't need a node or a manual call. Set an `ObservabilityMode` when you create the Trellar agent, and Trellar evaluates the run when it finishes. The graph is never affected.

```python
from trellar import trellar_langchain_agent, ObservabilityMode

trellar_agent = trellar_langchain_agent("research-agent", ObservabilityMode.ALWAYS)
graph.invoke(inputs, config={"callbacks": [trellar_agent]})
```

Use `ObservabilityMode.IF_NOT_EVALUATED` to combine both ways: gate nodes score the run where you need them, and any run that no node scored is still evaluated when it finishes. See [`ObservabilityMode`](#observabilitymode) for the full list of values.

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

> **Renamed in this release.** The `get_*_guard` functions are now `trellar_<framework>_agent` / `trellar_<framework>_single_call`. The old names still work but emit a `DeprecationWarning` and will be removed in a future release.
>
> | Deprecated | Use instead |
> |---|---|
> | `get_agent_guard` | `trellar_langchain_agent` |
> | `get_single_call_guard` | `trellar_langchain_single_call` |
> | `get_strands_guard` | `trellar_strands_agent` |
> | `get_strands_single_call_guard` | `trellar_strands_single_call` |

### `trellar_langchain_agent`

```python
trellar_langchain_agent(
    agent_name: str,
    observability_mode: ObservabilityMode = ObservabilityMode.NONE,
) -> BaseCallbackHandler
```

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` | Stable, unique name identifying this agent graph (e.g. `"research-agent"`) |
| `observability_mode` | `ObservabilityMode` | Controls whether `evaluate_confidence()` is auto-triggered when the graph run finishes. Default `ObservabilityMode.NONE` (no auto-trigger). |

Returns a LangChain callback handler bound to `agent_name`. Pass it to `graph.invoke(..., config={"callbacks": [trellar_agent]})`.

**Raises:**
- `ValueError` — if `agent_name` is empty or blank

#### `ObservabilityMode`

Controls whether the Trellar agent automatically calls `evaluate_confidence()` for you when the graph run finishes (the root `graph.invoke()` call completes), so you don't have to add a manual call yourself.

| Value | Behavior |
|---|---|
| `ObservabilityMode.NONE` | Never auto-call. Default; identical to not passing `observability_mode` at all. |
| `ObservabilityMode.ALWAYS` | Always call `evaluate_confidence()` when the run finishes. |
| `ObservabilityMode.IF_NOT_EVALUATED` | Call `evaluate_confidence()` when the run finishes only if it was not already successfully called earlier in the run (e.g. from a gate node). |

```python
from trellar import trellar_langchain_agent, ObservabilityMode

trellar_agent = trellar_langchain_agent("research-agent", ObservabilityMode.IF_NOT_EVALUATED)
graph.invoke(inputs, config={"callbacks": [trellar_agent]})
# evaluate_confidence() has already run automatically if no node called it.
```

Auto-triggered calls never raise: any error (missing API key, HTTP error, `NetworkHaltedError`, etc.) is caught and logged instead of propagating out of `graph.invoke()`. A manual call to `evaluate_confidence()` still raises normally.

Requests triggered this way are marked in the payload sent to the backend with `observability_call: true` (`false` for a normal, manually-invoked call), so the backend can distinguish automatic observability calls from explicit ones.

### `trellar_langchain_single_call`

```python
trellar_langchain_single_call(
    agent_name: str,
    observability_mode: ObservabilityMode = ObservabilityMode.NONE,
) -> BaseCallbackHandler
```

Use this instead of `trellar_langchain_agent` when you are calling a chat model directly (`llm.invoke(...)`) with no wrapping LangGraph/chain. Agents built with `create_react_agent` are already compiled graphs under the hood, so they work with `trellar_langchain_agent` as usual — `trellar_langchain_single_call` is only for a genuinely bare model call.

A bare `llm.invoke()` call has no node to call `evaluate_confidence()` from mid-run, and the callback handler is released as soon as the call finishes — so a manual call is never supported here. Use `ObservabilityMode.ALWAYS` (or `IF_NOT_EVALUATED`) to auto-trigger the evaluation, then read the result back from the Trellar agent:

```python
from trellar import trellar_langchain_single_call, ObservabilityMode

trellar_agent = trellar_langchain_single_call("single-llm-call", ObservabilityMode.ALWAYS)
llm.invoke(messages, config={"callbacks": [trellar_agent]})

result = trellar_agent.trellar_evaluate_result   # AgentLoopResult, or None if not yet evaluated
error = trellar_agent.trellar_evaluate_error     # the exception, if the auto-triggered call failed
```

Requests made through this Trellar agent are marked in the payload with `single_call: true` (`false` for `trellar_langchain_agent`), so the backend can tell the two apart.

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

`context`, `trace_id`, and `agent_name` are resolved automatically from the active Trellar agent created by `trellar_langchain_agent` — there is no way to pass them manually. Requests always go to `https://api.trellar.io`; callers cannot redirect them.

**Raises:**
- `ValueError` — if no active Trellar agent is found, its `trace_id` cannot be resolved, or `api_key` is missing
- `requests.HTTPError` — on non-2xx HTTP responses

### `AgentLoopResult`

A frozen dataclass with:

| Field | Type | Description |
|---|---|---|
| `score` | `int` | Confidence score from 1 (low) to 10 (high) |
| `explanation` | `str` | Human-readable explanation of the score |
| `decision_identifier` | `str` | Unique ID for this evaluation, for cross-referencing with the backend |
| `should_stop_network` | `bool` | Whether the backend signaled the run should halt (see `NetworkHaltedError`) |


---
## License

MIT — see [LICENSE](LICENSE) for details.
