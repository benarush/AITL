# agent-in-the-loop

[![PyPI version](https://img.shields.io/pypi/v/agent-in-the-loop.svg)](https://pypi.org/project/agent-in-the-loop/)
[![Python](https://img.shields.io/pypi/pyversions/agent-in-the-loop.svg)](https://pypi.org/project/agent-in-the-loop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/AITL/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/AITL/actions/workflows/ci.yml)

A lightweight Python client for the **Agent In The Loop (AITL)** confidence evaluation API. Attach a callback to your LangChain / LangGraph run, then get a structured confidence score for the whole agent run — no manual context or trace-ID wiring required.

---

## Installation

```bash
pip install "agent-in-the-loop[langchain]"
```

The `langchain` extra is required because agent runs are captured via a LangChain callback handler (`get_agent_guard`). Requires Python 3.9+.

---

## Quick Start

```python
from agent_in_the_loop import get_agent_guard, evaluate_confidence

# agent_name must be a stable, unique name for this agent graph — the
# backend uses it to track the graph's network profile across runs.
guard = get_agent_guard("research-agent")

# Attach the guard to your LangGraph / LangChain invocation
graph.invoke(inputs, config={"callbacks": [guard]})

# context, trace_id, and agent_name are all picked up automatically from
# the guard above — nothing else to pass in.
result = evaluate_confidence(api_key="your-api-key")

print(result.score)        # int, 1-10
print(result.explanation)  # str, human-readable reasoning
```

---

## Environment Variables

The SDK always talks to the managed AITL backend at `https://api.trellar.io` — this is fixed and cannot be overridden via an environment variable or function argument.

Instead of passing `api_key` on every call, set it as an environment variable:

| Variable | Description | Default |
|---|---|---|
| `AGENT_IN_THE_LOOP_API_KEY` | Bearer token for authentication | *(required)* |

```bash
export AGENT_IN_THE_LOOP_API_KEY=your-api-key
```

```python
from agent_in_the_loop import get_agent_guard, evaluate_confidence

guard = get_agent_guard("research-agent")
graph.invoke(inputs, config={"callbacks": [guard]})

result = evaluate_confidence()
```

---

## How It Works

`get_agent_guard(agent_name)` returns a LangChain callback handler that:

- Records every LLM, tool, and chain/graph lifecycle event fired during the run (`on_llm_start`, `on_tool_end`, `on_chain_start`, etc.), in order.
- Captures the root run's `run_id` as the `trace_id` for the whole graph.
- Registers itself in a `ContextVar` as the "active" guard for the current thread/async task, so `evaluate_confidence()` can find it automatically — no need to pass it around.

Once the graph run finishes, calling `evaluate_confidence()` sends the accumulated events, trace ID, and agent name to the AITL backend and returns a confidence score.

Different graphs in the same codebase must use different `agent_name` values.

---

## API Reference

### `get_agent_guard`

```python
get_agent_guard(agent_name: str) -> BaseCallbackHandler
```

| Parameter | Type | Description |
|---|---|---|
| `agent_name` | `str` | Stable, unique name identifying this agent graph (e.g. `"research-agent"`) |

Returns a LangChain callback handler bound to `agent_name`. Pass it to `graph.invoke(..., config={"callbacks": [guard]})`.

**Raises:**
- `ValueError` — if `agent_name` is empty or blank

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
| `api_key` | `str \| None` | Bearer token. Falls back to `AGENT_IN_THE_LOOP_API_KEY` |
| `timeout` | `float` | HTTP request timeout in seconds (default `30.0`) |

`context`, `trace_id`, and `agent_name` are all resolved automatically from the active guard created by `get_agent_guard` — there is no way to pass them manually. Requests are always sent to the fixed backend domain (`https://api.trellar.io`); there is no way for callers to redirect them elsewhere.

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

## Running Tests

```bash
pip install -e ".[test]"
pytest
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.
