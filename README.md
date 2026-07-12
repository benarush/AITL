# agent-in-the-loop

[![PyPI version](https://img.shields.io/pypi/v/agent-in-the-loop.svg)](https://pypi.org/project/agent-in-the-loop/)
[![Python](https://img.shields.io/pypi/pyversions/agent-in-the-loop.svg)](https://pypi.org/project/agent-in-the-loop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/benarush/AITL/actions/workflows/ci.yml/badge.svg)](https://github.com/benarush/AITL/actions/workflows/ci.yml)

A lightweight Python client for the **Agent In The Loop (AITL)** confidence evaluation API. Send your LLM agent's execution context to the AITL backend and receive a structured confidence score — with optional OpenTelemetry trace ID auto-detection.

---

## Installation

```bash
pip install agent-in-the-loop
```

Requires Python 3.10+.

---

## Quick Start

```python
from agent_in_the_loop import evaluate_confidence

result = evaluate_confidence(
    extra_context="The agent searched the web, found 3 sources, and summarised them.",
    trace_id="your-trace-id-here",
    api_key="your-api-key",
)

print(result.score)  # int, 1-10
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
from agent_in_the_loop import evaluate_confidence

result = evaluate_confidence(
    extra_context="Agent context here...",
    trace_id="your-trace-id",
)
```

---

## OpenTelemetry Integration

If your application already uses OpenTelemetry tracing, `TraceIdCapture` automatically captures the current trace ID so you never need to pass it manually.

```python
from opentelemetry.sdk.trace import TracerProvider
from agent_in_the_loop import TraceIdCapture, evaluate_confidence

# Register the processor once at startup
provider = TracerProvider()
provider.add_span_processor(TraceIdCapture())

tracer = provider.get_tracer("my-agent")

with tracer.start_as_current_span("agent-run"):
    # trace_id is captured automatically — no need to pass it
    result = evaluate_confidence(
        extra_context="Agent finished reasoning step...",
    )
    print(result.score)
```

`TraceIdCapture` implements the OpenTelemetry `SpanProcessor` interface and stores the active trace ID in a `ContextVar`, providing full thread-safety and async-safety.

---

## API Reference

### `evaluate_confidence`

```python
evaluate_confidence(
    context: str,
    trace_id: str | None = None,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> AgentLoopResult
```

| Parameter | Type | Description |
|---|---|---|
| `context` | `str` | Conversation and graph flow to evaluate |
| `trace_id` | `str \| None` | Trace ID for the agent run. Auto-detected when `TraceIdCapture` is registered |
| `api_key` | `str \| None` | Bearer token. Falls back to `AGENT_IN_THE_LOOP_API_KEY` |
| `timeout` | `float` | HTTP request timeout in seconds (default `30.0`) |

Requests are always sent to the fixed backend domain (`https://api.trellar.io`); there is no way for callers to redirect them elsewhere.

**Raises:**
- `ValueError` — if `trace_id` cannot be resolved or `api_key` is missing
- `requests.HTTPError` — on non-2xx HTTP responses

### `AgentLoopResult`

A frozen dataclass with two fields:

| Field | Type | Description |
|---|---|---|
| `score` | `int` | Confidence score from 1 (low) to 10 (high) |
| `explanation` | `str` | Human-readable explanation of the score |

### `TraceIdCapture`

An OpenTelemetry `SpanProcessor` that captures the trace ID on span start. Register it with your `TracerProvider` as shown above.

---

## LangChain Callback

`DebugCallbackHandler` is a LangChain callback that prints every lifecycle event (LLM, tool, and chain/graph) with its full payload. It is useful for inspecting what data is available at each step of an agent run.

### Installation

```bash
pip install "agent-in-the-loop[langchain]"
```

### Usage

```python
from agent_in_the_loop.callbacks.langchain_callback import DebugCallbackHandler

handler = DebugCallbackHandler()

# Attach to an LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(callbacks=[handler])

# Or attach to a LangGraph / chain invocation
result = graph.invoke(inputs, config={"callbacks": [handler]})
```

Each event is printed with a sequential counter, the event name, the `run_id`, and the full payload — making it easy to trace exactly what LangChain passes at every stage.

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
