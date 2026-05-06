from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider

from agent_in_the_loop.agent_loop import (
    AgentLoopResult,
    TraceIdCapture,
    _current_trace_id,
    evaluate_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span(trace_id: int) -> MagicMock:
    """Return a mock span whose get_span_context().trace_id == trace_id."""
    span_ctx = MagicMock()
    span_ctx.trace_id = trace_id
    span = MagicMock()
    span.get_span_context.return_value = span_ctx
    return span


def _successful_post(score: int = 8, explanation: str = "looks good"):
    """Return a mock requests.post response with a JSON body."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"score": score, "explanation": explanation}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture(autouse=True)
def _reset_trace_id():
    """Reset the ContextVar before and after every test."""
    token = _current_trace_id.set(None)
    yield
    _current_trace_id.reset(token)


# ---------------------------------------------------------------------------
# TraceIdCapture — basic behaviour
# ---------------------------------------------------------------------------

class TestTraceIdCapture:
    def test_on_start_stores_hex_trace_id(self):
        processor = TraceIdCapture()
        span = _make_span(trace_id=0xABCDEF1234567890ABCDEF1234567890)

        processor.on_start(span)

        assert _current_trace_id.get() == "abcdef1234567890abcdef1234567890"

    def test_on_start_overwrites_previous(self):
        processor = TraceIdCapture()

        processor.on_start(_make_span(trace_id=1))
        processor.on_start(_make_span(trace_id=2))

        assert _current_trace_id.get() == format(2, "032x")

    def test_on_start_pads_to_32_chars(self):
        processor = TraceIdCapture()
        processor.on_start(_make_span(trace_id=0xFF))

        result = _current_trace_id.get()
        assert len(result) == 32
        assert result == "000000000000000000000000000000ff"

    def test_force_flush_returns_true(self):
        assert TraceIdCapture().force_flush() is True

    def test_integrates_with_real_tracer_provider(self):
        provider = TracerProvider()
        capture = TraceIdCapture()
        provider.add_span_processor(capture)

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("my-span") as span:
            expected = format(span.get_span_context().trace_id, "032x")

        assert _current_trace_id.get() == expected
        provider.shutdown()


# ---------------------------------------------------------------------------
# TraceIdCapture — thread safety
# ---------------------------------------------------------------------------

class TestTraceIdCaptureThreadSafety:
    def test_threads_get_isolated_trace_ids(self):
        """Each thread should see only the trace_id it set itself."""
        processor = TraceIdCapture()
        barrier = threading.Barrier(4)
        results: dict[int, Optional[str]] = {}

        def worker(tid_int: int) -> None:
            ctx = copy_context()

            def _inner():
                processor.on_start(_make_span(trace_id=tid_int))
                barrier.wait(timeout=5)
                results[tid_int] = _current_trace_id.get()

            ctx.run(_inner)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        for tid_int, captured in results.items():
            assert captured == format(tid_int, "032x"), (
                f"Thread with trace_id={tid_int} saw {captured}"
            )

    def test_thread_pool_executor_isolation(self):
        processor = TraceIdCapture()

        def work(tid_int: int) -> tuple[int, str]:
            ctx = copy_context()

            def _inner():
                processor.on_start(_make_span(trace_id=tid_int))
                return _current_trace_id.get()

            return tid_int, ctx.run(_inner)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work, i) for i in range(20)]
            for f in as_completed(futures):
                tid_int, captured = f.result()
                assert captured == format(tid_int, "032x")


# ---------------------------------------------------------------------------
# TraceIdCapture — async safety
# ---------------------------------------------------------------------------

class TestTraceIdCaptureAsyncSafety:
    def test_async_tasks_get_isolated_trace_ids(self):
        processor = TraceIdCapture()

        async def _task(tid_int: int, ready: asyncio.Event, go: asyncio.Event) -> tuple[int, str]:
            processor.on_start(_make_span(trace_id=tid_int))
            ready.set()
            await go.wait()
            return tid_int, _current_trace_id.get()

        async def _run():
            go = asyncio.Event()
            readies = [asyncio.Event() for _ in range(10)]
            tasks = [asyncio.create_task(_task(i, readies[i], go)) for i in range(10)]
            for r in readies:
                await r.wait()
            go.set()
            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        for tid_int, captured in results:
            assert captured == format(tid_int, "032x"), (
                f"Async task with trace_id={tid_int} saw {captured}"
            )


# ---------------------------------------------------------------------------
# evaluate_confidence
# ---------------------------------------------------------------------------

class TestEvaluateConfidence:
    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_uses_explicit_trace_id(self, mock_post):
        mock_post.return_value = _successful_post()

        result = evaluate_confidence(
            extra_context="test context",
            trace_id="explicit-id",
            endpoint="http://test:8000",
            api_key="key-123",
        )

        assert isinstance(result, AgentLoopResult)
        assert result.score == 8
        payload = mock_post.call_args.kwargs["json"]
        assert payload["trace_id"] == "explicit-id"

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_auto_detects_trace_id_from_contextvar(self, mock_post):
        mock_post.return_value = _successful_post()
        processor = TraceIdCapture()
        processor.on_start(_make_span(trace_id=42))

        evaluate_confidence(
            extra_context="test context",
            endpoint="http://test:8000",
            api_key="key-123",
        )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["trace_id"] == format(42, "032x")

    def test_raises_when_no_trace_id(self):
        with pytest.raises(ValueError, match="trace_id could not be resolved"):
            evaluate_confidence(
                extra_context="test",
                endpoint="http://test:8000",
                api_key="key-123",
            )

    @patch("agent_in_the_loop.agent_loop.settings")
    def test_raises_when_no_api_key(self, mock_settings):
        mock_settings.get_env_endpoint.return_value = "http://test:8000"
        mock_settings.get_env_api_key.return_value = None
        mock_settings.ENV_AITL_API_KEY = "AGENT_IN_THE_LOOP_API_KEY"

        with pytest.raises(ValueError, match="api_key must be provided"):
            evaluate_confidence(extra_context="test", trace_id="abc")

    @patch("agent_in_the_loop.agent_loop.requests.post")
    def test_returns_dataclass_fields(self, mock_post):
        mock_post.return_value = _successful_post(score=3, explanation="meh")

        result = evaluate_confidence(
            extra_context="ctx",
            trace_id="tid",
            endpoint="http://test:8000",
            api_key="k",
        )

        assert result.score == 3
        assert result.explanation == "meh"
