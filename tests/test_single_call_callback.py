from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from trellar import ObservabilityMode
from trellar._context import _current_callback
from trellar.agent_loop import AgentLoopResult
from trellar.callbacks.langchain_callback import _AgentGuardCallback
from trellar.callbacks.single_call_callback import _SingleCallGuardCallback

from tests.factories import make_ai_message, make_llm_result


@pytest.fixture
def single_call_handler():
    """A fresh _SingleCallGuardCallback, not yet registered in _current_callback.

    Mirrors conftest.py's `active_handler` fixture but for the single-call
    subclass, and without pre-setting trace_id -- these tests exercise the
    root-run detection itself.
    """
    token = _current_callback.set(None)
    handler = _SingleCallGuardCallback(agent_name="single-test")
    yield handler
    _current_callback.reset(token)


def _fake_result(**overrides) -> AgentLoopResult:
    defaults = dict(explanation="looks good", score=8, decision_identifier="decision-1", should_stop_network=False)
    defaults.update(overrides)
    return AgentLoopResult(**defaults)


# ---------------------------------------------------------------------------
# Root-run detection -- on_llm_start / on_chat_model_start
# ---------------------------------------------------------------------------

class TestRootRunDetection:
    def test_on_chat_model_start_sets_trace_id_and_registers_callback(self, single_call_handler):
        run_id = uuid.uuid4()
        assert _current_callback.get() is None

        single_call_handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash-lite"}}, [[]], run_id=run_id, parent_run_id=None
        )

        assert single_call_handler.trace_id == run_id
        assert _current_callback.get() is single_call_handler

    def test_on_llm_start_sets_trace_id_and_registers_callback(self, single_call_handler):
        run_id = uuid.uuid4()

        single_call_handler.on_llm_start({}, ["prompt"], run_id=run_id, parent_run_id=None)

        assert single_call_handler.trace_id == run_id
        assert _current_callback.get() is single_call_handler

    def test_non_root_start_does_not_touch_trace_id(self, single_call_handler):
        # A child run (parent_run_id set) should never happen for a bare
        # llm.invoke(), but if it did, it must not be treated as root.
        run_id = uuid.uuid4()
        parent_id = uuid.uuid4()

        single_call_handler.on_chat_model_start(
            {"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=parent_id
        )

        assert single_call_handler.trace_id is None
        assert _current_callback.get() is None

    def test_second_root_call_resets_state_from_first(self, single_call_handler):
        run_id_1 = uuid.uuid4()
        run_id_2 = uuid.uuid4()

        single_call_handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id_1, parent_run_id=None)
        single_call_handler.on_llm_end(make_llm_result(message=make_ai_message("first")), run_id=run_id_1, parent_run_id=None)
        first_call_events = len(single_call_handler.events)
        assert first_call_events > 0

        single_call_handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id_2, parent_run_id=None)

        assert single_call_handler.trace_id == run_id_2
        # Reset wipes the first call's events -- only the new root's data remains.
        assert len(single_call_handler.events) == 1
        assert single_call_handler.trellar_evaluate_result is None
        assert single_call_handler.trellar_evaluate_error is None


# ---------------------------------------------------------------------------
# Recording -- inherited behavior from _AgentGuardCallback via super()
# ---------------------------------------------------------------------------

class TestRecordingIsInherited:
    def test_on_llm_end_records_response_text(self, single_call_handler):
        run_id = uuid.uuid4()
        single_call_handler.on_chat_model_start(
            {"kwargs": {"model": "gemini-2.5-flash-lite"}},
            [[]],
            run_id=run_id,
            parent_run_id=None,
        )
        single_call_handler.on_llm_end(
            make_llm_result(message=make_ai_message("the office opens at 9am")),
            run_id=run_id,
            parent_run_id=None,
        )

        end_event = single_call_handler.events[-1]
        assert end_event["event"] == "on_llm_end"
        assert end_event["output"]["response"] == "the office opens at 9am"

    def test_is_single_call_true_on_subclass_absent_on_base(self, single_call_handler):
        # _AgentGuardCallback is untouched -- it has no is_single_call
        # attribute at all. evaluate_confidence()'s payload construction
        # uses getattr(callback, "is_single_call", False), so the base
        # class correctly defaults to False without ever declaring it.
        base = _AgentGuardCallback(agent_name="base-test")
        assert single_call_handler.is_single_call is True
        assert not hasattr(base, "is_single_call")
        assert getattr(base, "is_single_call", False) is False


# ---------------------------------------------------------------------------
# Auto-evaluate trigger -- on_llm_end (local _auto_evaluate, not the base
# class's _maybe_auto_evaluate)
# ---------------------------------------------------------------------------

class TestAutoEvaluateTrigger:
    def _run_one_call(self, handler):
        run_id = uuid.uuid4()
        handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)
        handler.on_llm_end(make_llm_result(message=make_ai_message("hi")), run_id=run_id, parent_run_id=None)

    def test_none_mode_never_calls_evaluate_confidence(self, single_call_handler):
        with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
            self._run_one_call(single_call_handler)
        mock_eval.assert_not_called()
        assert single_call_handler.trellar_evaluate_result is None

    def test_always_mode_calls_evaluate_confidence_and_stores_trellar_evaluate_result(self):
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
            with patch("trellar.agent_loop.evaluate_confidence", return_value=_fake_result()) as mock_eval:
                self._run_one_call(handler)
            mock_eval.assert_called_once_with(_observability_call=True)
            assert handler.trellar_evaluate_result == _fake_result()
            assert handler.trellar_evaluate_error is None
        finally:
            _current_callback.reset(token)

    def test_always_mode_calls_even_if_already_evaluated(self):
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
            run_id = uuid.uuid4()
            handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)
            handler._evaluated = True
            with patch("trellar.agent_loop.evaluate_confidence", return_value=_fake_result()) as mock_eval:
                handler.on_llm_end(make_llm_result(message=make_ai_message("hi")), run_id=run_id, parent_run_id=None)
            mock_eval.assert_called_once()
        finally:
            _current_callback.reset(token)

    def test_if_not_evaluated_mode_calls_when_not_yet_evaluated(self):
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.IF_NOT_EVALUATED)
            with patch("trellar.agent_loop.evaluate_confidence", return_value=_fake_result()) as mock_eval:
                self._run_one_call(handler)
            mock_eval.assert_called_once()
            assert handler.trellar_evaluate_result == _fake_result()
        finally:
            _current_callback.reset(token)

    def test_if_not_evaluated_mode_skips_when_already_evaluated(self):
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.IF_NOT_EVALUATED)
            run_id = uuid.uuid4()
            handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)
            handler._evaluated = True
            with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
                handler.on_llm_end(make_llm_result(message=make_ai_message("hi")), run_id=run_id, parent_run_id=None)
            mock_eval.assert_not_called()
            assert handler.trellar_evaluate_result is None
        finally:
            _current_callback.reset(token)

    def test_auto_evaluate_failure_is_caught_and_stored_as_trellar_evaluate_error(self):
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
            boom = ValueError("no api key")
            with patch("trellar.agent_loop.evaluate_confidence", side_effect=boom):
                self._run_one_call(handler)  # must not raise
            assert handler.trellar_evaluate_result is None
            assert handler.trellar_evaluate_error is boom
        finally:
            _current_callback.reset(token)


# ---------------------------------------------------------------------------
# on_llm_end / on_llm_error — _current_callback release (root run only)
#
# Local equivalent of _AgentGuardCallback's on_chain_end/on_chain_error
# release: on_llm_start/on_chat_model_start register the handler, and
# whichever of on_llm_end / on_llm_error fires next (mutually exclusive per
# run_id) must release it.
# ---------------------------------------------------------------------------

class TestCurrentCallbackReleaseOnLlmEnd:
    def test_root_llm_end_clears_current_callback(self, single_call_handler):
        run_id = uuid.uuid4()
        single_call_handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)
        assert _current_callback.get() is single_call_handler

        single_call_handler.on_llm_end(make_llm_result(message=make_ai_message("hi")), run_id=run_id, parent_run_id=None)

        assert _current_callback.get() is None

    def test_root_llm_end_does_not_clobber_a_different_active_handler(self):
        # Defends the `is self` identity guard: if something else has since
        # become the active handler, this handler's own root on_llm_end must
        # not blindly clear/overwrite that registration.
        outer_token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a")
            run_id = uuid.uuid4()
            handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)

            other = _SingleCallGuardCallback(agent_name="b")
            token = _current_callback.set(other)
            try:
                handler.on_llm_end(make_llm_result(message=make_ai_message("hi")), run_id=run_id, parent_run_id=None)
                assert _current_callback.get() is other
            finally:
                _current_callback.reset(token)
        finally:
            _current_callback.reset(outer_token)


class TestCurrentCallbackReleaseOnLlmError:
    def test_root_llm_error_clears_current_callback(self, single_call_handler):
        run_id = uuid.uuid4()
        single_call_handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)

        single_call_handler.on_llm_error(RuntimeError("boom"), run_id=run_id, parent_run_id=None)

        assert _current_callback.get() is None

    def test_non_root_llm_error_does_not_clear_current_callback(self, single_call_handler):
        run_id = uuid.uuid4()
        single_call_handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)

        single_call_handler.on_llm_error(RuntimeError("boom"), run_id=uuid.uuid4(), parent_run_id=run_id)

        assert _current_callback.get() is single_call_handler

    def test_on_llm_end_never_fires_for_a_call_that_errored(self):
        # Regression guard mirroring the graph guard's equivalent: on_llm_end
        # and on_llm_error are mutually exclusive per run_id, so a crash must
        # go through the error-path release, not rely on a never-to-arrive
        # on_llm_end / _auto_evaluate() call.
        token = _current_callback.set(None)
        try:
            handler = _SingleCallGuardCallback(agent_name="a", observability_mode=ObservabilityMode.ALWAYS)
            run_id = uuid.uuid4()
            handler.on_chat_model_start({"kwargs": {"model": "m"}}, [[]], run_id=run_id, parent_run_id=None)

            with patch("trellar.agent_loop.evaluate_confidence") as mock_eval:
                handler.on_llm_error(RuntimeError("boom"), run_id=run_id, parent_run_id=None)

            mock_eval.assert_not_called()
            assert handler.trellar_evaluate_result is None
            assert _current_callback.get() is None
        finally:
            _current_callback.reset(token)
