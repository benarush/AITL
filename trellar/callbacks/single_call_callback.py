from __future__ import annotations

import logging
import uuid
from typing import Any, Optional, TYPE_CHECKING

from .._context import _current_callback
from ..agent_loop import ObservabilityMode
from .langchain_callback import _AgentGuardCallback

if TYPE_CHECKING:
    from ..agent_loop import AgentLoopResult

logger = logging.getLogger(__name__)


class _SingleCallGuardCallback(_AgentGuardCallback):
    """Guard for a single bare LLM call (no LangGraph/chain wrapper).

    This class is not part of the public API. Use :func:`get_single_call_guard`
    to obtain an instance.

    ``_AgentGuardCallback`` only sets ``trace_id``/registers ``_current_callback``
    inside ``on_chain_start``, and only auto-triggers ``evaluate_confidence()``
    inside ``on_chain_end`` -- both scoped to ``parent_run_id is None``. A bare
    ``llm.invoke(...)`` never fires either of those events (no chain involved),
    so this subclass does the equivalent work on the LLM run boundary instead:
    ``on_llm_start``/``on_chat_model_start`` (root-run reset + registration) and
    ``on_llm_end`` (auto-evaluate trigger).

    Deliberately does **not** modify ``_AgentGuardCallback`` in any way -- it
    only calls the parent's existing public methods via ``super()`` to reuse
    message serialization / event recording / tool-call folding. The small
    reset and auto-evaluate blocks are re-implemented locally here rather than
    extracted onto the base class, so the graph path stays fully untouched.
    """

    is_single_call = True

    def __init__(
        self,
        *,
        agent_name: str,
        observability_mode: ObservabilityMode = ObservabilityMode.NONE,
    ) -> None:
        super().__init__(agent_name=agent_name, observability_mode=observability_mode)
        # Populated by _auto_evaluate() when observability_mode triggers an
        # automatic evaluate_confidence() call. None until then, or if the
        # caller only ever evaluates manually (observability_mode=NONE).
        self.trellar_evaluate_result: Optional["AgentLoopResult"] = None
        self.trellar_evaluate_error: Optional[BaseException] = None

    def _reset_for_new_run(self, run_id: uuid.UUID) -> None:
        """Local equivalent of on_chain_start's root-reset block.

        Duplicated here rather than extracted onto the base class so
        _AgentGuardCallback (the graph path) is left completely untouched.
        """
        self.trace_id = run_id
        self.events = []
        self._step = 0
        self._run_registry = {}
        self._pending_llm_tool_calls = []
        self._evaluated = False
        self.trellar_evaluate_result = None
        self.trellar_evaluate_error = None
        # Self-register so evaluate_confidence() can pick us up automatically.
        _current_callback.set(self)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._reset_for_new_run(run_id)
        super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._reset_for_new_run(run_id)
        super().on_chat_model_start(serialized, messages, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
        if parent_run_id is None:
            self._auto_evaluate()

    def _auto_evaluate(self) -> None:
        """Local equivalent of _maybe_auto_evaluate; stores the outcome on
        this instance instead of discarding it. Not added to the base class.

        Errors are caught and logged, never raised -- an auto-triggered
        observability call can never crash the caller's llm.invoke().
        """
        if self.observability_mode is ObservabilityMode.NONE:
            return
        if self.observability_mode is ObservabilityMode.IF_NOT_EVALUATED and self._evaluated:
            return
        from ..agent_loop import evaluate_confidence

        try:
            self.trellar_evaluate_result = evaluate_confidence(_observability_call=True)
            self.trellar_evaluate_error = None
        except Exception as exc:
            self.trellar_evaluate_error = exc
            logger.warning("Auto-triggered evaluate_confidence() failed", exc_info=True)
