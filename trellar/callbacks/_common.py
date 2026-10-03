"""Framework-neutral helpers shared by the LangChain and Strands callbacks."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def compact_json(value: Any) -> str:
    """Render *value* as compact JSON, falling back to repr on failure."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def hash_tools(tools: list[Any]) -> str:
    """Stable content hash of a tool schema list.

    Used as the ``available_tools`` key and stamped on the matching
    ``on_chat_model_start`` event so the backend can link the two.
    """
    canonical = json.dumps(tools, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def to_jsonable(value: Any) -> Any:
    """Recursively coerce *value* into something ``json.dumps`` can handle."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def build_context(events: list[dict[str, Any]], trace_id: Optional[Any]) -> str:
    """Render recorded events as a numbered, step-by-step narrative."""
    lines: list[str] = [
        "=== Agent Run Context ===",
        f"Trace ID: {trace_id}",
        f"Total steps: {len(events)}",
        "",
    ]

    for event in events:
        step = event.get("graph_order", "?")
        event_name = event.get("event", "unknown")
        node_name = event.get("node_name") or ""
        node_type = event.get("node_type") or ""

        header = f"[Step {step}] {event_name}"
        if node_name:
            header += f"  ({node_type}: {node_name})"
        lines.append(header)

        # Per-event payload rendering
        if event_name in ("on_llm_start", "on_chat_model_start"):
            inp = event.get("input", {})
            if inp.get("system"):
                lines.append(f"  system: {inp['system']}")
            if inp.get("human"):
                lines.append(f"  human: {inp['human']}")

        elif event_name == "on_llm_end":
            out = event.get("output", {})
            usage = event.get("token_usage")
            lines.append(f"  response: {out.get('response', '')}")
            if usage:
                lines.append(f"  token_usage: {compact_json(usage)}")

        elif event_name == "on_tool_start":
            lines.append(f"  tool: {event.get('tool', '')}")
            lines.append(f"  input: {compact_json(event.get('input', {}))}")

        elif event_name == "on_tool_end":
            # MCP provenance is signal for the confidence evaluator, not noise.
            via_mcp = " (via MCP)" if event.get("is_mcp_tool") else ""
            lines.append(f"  output{via_mcp}: {compact_json(event.get('output', ''))}")

        elif event_name == "on_chain_start":
            lines.append(f"  inputs: {compact_json(event.get('inputs', {}))}")

        elif event_name == "on_chain_end":
            lines.append(f"  outputs: {compact_json(event.get('outputs', {}))}")

        elif event_name in ("on_llm_error", "on_tool_error", "on_chain_error"):
            lines.append(f"  error: {event.get('error', '')}")

        lines.append("")  # blank line between steps

    return "\n".join(lines)
