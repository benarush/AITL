from typing import Any, Optional

from ..._common import to_jsonable
from ...utils import extract_reasoning


def _model_name(agent: Any) -> str:
    """Real model id (e.g. 'gemini-2.5-flash-lite'); class name as fallback."""
    try:
        config = agent.model.get_config() or {}
        name = config.get("model_id") or config.get("model")
    except Exception:
        name = None
    return str(name or type(agent.model).__name__)


def _thinking_config(agent: Any) -> Optional[dict[str, Any]]:
    """The agent model's reasoning/thinking settings as configured, or ``None``.

    Reads ``agent.model.get_config()`` (reasoning options sit in its ``params`` /
    ``additional_request_fields``); the provider is inferred from the model class
    (module + name) and the model id. Never raises; see :func:`extract_reasoning`.
    """
    try:
        model = agent.model
        config = model.get_config() or {}
        return extract_reasoning(
            [config],
            provider_hint=f"{type(model).__module__} {type(model).__name__}",
            model_name=_model_name(agent),
        )
    except Exception:
        return None


def _openai_tools(agent: Any) -> list[dict[str, Any]]:
    """Agent's tools in the OpenAI function shape the backend expects."""
    tools = []
    for spec in agent.tool_registry.get_all_tools_config().values():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": (spec.get("inputSchema") or {}).get("json", {}),
                },
            }
        )
    return to_jsonable(tools)
