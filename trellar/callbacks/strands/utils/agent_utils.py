from typing import Any

from ..._common import to_jsonable


def _model_name(agent: Any) -> str:
    """Real model id (e.g. 'gemini-2.5-flash-lite'); class name as fallback."""
    try:
        config = agent.model.get_config() or {}
        name = config.get("model_id") or config.get("model")
    except Exception:
        name = None
    return str(name or type(agent.model).__name__)


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
