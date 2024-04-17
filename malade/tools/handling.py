import langroid as lr
from typing import Callable, Optional, Any

def handle_with(
        agent: lr.Agent,
        tool: type[lr.ToolMessage],
        handler: Callable[[Any], Optional[str | lr.ChatDocument]],
) -> None:
    """Handles a `ToolMessage` with a specific handler."""
    tool_name = tool.default_value("request")
    setattr(agent, tool_name, handler)
