"""Tool node that executes the tools and writes all returned keys and values to the state variable."""

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import _get_state_args

from agent.state import State
from agent.tools import ALL_TOOLS

tools_by_name: dict[str, BaseTool] = {_tool.name: _tool for _tool in ALL_TOOLS}


def tool_node(state: State, config: RunnableConfig) -> dict[str, Any]:
    """Initialise a tool node that executes the tools and writes all returned keys and values to the state variable.

    TODO: This should be a modular class.
    """
    messages = []
    out = {}
    for tool_call in state.messages[-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        state_args = {var: state for var in _get_state_args(tool)}
        observation = tool.invoke({**tool_call["args"], **state_args}, config=config)

        message_content = observation.get("output")
        if observation.get("error") is not None:
            message_content = observation.get("error")

        messages.append(
            ToolMessage(content=message_content, tool_call_id=tool_call["id"])
        )
        out = {**out, **observation}

    return {"messages": messages, **out}
