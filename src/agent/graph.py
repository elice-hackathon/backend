"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Any, Dict, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from typing_extensions import Annotated, TypedDict

from agent import utils
from agent.configuration import Configuration
from agent.state import PurchaseInformation, State

# from agent.tool_node import tool_node
from agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


llm = init_chat_model()


class EmotionalResponse(TypedDict):
    """Emotional response. It's an AI response that includes the response and the related emotion."""

    response: Annotated[str, ..., "The response of the chat"]
    emotion: Annotated[
        str,
        ...,
        "The emotion of the chat. It should be one of 'happy', 'sad', 'neutral', 'angry', 'surprised', 'disgusted', 'fearful', 'disappointed'",
    ]


async def agent(
    state: State, config: RunnableConfig, *, store: BaseStore
) -> Dict[str, Any]:
    """Extract the user's state from the conversation and update the memory."""
    configurable = Configuration.from_runnable_config(config)

    # Retrieve the most recent memories for context
    memories = await store.asearch(("memories", configurable.user_id), limit=10)

    # Format memories for inclusion in the prompt
    formatted = "\n".join(f"[{mem.key}]: {mem.value}" for mem in memories)
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.
    msg = await llm.bind_tools(ALL_TOOLS).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": utils.split_model_and_provider(configurable.model)},
    )

    return {"messages": [msg]}


def should_continue(state: State):
    """Schedule the next node after the agent's action.

    This function determines the next step in the research process based on the
    last message in the state. It handles three main scenarios:
    """

    def tool_router(tool_name: str) -> str:
        if tool_name == "purchase_burger_items":
            return "prepare_purchase_burger_items"
        elif tool_name == "add_burger_to_cart_tool":
            return "add_burger_to_cart"
        elif tool_name == "remove_burger_from_cart_tool":
            return "remove_burger_from_cart"
        else:
            return "tools"

    last_message = state.messages[-1]

    # "If for some reason the last message is not an AIMessage (due to a bug or unexpected behavior elsewhere in the code),
    # it ensures the system doesn't crash but instead tries to recover by calling the agent model again.
    if not isinstance(last_message, AIMessage):
        return "agent"

    # Normal cases where the last message is an AIMessage
    if last_message.type != "ai" or not last_message.tool_calls:
        return END

    if not last_message.tool_calls:
        raise ValueError("Expected tool_calls to be an array with at least one element")

    return [tool_router(tc["name"]) for tc in last_message.tool_calls]


def add_burger_to_cart(state: State, config: RunnableConfig):
    """Add a burger item to the cart for later purchase. Update the state with the new purchase information."""
    user_id = Configuration.from_runnable_config(config).user_id

    last_message = state.messages[-1]
    if last_message.type != "ai":
        raise ValueError("Expected the last message to be an AI message")

    add_burger_to_cart_tool = next(
        (
            tc
            for tc in last_message.tool_calls
            if tc["name"] == "add_burger_to_cart_tool"
        ),
        None,
    )
    if not add_burger_to_cart_tool:
        raise ValueError(
            "Expected the last AI message to have a add_burger_to_cart_tool tool call"
        )

    purchase_burger_item = add_burger_to_cart_tool["args"].get("purchase_burger_item")

    if state.purchase_information is None:
        state.purchase_information = PurchaseInformation()

    state.purchase_information.items.append(purchase_burger_item)
    state.purchase_information.total_items += 1
    state.purchase_information.total_quantity += purchase_burger_item["quantity"]
    state.purchase_information.total_price += (
        purchase_burger_item["price"] * purchase_burger_item["quantity"]
    )

    return {
        "messages": [
            {
                "tool_call_id": add_burger_to_cart_tool["id"],
                "role": "tool",
                "name": add_burger_to_cart_tool["name"],
                "content": f"Successfully added {purchase_burger_item['name']} to {user_id}'s cart.",
            }
        ],
        "purchase_information": state.purchase_information,
    }


def remove_burger_from_cart(state: State, config: RunnableConfig):
    """Remove a burger item from the cart. Update the state with the new purchase information."""
    user_id = Configuration.from_runnable_config(config).user_id

    last_message = state.messages[-1]
    if last_message.type != "ai":
        raise ValueError("Expected the last message to be an AI message")

    remove_burger_from_cart_tool = next(
        (
            tc
            for tc in last_message.tool_calls
            if tc["name"] == "remove_burger_from_cart_tool"
        ),
        None,
    )
    if not remove_burger_from_cart_tool:
        raise ValueError(
            "Expected the last AI message to have a remove_burger_from_cart_tool tool call"
        )

    purchase_burger_item = remove_burger_from_cart_tool["args"].get(
        "purchase_burger_item"
    )

    if state.purchase_information is None:
        state.purchase_information = PurchaseInformation()

    state.purchase_information.items.remove(purchase_burger_item)
    state.purchase_information.total_items -= 1
    state.purchase_information.total_quantity -= purchase_burger_item["quantity"]
    state.purchase_information.total_price -= (
        purchase_burger_item["price"] * purchase_burger_item["quantity"]
    )

    return {
        "messages": [
            {
                "tool_call_id": remove_burger_from_cart_tool["id"],
                "role": "tool",
                "name": remove_burger_from_cart_tool["name"],
                "content": f"Successfully removed {purchase_burger_item['name']} to {user_id}'s cart.",
            }
        ],
        "purchase_information": state.purchase_information,
    }


def prepare_purchase_burger_items(state: State, config: RunnableConfig):
    """Prepare the purchase of the selected burger items."""
    last_message = state.messages[-1]
    if last_message.type != "ai":
        raise ValueError("Expected the last message to be an AI message")

    purchase_burger_items_tool = next(
        (tc for tc in last_message.tool_calls if tc["name"] == "purchase_burger_items"),
        None,
    )
    if not purchase_burger_items_tool:
        raise ValueError(
            "Expected the last AI message to have a purchase_burger_items tool call"
        )

    purchase_information = purchase_burger_items_tool["args"].get(
        "purchase_information"
    )

    if not purchase_information:
        tool_messages = [
            {
                "role": "tool",
                "content": f"Please provide the missing information for the {tc['name']} tool.",
                "id": tc["id"],
            }
            for tc in last_message.tool_calls
        ]

        return {
            "messages": tool_messages
            + [
                {
                    "role": "assistant",
                    "content": "Please provide the purchasing burgers to further process your order.",
                }
            ],
        }

    return {"purchase_information": purchase_information}


async def purchase_approval(state: State, config: RunnableConfig, *, store: BaseStore):
    """Approve a purchase."""
    last_message = state.messages[-1]
    if not isinstance(last_message, ToolMessage):
        raise ValueError("Please confirm the purchase before executing.")


async def execute_purchase(state: State, config: RunnableConfig, *, store: BaseStore):
    """Execute a purchase for the burger order."""
    purchase_information = state.purchase_information
    if not purchase_information:
        raise ValueError("Expected purchase_information to be present")

    tool_call_id = f"tool_{random.random()}"

    return {
        "messages": [
            {
                "type": "ai",
                "tool_calls": [
                    {
                        "name": "execute_purchase",
                        "id": tool_call_id,
                        "args": {
                            "purchase_information": purchase_information,
                        },
                    }
                ],
            },
            {
                "type": "tool",
                "name": "execute_purchase",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"success": True}),
            },
            {
                "type": "ai",
                "content": f"Successfully purchased {purchase_information.total_quantity} items for a total of {purchase_information.total_price} KRW.",
            },
        ],
    }


async def should_execute_purchase(
    state: State, config: RunnableConfig, *, store: BaseStore
):
    """Determine whether to execute a purchase."""
    decision = state.messages[-1].content.lower()
    if decision == "yes" or decision == "y":
        return "execute_purchase"
    return "agent"


##### GRAPH #####

workflow = StateGraph(State, config_schema=Configuration)

# LLM Node
workflow.add_node("agent", agent)


# Tool Node
workflow.add_node("tools", ToolNode(ALL_TOOLS))
# workflow.add_node("tools", tool_node)

# Add Burger to Cart Node
workflow.add_node("add_burger_to_cart", add_burger_to_cart)

# Remove Burger from Cart Node
workflow.add_node("remove_burger_from_cart", remove_burger_from_cart)

# Prepare Purchase Burger Item Node
workflow.add_node("prepare_purchase_burger_items", prepare_purchase_burger_items)

# Approval Node
workflow.add_node("purchase_approval", purchase_approval)

# Execute Node
workflow.add_node("execute_purchase", execute_purchase)

# Start with LLM model
workflow.add_edge(START, "agent")

# Link the model with tools
workflow.add_edge("tools", "agent")

# Link the model with tools
workflow.add_edge("add_burger_to_cart", "agent")

# Link the model with tools
workflow.add_edge("remove_burger_from_cart", "agent")

# Finish the flow when the purchase is done
workflow.add_edge("execute_purchase", END)

# Link purchase flow
workflow.add_edge("prepare_purchase_burger_items", "purchase_approval")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    [
        "tools",
        END,
        "prepare_purchase_burger_items",
        "add_burger_to_cart",
        "remove_burger_from_cart",
    ],
)
workflow.add_conditional_edges(
    "purchase_approval", should_execute_purchase, ["execute_purchase", "agent"]
)


graph = workflow.compile()
graph.name = "Elice Agent"
