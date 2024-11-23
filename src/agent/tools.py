"""Define he agent's tools."""

import uuid
from typing import Annotated, Any, Optional, cast

import pandas as pd
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.prebuilt import InjectedState
from langgraph.store.base import BaseStore

from agent import utils
from agent.configuration import Configuration
from agent.constants import burger_menus, emotions

# from agent.graph import EmotionalResponse
from agent.state import PurchaseBurgerItem, PurchaseInformation, State


@tool
async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


@tool
async def search_burger_info_by_id(
    burger_id: int,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Search for burger information based on the burger ID."""
    # TODO: Implement the search logic here.
    user_id = Configuration.from_runnable_config(config).user_id
    return f"Searching for burger info for {burger_id} for {user_id}."


@tool
async def suggest_burgers(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Suggestions for burgers based on user preferences.

    This tool will be called when the user wants to see suggestions.
    """
    # TODO: Implement the search logic here.
    # user_id = Configuration.from_runnable_config(config).user_id

    return {"burgerItems": burger_menus}


# @tool
# async def recommend_by_csv(
#     user_prefrences: dict,
#     *,
#     config: Annotated[RunnableConfig, InjectedToolArg],
#     store: Annotated[BaseStore, InjectedToolArg],
# ):
#     """Recommend a list of burgers from a curated list based on user preferences."""
#     csv_path = "/Users/jackahn/codes/langgraph-studio/new-langgraph-project/src/fixtures/burger_sales_data.csv"
#     user_id = Configuration.from_runnable_config(config).user_id
#     df = pd.read_csv(csv_path)

#     return f"Recommendations for {user_id} based on {csv_path}."


@tool
async def purchase_burger_items(
    purchase_information: PurchaseInformation,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Purchase the selected burger items.

    This tool will be called when the user confirms the purchase of the burger.
    """
    user_id = Configuration.from_runnable_config(config).user_id
    return f"Confirmed purchase of {purchase_information.total_quantity} for {user_id}."


@tool
async def get_current_purchase_information(
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Get the current purchase information for the user.

    This tool will be called when the user wants to view the cart.
    """
    return state.purchase_information


@tool
def add_burger_to_cart_tool(
    purchase_burger_item: PurchaseBurgerItem,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Add a burger item to the cart for later purchase.

    This tool will be called when the user wants to add a burger to the cart.
    """
    if state.purchase_information is None:
        state.purchase_information = PurchaseInformation()

    state.purchase_information.items.append(purchase_burger_item)
    state.purchase_information.total_items += 1
    state.purchase_information.total_quantity += purchase_burger_item.quantity
    state.purchase_information.total_price += (
        purchase_burger_item.price * purchase_burger_item.quantity
    )

    return state.purchase_information.json()


@tool
def remove_burger_from_cart_tool(
    purchase_burger_item: PurchaseBurgerItem,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg],
    state: Annotated[State, InjectedState],
):
    """Remove a burger item from the cart.

    This tool will be called when the user wants to remove a burger from the cart.
    """
    if state.purchase_information is None:
        state.purchase_information = PurchaseInformation()

    state.purchase_information.items.remove(purchase_burger_item)
    state.purchase_information.total_items -= 1
    state.purchase_information.total_quantity -= purchase_burger_item.quantity
    state.purchase_information.total_price -= (
        purchase_burger_item.price * purchase_burger_item.quantity
    )

    return state.purchase_information.json()


# @tool
# async def emotion_agent(
#     text: str,
#     *,
#     config: Annotated[RunnableConfig, InjectedToolArg],
#     state: Annotated[State, InjectedState],
# ):
#     """Return an emotional response based on the ai's response."""
#     configurable = Configuration.from_runnable_config(config)
#     sys = configurable.emotion_system_prompt

#     user_prompt = configurable.emotion_user_prompt.format(
#         emotions=", ".join(emotions), text=text
#     )

#     msg = await llm.with_structured_output(EmotionalResponse).ainvoke(
#         [{"role": "system", "content": sys}, {"role": "user", "content": user_prompt}],
#         {"configurable": utils.split_model_and_provider(configurable.model)},
#     )

#     return msg


ALL_TOOLS = [
    # search,
    purchase_burger_items,
    suggest_burgers,
    get_current_purchase_information,
    add_burger_to_cart_tool,
    remove_burger_from_cart_tool,
    # emotion_agent,
]
