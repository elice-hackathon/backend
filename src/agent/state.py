"""Define the state structures for the agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass
class BurgerOption:
    """Defines the structure of a burger option."""

    id: int
    name: str
    price: float

    def dict(self):
        """Return the state as a dictionary."""
        return {k: str(v) for k, v in asdict(self).items()}

    def json(self):
        """Return the state as a JSON string."""
        return json.dump(self.dict())


@dataclass
class PurchaseBurgerItem:
    """Defines the structure of a purchase burger item."""

    id: int
    name: str
    price: float
    quantity: int
    options: list[BurgerOption] = field(default_factory=list)

    def dict(self):
        """Return the state as a dictionary."""
        return {k: str(v) for k, v in asdict(self).items()}

    def json(self):
        """Return the state as a JSON string."""
        return json.dump(self.dict())


@dataclass
class PurchaseInformation:
    """Defines the structure of a purchase information."""

    items: list[PurchaseBurgerItem] = field(default_factory=list)
    total_price: float = 0
    total_items: int = 0
    total_quantity: int = 0
    # payment_type: Literal["card", "cash"] = "card"

    def dict(self):
        """Return the state as a dictionary."""
        return {k: str(v) for k, v in asdict(self).items()}

    def json(self):
        """Return the state as a JSON string."""
        return json.dumps(self.dict())


@dataclass
class State:
    """Main Graph state."""

    messages: Annotated[list[AnyMessage], add_messages]
    purchase_information: Optional[PurchaseInformation] = field(default=None)

    def dict(self):
        """Return the state as a dictionary."""
        return {k: str(v) for k, v in asdict(self).items()}

    def json(self):
        """Return the state as a JSON string."""
        return json.dump(self.dict())
