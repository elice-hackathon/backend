"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated

from agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    user_id: str = "default"
    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o-mini",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    system_prompt: str = prompts.SYSTEM_PROMPT

    emotion_system_prompt: str = prompts.EMOTION_RESPONSE_SYSTEM_PROMPT

    emotion_user_prompt: str = prompts.EMOTION_RESPONSE_USER_PROMPT

    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }

        return cls(**{k: v for k, v in values.items() if v})
