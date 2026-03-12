"""OpenAI chat model — thin wrapper around ``langchain-openai``'s ``ChatOpenAI``."""

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def create_openai_model(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: Optional[int] = 1024,
    base_url: Optional[str] = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance."""
    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if base_url:
        kwargs["base_url"] = base_url

    logger.info("Creating OpenAI model: %s", model)
    return ChatOpenAI(**kwargs)
