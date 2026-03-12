"""LLM factory — returns the configured chat model based on ``LLM_PROVIDER``.

Supported providers: ``nestle`` (primary) and ``openai`` (fallback).
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from agent_eval.config import get_nesgen_config, get_openai_config

load_dotenv()

logger = logging.getLogger(__name__)


def create_model(provider: Optional[str] = None) -> BaseChatModel:
    """Create and return a LangChain chat model for the given provider.

    Args:
        provider: "nestle" or "openai". Defaults to LLM_PROVIDER env var, then "nestle".

    Raises:
        ValueError: If required credentials are missing or provider is unknown.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "nestle").strip().lower()

    if provider == "nestle":
        return _create_nestle()
    elif provider == "openai":
        return _create_openai()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{provider}'. Supported: 'nestle', 'openai'")


def _create_openai() -> BaseChatModel:
    from agent_eval.llm.openai_chat_model import create_openai_model

    openai_config = get_openai_config()
    api_key = openai_config.api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai. Set it in your .env file.")

    model_name = openai_config.model
    temperature = openai_config.temperature
    max_tokens = openai_config.max_tokens
    base_url = openai_config.base_url or None

    logger.info("Using OpenAI provider (model=%s)", model_name)
    return create_openai_model(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def _create_nestle() -> BaseChatModel:
    from agent_eval.llm.nestle_chat_model_v2 import NesGenChatModel

    nesgen_config = get_nesgen_config()
    logger.info("Using NESGEN provider")
    model = NesGenChatModel(
        client_id=nesgen_config.client_id,
        client_secret=nesgen_config.client_secret,
        model_endpoint=nesgen_config.model_endpoint,
        api_base=nesgen_config.api_base,
        model=nesgen_config.model,
        api_version=nesgen_config.api_version,
    )
    if not model.client_id or not model.client_secret:
        raise ValueError("NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET are required. Set them in your .env file.")
    return model


def check_credentials(provider: Optional[str] = None) -> tuple[bool, str]:
    """Check if credentials are configured for the given provider.

    Returns (ok, message) where ok is True if credentials are present.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "nestle").strip().lower()

    if provider == "nestle":
        nesgen_config = get_nesgen_config()
        cid = nesgen_config.client_id
        csec = nesgen_config.client_secret
        if cid and csec:
            return True, "NESGEN credentials configured"
        return False, "Set NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET in .env"

    elif provider == "openai":
        openai_config = get_openai_config()
        key = openai_config.api_key
        if key:
            return True, "OpenAI API key configured"
        return False, "Set OPENAI_API_KEY in .env"

    return False, f"Unknown provider: {provider}"


def check_llm_connectivity(provider: Optional[str] = None) -> tuple[bool, str]:
    """Check if the LLM is accessible and responding.

    Returns (ok, message) where ok is True if connection successful.
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "nestle").strip().lower()

    creds_ok, creds_msg = check_credentials(provider)
    if not creds_ok:
        return False, f"Credentials check failed: {creds_msg}"

    logger.info("Credentials OK. Testing connectivity to %s...", provider)

    try:
        model = create_model(provider)
        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content="Hello! Please respond with 'OK' if you can read this.")])
        if response and response.content:
            logger.info("✓ Received response from %s", provider)
            return True, f"✓ Successfully connected to {provider}. Response: {response.content[:100]}"
        else:
            return False, f"✗ Received empty response from {provider}"
    except Exception as e:
        error_msg = f"✗ Connection failed to {provider}: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def main():
    """Test LLM connectivity. Run with: python -m agent_eval.llm.factory"""
    import sys

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("LLM Connectivity Check")
    print("=" * 60)

    provider = os.getenv("LLM_PROVIDER", "nestle").strip().lower()
    print(f"\nProvider: {provider}")
    print("-" * 60)

    print("\n1. Checking credentials...")
    creds_ok, creds_msg = check_credentials(provider)
    print(f"   {creds_msg}")

    if not creds_ok:
        print("\n❌ Credentials check failed. Please configure your .env file.")
        sys.exit(1)

    print("\n2. Testing LLM connectivity...")
    conn_ok, conn_msg = check_llm_connectivity(provider)
    print(f"   {conn_msg}")

    print("\n" + "=" * 60)
    if conn_ok:
        print("✅ LLM connectivity check PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ LLM connectivity check FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
