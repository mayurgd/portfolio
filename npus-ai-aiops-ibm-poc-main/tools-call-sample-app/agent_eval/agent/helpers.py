"""Helper functions for graph_v2.py — intent classification, query decomposition,
message processing, and tracing support.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from agent_eval.agent.prompts import INTENT_ROUTER_PROMPT, DECOMPOSITION_PROMPT

logger = logging.getLogger(__name__)


def build_tool_descriptions(tools: List[Any]) -> str:
    """Build formatted tool descriptions for LLM context."""
    descriptions = []
    for tool in tools:
        name = getattr(tool, "name", "unknown")
        desc = getattr(tool, "description", "No description")
        descriptions.append(f"- {name}: {desc}")
    return "\n".join(descriptions)


def classify_intent(model: BaseChatModel, query: str, tool_descriptions: str) -> str:
    """Classify user intent as GENERIC or SPECIALIZED using the LLM.

    Returns "GENERIC" or "SPECIALIZED". Defaults to SPECIALIZED on failure.
    """
    try:
        prompt = INTENT_ROUTER_PROMPT.format(tool_descriptions=tool_descriptions)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ]
        response = model.invoke(messages)
        content = extract_response_content(response).strip().upper()
        intent = "SPECIALIZED" if "SPECIALIZED" in content else "GENERIC"
        logger.info("Intent: %s -> %s (LLM)", query[:50], intent)
        return intent
    except Exception as e:
        logger.warning("Intent classification failed, defaulting to SPECIALIZED: %s", e)
        return "SPECIALIZED"


def analyze_complexity(model: BaseChatModel, query: str, tool_descriptions: str) -> tuple[str, List[str]]:
    """Analyze query complexity and determine required tools using the LLM.

    Returns a tuple of (complexity, tool_names) where complexity is
    "SINGLE_TOOL" or "MULTI_TOOL". Adding a new tool to the MCP server
    is sufficient for it to be recognised here automatically.
    """
    raw_content = ""
    try:
        prompt = DECOMPOSITION_PROMPT.format(tool_descriptions=tool_descriptions)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ]
        response = model.invoke(messages)
        raw_content = extract_response_content(response).strip()

        # Strip markdown code fences if present
        content = raw_content
        if content.startswith("```"):
            parts = content.split("```")
            content = parts[1] if len(parts) > 1 else content
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        complexity = str(parsed.get("complexity", "SINGLE_TOOL")).strip().upper()
        if complexity not in ("SINGLE_TOOL", "MULTI_TOOL"):
            complexity = "SINGLE_TOOL"
        tool_names: List[str] = [str(t).strip().lower() for t in parsed.get("tools", [])]

        logger.info("Complexity: %s -> %s (LLM, tools: %s)", query[:50], complexity, tool_names)
        return complexity, tool_names

    except (json.JSONDecodeError, KeyError, TypeError) as parse_err:
        # LLM returned non-JSON — salvage from raw text
        logger.warning("JSON parse failed (%s), falling back to text scan of: %s", parse_err, raw_content[:100])
        upper = raw_content.upper()
        complexity = "MULTI_TOOL" if "MULTI_TOOL" in upper else "SINGLE_TOOL"
        known_tools = [
            line.split(":")[0].strip().lstrip("- ").lower()
            for line in tool_descriptions.splitlines()
            if line.strip().startswith("-")
        ]
        raw_lower = raw_content.lower()
        tool_names = [t for t in known_tools if t and t in raw_lower]
        logger.info("Complexity (text fallback): %s -> %s, tools: %s", query[:50], complexity, tool_names)
        return complexity, tool_names

    except Exception as e:
        logger.warning("Complexity analysis failed, defaulting to SINGLE_TOOL: %s", e)
        return "SINGLE_TOOL", []


def extract_user_message(messages: list) -> Optional[str]:
    """Extract the last user message from conversation history."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    str(item) if isinstance(item, str) else str(item.get("text", ""))
                    for item in content
                )
            else:
                return str(content)
    return None


def messages_to_dicts(messages: list) -> List[Dict[str, Any]]:
    """Convert LangChain messages to dicts for logging."""
    return [
        {"role": getattr(m, "type", "unknown"), "content": str(m.content)}
        for m in messages
    ]


def extract_token_usage(response: Any) -> Dict[str, int]:
    """Extract token usage from model response metadata.

    Handles both OpenAI and NESGEN response formats.
    """
    usage = {}

    # NESGEN raw response in additional_kwargs
    if hasattr(response, "additional_kwargs"):
        raw_response = response.additional_kwargs.get("raw_response", {})
        if raw_response and "usage" in raw_response:
            nesgen_usage = raw_response["usage"]
            if nesgen_usage.get("prompt_tokens") is not None:
                usage["input"] = nesgen_usage["prompt_tokens"]
            if nesgen_usage.get("completion_tokens") is not None:
                usage["output"] = nesgen_usage["completion_tokens"]
            if nesgen_usage.get("total_tokens") is not None:
                usage["total"] = nesgen_usage["total_tokens"]
            return usage

    # Standard OpenAI format
    if hasattr(response, "response_metadata"):
        meta = response.response_metadata or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if token_usage:
            if token_usage.get("prompt_tokens") is not None:
                usage["input"] = token_usage["prompt_tokens"]
            if token_usage.get("completion_tokens") is not None:
                usage["output"] = token_usage["completion_tokens"]
            if token_usage.get("total_tokens") is not None:
                usage["total"] = token_usage["total_tokens"]
    return usage


def extract_response_content(response: Any) -> str:
    """Extract text content from a model response.

    Handles both OpenAI (response.content) and NESGEN
    (raw_response.output[0].content[0].text) formats.
    Returns empty string for tool-call-only messages.
    """
    import json
    import logging

    logger = logging.getLogger(__name__)
    content = response.content

    # Tool-call messages have no text content to extract
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    if has_tool_calls:
        return str(content) if content else ""

    # NESGEN responses endpoint format: output[0].content[0].text
    if hasattr(response, "additional_kwargs"):
        raw_response = response.additional_kwargs.get("raw_response", {})
        if raw_response and raw_response.get("object") == "response":
            if "output" in raw_response:
                output_data = raw_response["output"]
                if isinstance(output_data, list) and len(output_data) > 0:
                    first_output = output_data[0]
                    if isinstance(first_output, dict) and "content" in first_output:
                        content_list = first_output["content"]
                        if isinstance(content_list, list) and len(content_list) > 0:
                            first_content = content_list[0]
                            if isinstance(first_content, dict) and "text" in first_content:
                                text_content = first_content["text"]
                                logger.debug(f"Extracted from NESGEN output[0].content[0].text: {text_content[:100]}")
                                return str(text_content) if text_content else ""

    return str(content) if content else ""


def prepare_generation_output(response: Any) -> Dict[str, Any]:
    """Prepare generation output dict from a model response."""
    content = extract_response_content(response)
    output: Dict[str, Any] = {"content": content}

    if hasattr(response, "tool_calls") and response.tool_calls:
        output["tool_calls"] = [
            {"name": tc["name"], "args": tc.get("args", {})}
            for tc in response.tool_calls
        ]

    return output


def get_langfuse_span(config: RunnableConfig) -> Any:
    """Extract Langfuse span from RunnableConfig."""
    return (config.get("configurable") or {}).get("langfuse_span")


def extract_tool_result_text(result: Any) -> str:
    """Extract plain-text JSON from an MCP tool result.

    MCP tools return a list of TextContent items. This unwraps the first item
    so the LLM receives clean, parseable JSON rather than a Python repr string.
    """
    if isinstance(result, list) and len(result) > 0:
        content_item = result[0]
        if hasattr(content_item, "text"):
            return content_item.text
        if isinstance(content_item, dict) and "text" in content_item:
            return content_item["text"]
        return str(content_item)

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        import json
        return json.dumps(result)

    return str(result)


