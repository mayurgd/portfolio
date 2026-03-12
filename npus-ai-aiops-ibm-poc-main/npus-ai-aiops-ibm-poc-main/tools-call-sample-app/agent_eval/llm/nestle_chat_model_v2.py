"""NESGEN Chat Model with tool calling support.

Provides a LangChain-compatible NesGenChatModel and standalone connectivity tests.
"""

import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from agent_eval.config import get_nesgen_config
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

load_dotenv()


class NesGenChatModel(BaseChatModel):
    """LangChain-compatible NESGEN Chat Model with LLM-driven tool call inference."""

    client_id: str = Field(default="")
    client_secret: str = Field(default="")
    model_endpoint: str = Field(default="")
    api_base: str = Field(default="")
    model: str = Field(default="")
    api_version: str = Field(default="")
    temperature: float = Field(default=0.0)
    max_retries: int = Field(default=3)
    timeout: int = Field(default=180)

    bound_tools: List[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "nesgen"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "NesGenChatModel":  # type: ignore[override]
        """Return a new instance with tools bound."""
        new_instance = self.__class__(
            client_id=self.client_id,
            client_secret=self.client_secret,
            model_endpoint=self.model_endpoint,
            api_base=self.api_base,
            model=self.model,
            api_version=self.api_version,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
            bound_tools=tools,  # Pass as constructor argument
        )
        return new_instance

    def _make_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make a request to NESGEN API.

        Uses NESGEN_URL (responses endpoint) if set, otherwise falls back to
        the chat completions endpoint from NESGEN_API_BASE + model.
        """
        if self.model_endpoint:
            url = self.model_endpoint
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            lower_url = url.lower()
            if "/completions" in lower_url:
                payload: Dict[str, Any] = {
                    "messages": messages,
                    "temperature": self.temperature,
                }
            else:
                payload = {
                    "model": self.model,
                    "input": messages,
                }
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        else:
            url = f"{self.api_base.rstrip('/')}/openai/deployments/{self.model}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
            payload = {
                "messages": messages,
                "temperature": self.temperature,
            }
            response = requests.post(
                url,
                params={"api-version": self.api_version},
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

        response.raise_for_status()

        return response.json()

    def _extract_response_text(self, response_json: Dict[str, Any]) -> str:
        """Extract text from NESGEN API response (chat completions or responses format)."""
        logger = logging.getLogger(__name__)

        # Primary: standard OpenAI chat completions format
        try:
            choices = response_json.get("choices", [])
            if isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if content:
                            return str(content)
        except Exception as e:
            logger.error(f"Primary extraction (choices) failed: {e}")

        # Fallback: NESGEN responses-style format
        try:
            output = response_json.get("output", [])
            if isinstance(output, list) and len(output) > 0:
                first_output = output[0]
                if isinstance(first_output, dict):
                    content = first_output.get("content", [])
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            text = first_content.get("text", "")
                            if text:
                                return str(text)
                    if "text" in first_output:
                        return str(first_output["text"])
        except Exception as e:
            logger.error(f"Fallback extraction (output) failed: {e}")

        if "text" in response_json:
            return str(response_json["text"])

        if "message" in response_json:
            msg = response_json["message"]
            if isinstance(msg, str):
                return msg
            elif isinstance(msg, dict) and "content" in msg:
                return str(msg["content"])

        logger.error("All extraction methods failed!")
        logger.error(f"Response keys: {list(response_json.keys())}")
        logger.error(
            f"Full response (first 2000 chars): {json.dumps(response_json, indent=2)[:2000]}"
        )

        return ""

    def _build_tool_schema_prompt(self) -> str:
        """Build a JSON schema description of bound tools for the LLM tool-selection prompt."""
        if not self.bound_tools:
            return "[]"

        tool_defs = []
        for tool in self.bound_tools:
            name = getattr(tool, "name", "unknown")
            description = getattr(tool, "description", "")

            params: Dict[str, Any] = {}
            args_schema = getattr(tool, "args_schema", None)
            if args_schema is not None:
                try:
                    raw_schema = (
                        args_schema.schema()
                        if callable(getattr(args_schema, "schema", None))
                        else {}
                    )
                    params = raw_schema.get("properties", {})
                except Exception:
                    pass

            tool_defs.append(
                {
                    "name": name,
                    "description": description,
                    "parameters": params,
                }
            )

        return json.dumps(tool_defs, indent=2)

    def _llm_select_tool(self, user_query: str) -> List[Dict[str, Any]]:
        """Ask the LLM to select tools and extract arguments from the user query.

        Returns a list of tool-call dicts with keys ``name``, ``args``, ``id``.
        """
        logger = logging.getLogger(__name__)

        tool_schema = self._build_tool_schema_prompt()

        selection_prompt = textwrap.dedent(f"""
        You are a tool-selection assistant.

        Given the user query and the list of available tools below, decide which tool(s)
        to call and extract the required arguments directly from the query.

        Available tools (JSON schema):
        {tool_schema}

        Rules:
        1. Return ONLY a valid JSON array of tool calls — no prose, no markdown fences.
        2. Each element must have: "name" (tool name), "args" (object with parameter values).
        3. Only include tools that are genuinely needed.
        4. If no tool is needed, return an empty array: []
        5. Extract argument values exactly as they appear in the query.

        User query: {user_query}

        JSON array of tool calls:""").strip()

        try:
            nesgen_messages = [{"role": "user", "content": selection_prompt}]
            response_json = self._make_request(nesgen_messages)
            raw_text = self._extract_response_text(response_json)

            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```[a-zA-Z]*\n?", "", raw_text)
                raw_text = re.sub(r"\n?```$", "", raw_text)
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                logger.warning(
                    "LLM tool selection returned a single object instead of an array — wrapping: %s",
                    raw_text[:200],
                )
                parsed = [parsed]
            if not isinstance(parsed, list):
                logger.warning(
                    "LLM tool selection returned non-list: %s", raw_text[:200]
                )
                return []

            tool_calls = []
            for i, item in enumerate(parsed):
                if not isinstance(item, dict) or "name" not in item:
                    continue
                tool_calls.append(
                    {
                        "id": f"call_{item['name']}_{i}",
                        "name": item["name"].strip().lower(),
                        "args": item.get("args", {}),
                    }
                )

            logger.info(
                "LLM tool selection → %d call(s): %s",
                len(tool_calls),
                [tc["name"] for tc in tool_calls],
            )
            return tool_calls

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "LLM tool selection failed (%s), falling back to empty.", exc
            )
            return []

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat completion via the NESGEN API."""
        last_user_message = ""
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if hasattr(msg, "type") and msg.type == "human":
                last_user_message = msg.content
                last_user_index = i
                break

        has_tool_results = False
        tool_result_count = 0
        if last_user_index >= 0:
            for i in range(last_user_index + 1, len(messages)):
                if hasattr(messages[i], "type") and messages[i].type == "tool":
                    has_tool_results = True
                    tool_result_count += 1

        nesgen_messages = []

        if has_tool_results and last_user_index >= 0:
            # Tool results present: send system + user query + tool results for synthesis
            base_system = ""
            if (
                messages
                and hasattr(messages[0], "type")
                and messages[0].type == "system"
            ):
                base_system = str(messages[0].content)

            synthesis_instruction = (
                "\n\nIMPORTANT: You have just received tool results. "
                "Summarise them in clear, friendly, natural language for the user. "
                "Do NOT output raw JSON or tool arguments. "
                "Present the information in a readable format (e.g. bullet points or a short paragraph)."
            )
            nesgen_messages.append(
                {"role": "system", "content": base_system + synthesis_instruction}
            )
            nesgen_messages.append({"role": "user", "content": last_user_message})

            for i in range(last_user_index + 1, len(messages)):
                msg = messages[i]
                if hasattr(msg, "type"):
                    role = msg.type

                    if role == "tool":
                        tool_name = getattr(msg, "name", "unknown_tool")
                        nesgen_messages.append(
                            {
                                "role": "user",
                                "content": f"Tool '{tool_name}' returned: {msg.content}",
                            }
                        )
                    elif role == "ai" and msg.content:
                        nesgen_messages.append(
                            {"role": "assistant", "content": msg.content}
                        )
        else:
            # No tool results: send system + current user message only
            if (
                messages
                and hasattr(messages[0], "type")
                and messages[0].type == "system"
            ):
                nesgen_messages.append(
                    {"role": "system", "content": messages[0].content}
                )
            if last_user_message:
                nesgen_messages.append({"role": "user", "content": last_user_message})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger = logging.getLogger(__name__)
                logger.info(
                    f"NESGEN REQUEST - Messages count: {len(nesgen_messages)}, has_tool_results: {has_tool_results}"
                )

                response_json = self._make_request(nesgen_messages)
                logger.debug(f"NESGEN RAW RESPONSE FULL: {json.dumps(response_json)}")
                response_text = self._extract_response_text(response_json)
                logger.info(
                    f"NESGEN RESPONSE - Text preview: {response_text[:200] if response_text else '(EMPTY RESPONSE)'}"
                )

                is_json_response = False
                try:
                    stripped = response_text.strip()
                    if (stripped.startswith("{") and stripped.endswith("}")) or (
                        stripped.startswith("[") and stripped.endswith("]")
                    ):
                        json.loads(stripped)
                        is_json_response = True
                        logger.warning(
                            f"NESGEN returned JSON instead of natural language: {response_text[:100]}"
                        )
                except Exception:
                    pass

                tool_calls = []
                if self.bound_tools and not has_tool_results:
                    user_msg_str = (
                        last_user_message
                        if isinstance(last_user_message, str)
                        else str(last_user_message)
                    )
                    tool_calls = self._llm_select_tool(user_msg_str)
                    logger.info(
                        f"NESGEN LLM TOOL SELECTION - Found {len(tool_calls)} tool calls: {[tc['name'] for tc in tool_calls]}"
                    )
                    if tool_calls:
                        # Empty content when tool calls are present (standard LangChain convention)
                        response_text = ""
                        is_json_response = False

                if (not response_text or is_json_response) and has_tool_results:
                    logger.error(
                        "NESGEN returned invalid response after tool execution!"
                    )
                    tool_data_parts = []
                    for i in range(last_user_index + 1, len(messages)):
                        msg = messages[i]
                        if hasattr(msg, "type") and msg.type == "tool":
                            tool_name = getattr(msg, "name", "tool")
                            tool_data_parts.append(
                                f"[{tool_name} result]: {msg.content}"
                            )

                    if tool_data_parts:
                        response_text = (
                            "Here are the results from the tool execution:\n\n"
                            + "\n\n".join(tool_data_parts)
                        )
                    else:
                        response_text = "I've processed your request. The operation completed successfully."

                if tool_calls:
                    message = AIMessage(
                        content=response_text,
                        tool_calls=tool_calls,
                        additional_kwargs={"raw_response": response_json},
                    )
                else:
                    message = AIMessage(
                        content=response_text,
                        additional_kwargs={"raw_response": response_json},
                    )

                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(
                        f"NESGEN API request failed after {self.max_retries} attempts: {str(e)}"
                    )
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                raise Exception(f"Failed to parse NESGEN API response: {str(e)}")

        raise Exception(f"NESGEN API request failed: {str(last_error)}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation — delegates to sync implementation."""
        return self._generate(messages, stop, None, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "model_endpoint": self.model_endpoint,
            "api_base": self.api_base,
            "api_version": self.api_version,
            "temperature": self.temperature,
        }


def test_basic_connectivity():
    """Test basic connectivity without tool calling."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Connectivity")
    print("=" * 70)

    cfg = get_nesgen_config()
    chat_model = NesGenChatModel(
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        model_endpoint=cfg.model_endpoint,
        api_base=cfg.api_base,
        model=cfg.model,
        api_version=cfg.api_version,
    )
    test_message = HumanMessage(
        content="What is Generative AI? Please define in 1 line."
    )

    result = chat_model._generate([test_message])
    response_text = result.generations[0].message.content

    print("✅ SUCCESS")
    print(f"Response: {response_text}")
    return True


def _run_test(name: str, fn: Any) -> tuple:
    """Run a single test function, catching any exception."""
    try:
        result = fn()
        return (name, result, None)
    except Exception as exc:
        print(f"❌ EXCEPTION in '{name}': {exc}")
        return (name, False, exc)


def main():
    """Run all connectivity and tool-calling tests."""
    print("\n" + "=" * 70)
    print("NESGEN LLM CONNECTIVITY & TOOL CALLING TESTS")
    print("=" * 70)

    cfg = get_nesgen_config()
    CLIENT_ID = cfg.client_id
    CLIENT_SECRET = cfg.client_secret
    MODEL_ENDPOINT = cfg.model_endpoint
    API_BASE = cfg.api_base

    print(f"Endpoint (NESGEN_URL): {MODEL_ENDPOINT or '(not set)'}")
    print(f"API Base (NESGEN_API_BASE): {API_BASE or '(not set)'}")
    print(f"Client ID: {CLIENT_ID[:10]}..." if CLIENT_ID else "Client ID: Not set")

    if not CLIENT_ID or not CLIENT_SECRET:
        print("\n❌ ERROR: Missing environment variables")
        print("Please set NESTLE_CLIENT_ID and NESTLE_CLIENT_SECRET")
        return

    if not MODEL_ENDPOINT and not API_BASE:
        print("\n❌ ERROR: No API endpoint configured")
        print("Please set NESGEN_URL or NESGEN_API_BASE")
        return

    test_cases = [
        ("Basic Connectivity", test_basic_connectivity),
    ]

    results = [_run_test(name, fn) for name, fn in test_cases]

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, exc in results:
        if result:
            status = "✅ PASS"
        elif exc:
            status = f"💥 ERROR ({type(exc).__name__})"
        else:
            status = "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        if any(exc for _, _, exc in results):
            print(
                "\nNote: Some tests raised exceptions — check network/VPN access to the NESGEN API."
            )


if __name__ == "__main__":
    main()
