"""
AgentTool Eval App - Reference Implementation

Streamlit chat UI demonstrating a agent-eval with tool-calling capabilities.

This is an example application for AI developers and backend engineers to learn from
and adapt for their own agent-with-tool-calling projects.

The agent uses tools from the MCP server exclusively.
Langfuse tracing uses the Python SDK for controlled manual instrumentation.

Run::

    # Start MCP server first:
    python3 -m agent_eval.mcp_server.server

    # Then launch Streamlit:
    streamlit run agent_eval/ui/app.py
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import List, Optional

import nest_asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agent_eval.agent.graph_v2 import build_graph_v2
from agent_eval.config import get_mcp_config
from agent_eval.llm.factory import create_model, check_credentials
from agent_eval.observability.tracer import get_tracer

# Allow nested event loops (Streamlit compat)
nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


async def _load_mcp_tools() -> list:
    """Load tools from the MCP server. Raises exception if server is not available."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_config = get_mcp_config()
    mcp_client = MultiServerMCPClient(
        {
            "customer_support": {
                "url": mcp_config.url,
                "transport": "streamable_http",
            }
        }
    )
    tools = await mcp_client.get_tools()
    logger.info("Loaded %d tools from MCP server", len(tools))
    return tools


def _get_tools() -> list:
    """Get tools from MCP server."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_load_mcp_tools())


async def _invoke_agent(
    user_message: str,
    chat_history: List[dict],
    session_id: str,
    provider: str,
    tools: list,
) -> tuple[str, Optional[dict], list]:
    """Build the graph and invoke it. Returns (response_text, tool_selection_log, execution_plan_log)."""
    model = create_model(provider=provider)
    graph = build_graph_v2(model, tools)

    messages = []
    # Keep only the last 2 conversation turns to limit context
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    tracer = get_tracer()
    root_span = tracer.start_trace(
        name="agent_execution",
        session_id=session_id,
        user_id="streamlit_user",
        input=user_message,
        metadata={
            "type": "agent_trace",
            "provider": provider,
            "has_history": len(chat_history) > 0,
            "history_length": len(chat_history),
            "interface": "streamlit_ui",
            "tools_available_count": len(tools),
            "tools_available": [t.name for t in tools],
        },
        tags=["agent", "langgraph", "customer_support", provider],
    )

    run_config = RunnableConfig(configurable={"langfuse_span": root_span}) if root_span else None

    result = await graph.ainvoke(
        {
            "messages": messages,
            "intent": None,
            "complexity": None,
            "tool_results": {},
            "execution_plan": [],
            "tool_selection_log": None,
            "execution_plan_log": [],
        },
        config=run_config,
    )

    from agent_eval.agent.helpers import extract_response_content

    response_text = "I'm sorry, I couldn't process your request."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = extract_response_content(msg)
            if content:
                response_text = content
                break

    tool_selection_log: Optional[dict] = result.get("tool_selection_log")
    execution_plan_log: list = result.get("execution_plan_log") or []

    # Record actual tool calls in trace metadata so eval_ragas.py can read
    # ``expected_tool_calls`` directly from the trace (Priority 1 lookup).
    actual_tool_calls: List[dict] = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                actual_tool_calls.append({"name": tc["name"], "args": tc.get("args", {})})

    response_length = len(response_text)
    total_messages = len(result.get("messages", []))

    if root_span:
        try:
            root_span.update(
                output=response_text,
                metadata={
                    "response_length": response_length,
                    "total_messages_in_result": total_messages,
                    "conversation_turns": len(chat_history) + 1,
                    "expected_tool_calls": actual_tool_calls,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to update trace: {e}")
    
    tracer.flush()

    return response_text, tool_selection_log, execution_plan_log


def _render_agent_logs(tool_selection_log: Optional[dict], execution_plan_log: list) -> None:
    """Render tool selection and execution plan logs as collapsible UI sections."""

    if not tool_selection_log and not execution_plan_log:
        return

    with st.expander("🔍 Agent reasoning", expanded=False):

        if tool_selection_log:
            step = tool_selection_log.get("step", "")
            intent = tool_selection_log.get("intent", "—")
            routed_to = tool_selection_log.get("routed_to", "—")

            st.markdown("**🧭 Tool Selection**")

            col1, col2 = st.columns(2)
            with col1:
                intent_colour = "🟢" if intent == "GENERIC" else "🔵"
                st.markdown(f"**Intent:** {intent_colour} `{intent}`")
            with col2:
                st.markdown(f"**Routed to:** `{routed_to}`")

            if step == "query_decomposition":
                complexity = tool_selection_log.get("complexity", "—")
                plan = tool_selection_log.get("execution_plan", [])
                complexity_colour = "🟡" if complexity == "MULTI_TOOL" else "🔵"
                st.markdown(f"**Complexity:** {complexity_colour} `{complexity}`")
                if plan:
                    st.markdown("**Planned tools (in order):**")
                    for i, tool_name in enumerate(plan, 1):
                        st.markdown(f"&nbsp;&nbsp;{i}. `{tool_name}`")

        if execution_plan_log:
            st.markdown("---")
            st.markdown("**⚙️ Execution Plan**")

            for entry in execution_plan_log:
                step = entry.get("step", "unknown")

                if step == "execute_agent":
                    tools_selected = entry.get("tools_selected", [])
                    exec_time = entry.get("execution_time_seconds", 0)
                    has_calls = entry.get("has_tool_calls", False)
                    if has_calls and tools_selected:
                        st.markdown(
                            "🤖 **Agent** selected tool(s): "
                            + ", ".join(f"`{t}`" for t in tools_selected)
                            + f" *(in {exec_time}s)*"
                        )
                    else:
                        st.markdown(f"🤖 **Agent** responded directly *(in {exec_time}s)*")

                elif step == "execute_tools":
                    tool_results = entry.get("tool_results", [])
                    exec_time = entry.get("total_execution_time_seconds", 0)
                    success = entry.get("successful_tools", 0)
                    failed = entry.get("failed_tools", 0)
                    st.markdown(
                        f"🔧 **Tool execution** — {success} succeeded, {failed} failed"
                        f" *(total {exec_time}s)*"
                    )
                    for r in tool_results:
                        t_name = r.get("tool", "?")
                        t_status = r.get("status", "?")
                        t_time = r.get("execution_time", 0)
                        icon = "✅" if t_status == "success" else "❌"
                        err = f" — `{r['error']}`" if "error" in r else ""
                        st.markdown(
                            f"&nbsp;&nbsp;{icon} `{t_name}` ({t_status}, {t_time}s){err}"
                        )

                elif step == "execute_sequential_tools":
                    plan = entry.get("execution_plan", [])
                    executed = entry.get("tools_executed", [])
                    exec_time = entry.get("total_execution_time_seconds", 0)
                    st.markdown(
                        f"🔀 **Sequential execution** — {len(executed)}/{len(plan)} tools"
                        f" *(total {exec_time}s)*"
                    )
                    for detail in entry.get("step_details", []):
                        t_name = detail.get("tool", "?")
                        completed = detail.get("completed", False)
                        icon = "✅" if completed else "❌"
                        st.markdown(f"&nbsp;&nbsp;{icon} `{t_name}`")


def main() -> None:
    st.set_page_config(page_title="Bakehouse Data Agent", page_icon="🥐", layout="centered")
    st.title("🥐 Bakehouse Data Agent")
    st.caption("Databricks MCP · LangGraph · Langfuse")
    st.info("💡 Ask questions about franchises, customers, suppliers, transactions, and reviews — the agent generates SQL and queries Databricks automatically.")

    with st.sidebar:
        st.header("⚙️ Configuration")

        provider = st.radio(
            "LLM Provider",
            options=["nestle", "openai"],
            index=0,
            help="Choose which LLM backend to use. NESTLE is primary, OpenAI is fallback. Set the matching credentials in `.env`.",
        )

        creds_ok, creds_msg = check_credentials(provider=provider)
        if creds_ok:
            st.success(f"✅ {creds_msg}")
        else:
            st.error(f"❌ {creds_msg}")

        st.divider()
        st.header("💡 Try these queries")

        st.subheader("🟢 Single-Tool Queries")
        st.caption("Each answered with exactly one Databricks tool call")
        single_tool_samples = [
            "List all approved suppliers that provide Flour",
            "Show me the top 10 products by total quantity sold",
            "Show me the top 10 customers by total number of transactions",
        ]
        for s in single_tool_samples:
            st.code(s, language=None)

        st.subheader("🟣 Multi-Tool Queries (3+ tools)")
        st.caption("Each scenario requires more than 2 tool invocations to complete")
        multi_tool_samples = [
            "Find the top 3 franchises by total revenue, look up their supplier details, and then show the most recent customer reviews for each of those franchises",
            "Identify the customers who made the highest-value transactions, retrieve their full profile details including city and country, then find all franchises in those same countries and show the latest reviews for those franchises",
            "Get all suppliers located in Europe, find the franchises linked to those suppliers, retrieve the sales transactions for those franchises from the past year, and then pull the customer reviews for those franchises to compare satisfaction with revenue",
        ]
        for s in multi_tool_samples:
            st.code(s, language=None)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    tool_cache_key = "tools_mcp"
    if tool_cache_key not in st.session_state:
        try:
            st.session_state[tool_cache_key] = _get_tools()
        except Exception as e:
            st.error(f"❌ Failed to load tools from MCP server: {e}")
            st.info("Make sure the MCP server is running: `python3 -m agent_eval.mcp_server.server`")
            st.stop()
    
    tools = st.session_state[tool_cache_key]

    with st.sidebar:
        st.caption(f"🔧 {len(tools)} tools loaded from MCP server")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_agent_logs(
                    msg.get("tool_selection_log"),
                    msg.get("execution_plan_log", []),
                )

    if prompt := st.chat_input("How can I help you today?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            if not creds_ok:
                response = (
                    f"⚠️ **LLM credentials not configured for `{provider}`.**\n\n"
                    f"{creds_msg}\n\n"
                    "I need the LLM to generate responses — even for simple greetings."
                )
                tool_selection_log = None
                execution_plan_log = []
            else:
                with st.spinner("Thinking …"):
                    try:
                        response, tool_selection_log, execution_plan_log = asyncio.get_event_loop().run_until_complete(
                            _invoke_agent(
                                prompt,
                                st.session_state.messages[:-1],
                                st.session_state.session_id,
                                provider,
                                tools,
                            )
                        )
                    except Exception as e:
                        logger.exception("Agent invocation failed")
                        error_msg = str(e)
                        if "client_id" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
                            response = "⚠️ **Authentication failed.** Please check your LLM credentials in `.env`."
                        else:
                            response = f"⚠️ Sorry, something went wrong.\n\n`{error_msg}`"
                        tool_selection_log = None
                        execution_plan_log = []

            st.markdown(response)
            _render_agent_logs(tool_selection_log, execution_plan_log)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "tool_selection_log": tool_selection_log,
            "execution_plan_log": execution_plan_log,
        })


if __name__ == "__main__":
    main()
