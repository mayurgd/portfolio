"""AgentTool Eval App - Enhanced V2 Implementation

Enhanced LangGraph agent with advanced intent routing, query decomposition,
and sequential tool invocation capabilities.

This is an enhanced version that preserves the original graph.py unchanged while
implementing a more sophisticated multi-agent architecture.

Architecture:

    User Query → Intent Router → Route Decision
                     ↓              ↓
                (Generic)      (Complex)
                     ↓              ↓
            Direct LLM      Query Decomposer
            Response             ↓
                     ↓       Analyze Complexity
                     ↓              ↓
                     ↓      (Single-Tool) (Multi-Tool)
                     ↓              ↓              ↓
                     ↓      Direct Tool    Decompose into
                     ↓      Execution      Subtasks
                     ↓              ↓              ↓
                     ↓              └──────────────┘
                     ↓                     ↓
                     ↓          Sequential Tool Invocation
                     ↓                     ↓
                     └─────────────────────┴──────→ END

Features:
    - Intent routing agent (generic vs specialized)
    - Query decomposition agent (single vs multi-tool)
    - Sequential tool invocation with dependency management
    - Native Langfuse Python SDK tracing (no LangChain integration)
    - Full MCP tool compatibility
    - Enhanced observability and metrics

Note:
    Requires MCP server to be running for tool access.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent_eval.agent.prompts import SYSTEM_PROMPT, SIMPLE_RESPONSE_PROMPT, SEQUENTIAL_STEP_PROMPT, SYNTHESIS_PROMPT
from agent_eval.agent.helpers import (
    build_tool_descriptions,
    classify_intent,
    analyze_complexity,
    extract_user_message,
    messages_to_dicts,
    extract_token_usage,
    prepare_generation_output,
    extract_response_content,
    get_langfuse_span,
    extract_tool_result_text,
)
from agent_eval.observability.tracer import get_tracer

logger = logging.getLogger(__name__)

    
# ══════════════════════════════════════════════════════════════════════════════
# State Definition
# ══════════════════════════════════════════════════════════════════════════════

class AgentStateV2(TypedDict):
    """Enhanced state for V2 agent graph.
    
    Attributes:
        messages: Conversation history with add_messages reducer
        intent: Intent classification ("GENERIC" or "SPECIALIZED")
        complexity: Query complexity ("SINGLE_TOOL" or "MULTI_TOOL")
        tool_results: Results from executed tools
        execution_plan: Ordered list of tool names to execute
        tool_selection_log: Log entry from the intent/decomposition step
        execution_plan_log: Log entries from tool execution steps
    """
    messages: Annotated[list, add_messages]
    intent: Optional[str]
    complexity: Optional[str]
    tool_results: Dict[str, Any]
    execution_plan: List[str]
    tool_selection_log: Optional[Dict[str, Any]]
    execution_plan_log: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════════
# Routing Functions
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_intent(state: AgentStateV2) -> str:
    """Route based on intent classification."""
    intent = state.get("intent", "GENERIC")
    if intent == "SPECIALIZED":
        logger.info("Routing to decompose_query")
        return "decompose_query"
    else:
        logger.info("Routing to respond_directly")
        return "respond_directly"


def _route_after_decomposition(state: AgentStateV2) -> str:
    """Route based on query complexity."""
    complexity = state.get("complexity", "SINGLE_TOOL")
    if complexity == "MULTI_TOOL":
        logger.info("Routing to execute_sequential_tools")
        return "execute_sequential_tools"
    else:
        logger.info("Routing to execute_agent")
        return "execute_agent"


def _should_continue(state: AgentStateV2) -> str:
    """Check if agent needs to call tools or is done."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and hasattr(last, "tool_calls") and last.tool_calls:
        logger.info("Routing to execute_tools: %s", [tc["name"] for tc in last.tool_calls])
        return "execute_tools"
    # No more tool calls — route to finalize_trace before END
    return "finalize_trace"


# ══════════════════════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_v2(model: BaseChatModel, tools: list):
    """Build and compile the enhanced V2 LangGraph agent.

    Args:
        model: chat model (must support bind_tools)
        tools: List of LangChain tool instances

    Returns:
        Compiled StateGraph ready for invocation
    """
    model_with_tools = model.bind_tools(tools)
    tracer = get_tracer()
    
    # DEBUG: Verify tools are bound
    bound_tools_count = len(getattr(model_with_tools, 'bound_tools', []))
    logger.info(f"Graph V2: model_with_tools has {bound_tools_count} tools bound")
    
    # Get model name for tracing
    model_name = getattr(model, "model_name", None) or getattr(model, "model", "unknown")
    
    # Build tool descriptions once
    tool_descriptions = build_tool_descriptions(tools)
    
    # Create tool lookup map
    tool_map = {getattr(t, "name", ""): t for t in tools}
    
    # ── Node: Route Intent ────────────────────────────────────────────────
    
    def route_intent(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Route user query based on intent classification."""
        
        last_user_msg = extract_user_message(state["messages"])
        if not last_user_msg:
            return {"intent": "GENERIC"}
        
        parent = get_langfuse_span(config)
        
        # Create node-level span
        node_span = tracer.start_span(
            parent,
            name="route_intent",
            input={"query": last_user_msg, "message_count": len(state["messages"])},
            metadata={
                "type": "agent_node",
                "node_name": "route_intent",
                "node_type": "intent_router",
                "query_length": len(last_user_msg),
                "tools_available": len(tools),
            }
        )
        
        # Nest generation inside node span
        generation = tracer.start_generation(
            node_span or parent,
            name="intent_classification",
            model=model_name,
            input={"query": last_user_msg, "context": "Routing user intent"},
            metadata={
                "node": "route_intent",
                "step": "intent_routing",
                "query_length": len(last_user_msg),
            },
        )
        
        intent = classify_intent(model, last_user_msg, tool_descriptions)
        
        output = {
            "intent": intent,
            "query_preview": last_user_msg[:100],
        }
        
        tracer.end_generation(generation, output=output)
        tracer.end_span(node_span, output={"intent": intent})
        
        tool_selection_log = {
            "step": "intent_routing",
            "intent": intent,
            "query": last_user_msg[:200],
            "routed_to": "decompose_query" if intent == "SPECIALIZED" else "respond_directly",
        }
        
        return {"intent": intent, "tool_selection_log": tool_selection_log}
    
    # ── Node: Respond Directly ────────────────────────────────────────────
    
    def respond_directly(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Generate direct response for generic queries without tools."""
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SIMPLE_RESPONSE_PROMPT)] + messages
        
        parent = get_langfuse_span(config)
        
        last_user_msg = extract_user_message(state["messages"])
        node_span = tracer.start_span(
            parent,
            name="respond_directly",
            input={"query": last_user_msg, "message_count": len(messages)},
            metadata={
                "type": "agent_node",
                "node_name": "respond_directly",
                "node_type": "direct_responder",
                "tools_bound": False,
                "intent": state.get("intent", "unknown"),
            }
        )
        
        generation = tracer.start_generation(
            node_span or parent,
            name="direct_response",
            model=model_name,
            input=messages_to_dicts(messages),
            metadata={
                "node": "respond_directly",
                "step": "generic_response",
                "tools_bound": False,
                "response_type": "direct",
            },
        )
        
        try:
            response = model.invoke(messages)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model invocation failed in respond_directly: {error_msg}")
            error_response = AIMessage(
                content=f"I apologize, but I encountered an error. Please try again. Error: {error_msg[:100]}"
            )
            tracer.end_generation(generation, output={"error": error_msg})
            tracer.end_span(node_span, output={"error": error_msg})
            return {"messages": [error_response]}
        
        output = prepare_generation_output(response)
        content = extract_response_content(response)
        output["response_length"] = len(content)
        output["response_type"] = "direct"
        
        tracer.end_generation(
            generation,
            output=output,
            usage_details=extract_token_usage(response) or None,
        )
        
        tracer.end_span(node_span, output={"response": content[:200]})
        
        return {"messages": [response]}
    
    # ── Node: Decompose Query ─────────────────────────────────────────────
    
    def decompose_query(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Analyze query complexity and decompose if needed."""
        
        last_user_msg = extract_user_message(state["messages"])
        if not last_user_msg:
            return {
                "complexity": "SINGLE_TOOL",
                "execution_plan": []
            }
        
        parent = get_langfuse_span(config)
        
        node_span = tracer.start_span(
            parent,
            name="decompose_query",
            input={"query": last_user_msg},
            metadata={
                "type": "agent_node",
                "node_name": "decompose_query",
                "node_type": "query_decomposer",
                "intent": state.get("intent", "unknown"),
            }
        )
        
        generation = tracer.start_generation(
            node_span or parent,
            name="query_decomposition",
            model=model_name,
            input={"query": last_user_msg, "context": "Analyzing query complexity"},
            metadata={
                "node": "decompose_query",
                "step": "complexity_analysis",
            },
        )
        
        complexity, tool_names = analyze_complexity(model, last_user_msg, tool_descriptions)
        
        output = {
            "complexity": complexity,
            "tools_required": tool_names,
            "tool_count": len(tool_names),
        }
        
        tracer.end_generation(generation, output=output)
        tracer.end_span(node_span, output=output)
        
        tool_selection_log = {
            "step": "query_decomposition",
            "intent": state.get("intent", "unknown"),
            "complexity": complexity,
            "execution_plan": tool_names,
            "tool_count": len(tool_names),
            "query": last_user_msg[:200],
            "routed_to": "execute_sequential_tools" if complexity == "MULTI_TOOL" else "execute_agent",
        }
        
        return {
            "complexity": complexity,
            "execution_plan": tool_names,
            "tool_selection_log": tool_selection_log,
        }
    
    # ── Node: Execute Agent ───────────────────────────────────────────────
    
    def execute_agent(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Execute agent with tool access for single-tool queries."""
        start_time = time.time()
        
        last_user_msg = extract_user_message(state["messages"])
        
        # Detect whether tool results are already present in the message state.
        # execute_agent is called twice in the single-tool path:
        #   1st call (no tool results yet): generate the tool call — use only system + current
        #      user message to prevent prior conversation context from polluting SQL generation.
        #   2nd call (tool results present): synthesize the final response — must include the
        #      full message history so the LLM can see the tool output.
        has_tool_results = any(isinstance(m, ToolMessage) for m in state["messages"])
        
        if has_tool_results:
            # Second pass: include full message history for response synthesis.
            # The NesGenChatModel._generate() detects ToolMessages after the last
            # HumanMessage and automatically injects a synthesis instruction —
            # do NOT append an extra HumanMessage here or it will be treated as
            # the "last user message" and the tool results will be missed.
            messages: list = list(state["messages"])
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        else:
            # First pass: use only system prompt + current user message to avoid
            # context pollution from prior queries (e.g. reusing a country filter)
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            if last_user_msg:
                messages.append(HumanMessage(content=last_user_msg))
        
        parent = get_langfuse_span(config)
        
        node_span = tracer.start_span(
            parent,
            name="execute_agent",
            input={"query": last_user_msg, "message_count": len(messages)},
            metadata={
                "type": "agent_node",
                "node_name": "execute_agent",
                "node_type": "agent_with_tools",
                "tools_bound": True,
                "tool_count": len(tools),
                "complexity": state.get("complexity", "unknown"),
                "intent": state.get("intent", "unknown"),
            }
        )
        
        generation = tracer.start_generation(
            node_span or parent,
            name="agent_with_tools",
            model=model_name,
            input=messages_to_dicts(messages),
            metadata={
                "node": "execute_agent",
                "node_type": "agent_with_tools",
                "tools_bound": True,
                "tool_count": len(tools),
                "complexity": state.get("complexity", "unknown"),
            },
        )
        
        try:
            response = model_with_tools.invoke(messages)
        except Exception as e:
            # Handle NESGEN API errors gracefully
            error_msg = str(e)
            logger.error(f"Model invocation failed: {error_msg}")
            
            # Create an error response message
            error_response = AIMessage(
                content=f"I apologize, but I encountered an error while processing your request. "
                        f"Please try again or contact support if the issue persists. "
                        f"Error: {error_msg[:100]}"
            )
            
            execution_time = time.time() - start_time
            
            tracer.end_generation(
                generation,
                output={"error": error_msg},
                metadata={
                    "execution_time_seconds": round(execution_time, 3),
                    "error": True,
                    "error_type": type(e).__name__,
                },
            )
            
            tracer.end_span(node_span, output={"error": error_msg})
            
            return {"messages": [error_response]}
        
        execution_time = time.time() - start_time
        has_tool_calls = hasattr(response, "tool_calls") and bool(response.tool_calls)
        tool_call_count = len(response.tool_calls) if has_tool_calls else 0
        
        usage = extract_token_usage(response)
        
        tracer.end_generation(
            generation,
            output=prepare_generation_output(response),
            usage_details=usage or None,
            metadata={
                "execution_time_seconds": round(execution_time, 3),
                "has_tool_calls": has_tool_calls,
                "tool_call_count": tool_call_count,
                "tools_requested": [tc["name"] for tc in response.tool_calls] if has_tool_calls else [],
            },
        )
        
        tracer.end_span(node_span, output={
            "has_tool_calls": has_tool_calls,
            "tool_call_count": tool_call_count,
        })
        
        exec_log_entry: Dict[str, Any] = {
            "step": "execute_agent",
            "has_tool_calls": has_tool_calls,
            "tools_selected": [tc["name"] for tc in response.tool_calls] if has_tool_calls else [],
            "tool_call_count": tool_call_count,
            "execution_time_seconds": round(execution_time, 3),
        }
        existing_exec_log: List[Dict[str, Any]] = list(state.get("execution_plan_log") or [])
        existing_exec_log.append(exec_log_entry)
        
        return {"messages": [response], "execution_plan_log": existing_exec_log}
    
    # ── Node: Execute Tools ───────────────────────────────────────────────
    
    async def execute_tools(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Execute tool calls asynchronously (MCP tools require async)."""
        overall_start = time.time()
        
        parent = get_langfuse_span(config)
        
        last_msg = state["messages"][-1]
        pending_calls = []
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            pending_calls = last_msg.tool_calls
        
        tool_span = tracer.start_span(
            parent,
            name="execute_tools",
            input=[{"name": tc["name"], "args": tc.get("args", {})} for tc in pending_calls],
            metadata={
                "type": "agent_node",
                "node_name": "execute_tools",
                "node_type": "tool_executor",
                "tool_call_count": len(pending_calls),
                "tools_requested": [tc["name"] for tc in pending_calls],
            },
        )
        
        tool_messages = []
        tool_results = []
        successful_tools = 0
        failed_tools = 0
        
        for tc in pending_calls:
            tool_start = time.time()
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tool_status = "success"
            
            try:
                tool = tool_map.get(tool_name)
                if not tool:
                    tool_output = f"Error: Tool '{tool_name}' not found"
                    tool_status = "not_found"
                    failed_tools += 1
                else:
                    result = await tool.ainvoke(tool_args)
                    # Extract plain JSON text from MCP result list
                    # (avoids str([TextContent(...)]) which the LLM cannot parse)
                    tool_output = extract_tool_result_text(result)
                    successful_tools += 1
                
                tool_msg = ToolMessage(
                    content=tool_output,
                    tool_call_id=tc.get("id", ""),
                    name=tool_name,
                )
                tool_messages.append(tool_msg)
                
                tool_execution_time = time.time() - tool_start
                
                child_span = tracer.start_span(
                    tool_span or parent,
                    name=f"tool:{tool_name}",
                    input=tool_args,
                    output=tool_output,
                    metadata={
                        "tool_name": tool_name,
                        "tool_status": tool_status,
                        "execution_time_seconds": round(tool_execution_time, 3),
                        "has_error": tool_status != "success",
                    },
                )
                tracer.end_span(child_span)
                
                tool_results.append({
                    "tool": tool_name,
                    "status": tool_status,
                    "execution_time": round(tool_execution_time, 3),
                })
                
            except Exception as e:
                tool_execution_time = time.time() - tool_start
                error_details = str(e)
                tool_status = "error"
                failed_tools += 1
                
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                error_msg = ToolMessage(
                    content=f"Error executing tool: {error_details}",
                    tool_call_id=tc.get("id", ""),
                    name=tool_name,
                )
                tool_messages.append(error_msg)
                
                child_span = tracer.start_span(
                    tool_span or parent,
                    name=f"tool:{tool_name}",
                    input=tool_args,
                    output={"error": error_details},
                    metadata={
                        "tool_name": tool_name,
                        "tool_status": "error",
                        "execution_time_seconds": round(tool_execution_time, 3),
                        "error_message": error_details,
                        "has_error": True,
                    },
                )
                tracer.end_span(child_span)
                
                tool_results.append({
                    "tool": tool_name,
                    "status": "error",
                    "execution_time": round(tool_execution_time, 3),
                    "error": error_details,
                })
        
        overall_execution_time = time.time() - overall_start
        success_rate = (successful_tools / len(pending_calls) * 100) if pending_calls else 0
        
        tracer.end_span(
            tool_span,
            output=tool_results,
            metadata={
                "total_execution_time_seconds": round(overall_execution_time, 3),
                "total_tools": len(pending_calls),
                "successful_tools": successful_tools,
                "failed_tools": failed_tools,
                "success_rate_percent": round(success_rate, 2),
            },
        )
        
        exec_log_entry = {
            "step": "execute_tools",
            "tools_executed": [r["tool"] for r in tool_results],
            "tool_results": tool_results,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "total_execution_time_seconds": round(overall_execution_time, 3),
            "success_rate_percent": round(success_rate, 2),
        }
        existing_exec_log = list(state.get("execution_plan_log") or [])
        existing_exec_log.append(exec_log_entry)
        
        return {"messages": tool_messages, "execution_plan_log": existing_exec_log}
    
    # ── Node: Execute Sequential Tools ────────────────────────────────────
    
    async def execute_sequential_tools(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Execute multiple tools sequentially based on execution plan."""
        overall_start = time.time()
        
        parent = get_langfuse_span(config)
        execution_plan = state.get("execution_plan", [])
        
        node_span = tracer.start_span(
            parent,
            name="execute_sequential_tools",
            input={"execution_plan": execution_plan},
            metadata={
                "type": "agent_node",
                "node_name": "execute_sequential_tools",
                "node_type": "sequential_executor",
                "tool_count": len(execution_plan),
                "complexity": state.get("complexity", "unknown"),
            }
        )
        
        # For tool-based SQL generation, use only the system prompt + current user message.
        # Including prior conversation turns risks the LLM reusing filters from earlier queries.
        _current_user_msg = extract_user_message(state["messages"])
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        if _current_user_msg:
            messages.append(HumanMessage(content=_current_user_msg))
        
        all_tool_results = {}
        tool_messages = []
        
        # Execute each tool in the plan sequentially
        for idx, tool_name in enumerate(execution_plan):
            step_start = time.time()
            previous_keys = list(all_tool_results.keys())
            
            step_span = tracer.start_span(
                node_span or parent,
                name=f"sequential_step_{idx+1}",
                input={"tool_name": tool_name, "step": idx+1, "total_steps": len(execution_plan)},
                metadata={
                    "step_number": idx+1,
                    "tool_name": tool_name,
                    "previous_results": previous_keys,
                }
            )

            # Build context hint: if prior results exist, tell the LLM it can reference them
            context_hint = ""
            if previous_keys:
                context_hint = (
                    f" (you may reference results already retrieved from: {', '.join(previous_keys)})"
                )

            # Inject a per-step instruction so the LLM calls ONLY the target tool
            step_instruction = SEQUENTIAL_STEP_PROMPT.format(
                step_number=idx + 1,
                total_steps=len(execution_plan),
                tool_name=tool_name,
                context_hint=context_hint,
                previous_results=previous_keys if previous_keys else "none",
            )
            step_messages = messages + [HumanMessage(content=step_instruction)]

            # Ask LLM to generate tool call for this specific tool
            generation = tracer.start_generation(
                step_span or node_span or parent,
                name=f"generate_tool_call_{tool_name}",
                model=model_name,
                input=messages_to_dicts(step_messages),
                metadata={
                    "step": idx+1,
                    "target_tool": tool_name,
                    "sequential_execution": True,
                },
            )
            
            try:
                response = model_with_tools.invoke(step_messages)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Model invocation failed in sequential tool execution: {error_msg}")
                tracer.end_generation(generation, output={"error": error_msg})
                tracer.end_span(step_span, output={"error": error_msg})
                # Continue to next tool or return error
                error_response = AIMessage(
                    content=f"Error executing tool {tool_name}: {error_msg[:100]}"
                )
                messages.append(error_response)
                continue
            
            tracer.end_generation(
                generation,
                output=prepare_generation_output(response),
                usage_details=extract_token_usage(response) or None,
            )
            
            # Execute ALL tool calls from the response
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Add the AI response with tool calls first
                messages.append(response)
                
                # Process ALL tool calls to satisfy OpenAI's requirement
                for tc in response.tool_calls:
                    tc_name = tc["name"]
                    tc_args = tc.get("args", {})
                    tc_id = tc.get("id", "")
                    
                    try:
                        tool = tool_map.get(tc_name)
                        if tool:
                            result = await tool.ainvoke(tc_args)
                            # Extract plain JSON text from MCP result list
                            result_text = extract_tool_result_text(result)
                            
                            # Store result if it's our target tool
                            if tc_name == tool_name:
                                all_tool_results[tool_name] = result_text
                                logger.info(f"Sequential step {idx+1}: {tool_name} completed")
                            
                            tool_msg = ToolMessage(
                                content=result_text,
                                tool_call_id=tc_id,
                                name=tc_name,
                            )
                            tool_messages.append(tool_msg)
                            messages.append(tool_msg)
                        else:
                            # Tool not found - still need to respond to satisfy OpenAI
                            error_msg = ToolMessage(
                                content=f"Error: Tool '{tc_name}' not found",
                                tool_call_id=tc_id,
                                name=tc_name,
                            )
                            tool_messages.append(error_msg)
                            messages.append(error_msg)
                            
                    except Exception as e:
                        logger.error(f"Tool {tc_name} failed: {e}")
                        error_msg = ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tc_id,
                            name=tc_name,
                        )
                        tool_messages.append(error_msg)
                        messages.append(error_msg)
            
            step_time = time.time() - step_start
            tracer.end_span(step_span, output={
                "tool_name": tool_name,
                "execution_time": round(step_time, 3),
                "result_available": tool_name in all_tool_results,
            })
        
        overall_time = time.time() - overall_start
        
        exec_log_entry = {
            "step": "execute_sequential_tools",
            "execution_plan": execution_plan,
            "tools_executed": list(all_tool_results.keys()),
            "total_execution_time_seconds": round(overall_time, 3),
            "success_count": len(all_tool_results),
            "step_details": [
                {
                    "tool": tool_name,
                    "completed": tool_name in all_tool_results,
                }
                for tool_name in execution_plan
            ],
        }
        existing_exec_log = list(state.get("execution_plan_log") or [])
        existing_exec_log.append(exec_log_entry)
        
        tracer.end_span(node_span, output={
            "tools_executed": list(all_tool_results.keys()),
            "total_execution_time": round(overall_time, 3),
            "success_count": len(all_tool_results),
        })
        
        # Generate final response with all tool results.
        #
        # IMPORTANT: Do NOT pass the accumulated `messages` list directly to the
        # model here.  That list contains multiple HumanMessages (one per step
        # instruction), so NesGenChatModel._generate() will treat the LAST step
        # instruction as the "user query" and only see the single ToolMessage that
        # follows it — losing the original question and all earlier tool results.
        #
        # Instead, build a clean synthesis message list:
        #   [0] SystemMessage  (system prompt + synthesis instruction)
        #   [1] HumanMessage   (the ORIGINAL user query)
        #   [2..N] HumanMessage per tool result  (formatted so _generate() CASE 1
        #          picks them up as tool results after the last user message)
        #
        # _generate() CASE 1 triggers when ToolMessages exist after the last
        # HumanMessage.  Since `model` has no tools bound we use plain
        # HumanMessages formatted as "Tool 'X' returned: ..." — the synthesis
        # instruction in the system prompt tells the LLM to treat them as data.

        synthesis_system = SYSTEM_PROMPT + "\n\n" + SYNTHESIS_PROMPT
        synthesis_messages: list = [SystemMessage(content=synthesis_system)]

        if _current_user_msg:
            synthesis_messages.append(HumanMessage(content=_current_user_msg))

        # Append each tool result as a ToolMessage so _generate() CASE 1 fires
        for tm in tool_messages:
            synthesis_messages.append(tm)

        final_generation = tracer.start_generation(
            node_span or parent,
            name="final_response_generation",
            model=model_name,
            input=messages_to_dicts(synthesis_messages),
            metadata={
                "step": "final_synthesis",
                "tools_used": list(all_tool_results.keys()),
            },
        )
        
        try:
            final_response = model.invoke(synthesis_messages)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model invocation failed in final response generation: {error_msg}")
            error_response = AIMessage(
                content=f"I executed the tools but encountered an error generating the final response. Error: {error_msg[:100]}"
            )
            tracer.end_generation(final_generation, output={"error": error_msg})
            return {
                "messages": [error_response],
                "tool_results": all_tool_results
            }
        
        tracer.end_generation(
            final_generation,
            output=prepare_generation_output(final_response),
            usage_details=extract_token_usage(final_response) or None,
        )
        
        return {
            "messages": [final_response],
            "tool_results": all_tool_results,
            "execution_plan_log": existing_exec_log,
        }
    
    # ── Node: Finalize Trace ──────────────────────────────────────────────

    def finalize_trace(state: AgentStateV2, config: RunnableConfig) -> dict:
        """Update the root Langfuse trace with accurate post-run metrics.

        This node runs last on every execution path so the agent — not app.py —
        is responsible for recording what actually happened (LLM call count,
        tools called).  It does NOT modify agent state.
        """
        parent = get_langfuse_span(config)
        if parent is None:
            return {}

        execution_plan_log: List[Dict[str, Any]] = list(state.get("execution_plan_log") or [])
        steps = {e.get("step") for e in execution_plan_log}
        intent = state.get("intent", "GENERIC")
        complexity = state.get("complexity", "SINGLE_TOOL")

        # ── LLM call count ────────────────────────────────────────────────
        # route_intent always makes 1 LLM call (intent_classification).
        llm_call_count = 1

        if intent != "SPECIALIZED":
            # GENERIC path: intent(1) + direct_response(1)
            llm_call_count += 1
        elif complexity == "MULTI_TOOL" or "execute_sequential_tools" in steps:
            # SPECIALIZED / MULTI_TOOL path:
            #   intent(1) + decompose(1) + N×tool_call_generation + 1×final_synthesis
            seq_entry = next(
                (e for e in execution_plan_log if e.get("step") == "execute_sequential_tools"),
                {},
            )
            plan_len = len(seq_entry.get("execution_plan", []))
            llm_call_count += 1 + plan_len + 1  # decompose + N tool gens + final synthesis
        else:
            # SPECIALIZED / SINGLE_TOOL path:
            #   intent(1) + decompose(1) + agent_tool_call(1) + agent_synthesis(1)
            llm_call_count += 3

        # ── Tools actually called ─────────────────────────────────────────
        tools_called: List[str] = []
        for entry in execution_plan_log:
            if entry.get("step") in ("execute_tools", "execute_sequential_tools"):
                tools_called.extend(entry.get("tools_executed", []))

        try:
            parent.update(
                metadata={
                    "llm_call_count": llm_call_count,
                    "tools_called": tools_called,
                    "tools_called_count": len(tools_called),
                }
            )
        except Exception:
            logger.warning("finalize_trace: failed to update root trace metadata", exc_info=True)

        return {}

    # ── Assemble Graph ────────────────────────────────────────────────────
    
    graph = StateGraph(AgentStateV2)
    
    # Add nodes
    graph.add_node("route_intent", route_intent)
    graph.add_node("respond_directly", respond_directly)
    graph.add_node("decompose_query", decompose_query)
    graph.add_node("execute_agent", execute_agent)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("execute_sequential_tools", execute_sequential_tools)
    graph.add_node("finalize_trace", finalize_trace)
    
    # Define graph flow
    graph.set_entry_point("route_intent")
    
    # Route based on intent
    graph.add_conditional_edges("route_intent", _route_after_intent)
    
    # All terminal paths funnel through finalize_trace before END
    graph.add_edge("respond_directly", "finalize_trace")
    
    # Decompose query routes based on complexity
    graph.add_conditional_edges("decompose_query", _route_after_decomposition)
    
    # Single-tool path: execute_agent -> execute_tools -> execute_agent -> finalize_trace -> END
    graph.add_conditional_edges("execute_agent", _should_continue)
    graph.add_edge("execute_tools", "execute_agent")
    
    # Multi-tool path: execute_sequential_tools -> finalize_trace -> END
    graph.add_edge("execute_sequential_tools", "finalize_trace")
    
    # finalize_trace is always the last node
    graph.add_edge("finalize_trace", END)
    
    logger.info("Graph V2 compiled with enhanced architecture (%d tools)", len(tools))
    
    return graph.compile()





# ══════════════════════════════════════════════════════════════════════════════
# Main Function for Testing
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    """Main function to test the enhanced V2 graph implementation.
    
    Run this script directly to verify the implementation:
        python -m agent_eval.agent.graph_v2
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Testing Enhanced Graph V2 Implementation")
    print("=" * 80)
    
    # Initialize components
    print("\n1. Initializing LLM and tools...")
    try:
        from agent_eval.llm.factory import create_model
        model = create_model()
        model_name = getattr(model, 'model_name', None) or getattr(model, 'model', 'unknown')
        print(f"   ✓ LLM initialized: {model_name}")
    except Exception as e:
        print(f"   ✗ Failed to initialize LLM: {e}")
        sys.exit(1)
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from agent_eval.config import get_mcp_config
        
        mcp_config = get_mcp_config()
        mcp_client = MultiServerMCPClient({
            "customer_support": {
                "url": mcp_config.url,
                "transport": "streamable_http",
            }
        })
        tools = await mcp_client.get_tools()
        print(f"   ✓ MCP tools loaded: {len(tools)} tools")
        for tool in tools:
            print(f"     - {getattr(tool, 'name', 'unknown')}")
    except Exception as e:
        print(f"   ✗ Failed to load MCP tools: {e}")
        print("   Note: Make sure MCP server is running:")
        print("   python -m agent_eval.mcp_server.server")
        sys.exit(1)
    
    # Build graph
    print("\n2. Building enhanced V2 graph...")
    try:
        graph = build_graph_v2(model, tools)
        agent_graph = graph.get_graph()
        agent_graph.draw_png("agent_graph.png")
        print("Graph saved as agent_graph.png")
        print("   ✓ Graph V2 compiled successfully")   
    except Exception as e:
        print(f"   ✗ Failed to build graph: {e}")
        sys.exit(1)
    
    # Test queries
    test_cases = [
        {
            "name": "Generic Query (Direct Response)",
            "query": "Hello! What can you help me with?",
            "expected_path": "route_intent -> respond_directly"
        },
        {
            "name": "Single-Tool Query",
            "query": "Check the status of order ORD-1001",
            "expected_path": "route_intent -> decompose_query -> execute_agent -> execute_tools"
        },
        {
            "name": "Multi-Tool Query",
            "query": "Check order ORD-1001 and update the shipping address to 456 New St",
            "expected_path": "route_intent -> decompose_query -> execute_sequential_tools"
        }
    ]
    
    tracer = get_tracer()
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 80}")
        print(f"Test Case {idx}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Path: {test_case['expected_path']}")
        print(f"{'-' * 80}")
        
        try:
            # Create trace for this test
            root_trace = tracer.start_trace(
                name=f"test_v2_{idx}",
                session_id="test_session",
                user_id="test_user",
                input={"query": test_case['query']},
                metadata={
                    "test_case": test_case['name'],
                    "graph_version": "v2"
                },
                tags=["test", "graph_v2"]
            )
            
            # Invoke graph
            result = await graph.ainvoke(
                {
                    "messages": [{"role": "user", "content": test_case['query']}],
                    "intent": None,
                    "complexity": None,
                    "tool_results": {},
                    "execution_plan": [],
                    "tool_selection_log": None,
                    "execution_plan_log": [],
                },
                config={"configurable": {"langfuse_span": root_trace}}
            )
            
            # Extract response
            response_content = "No response"
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response_content = getattr(last_message, "content", str(last_message))
                print(f"\n✓ Response received:")
                print(f"  Intent:     {result.get('intent', 'N/A')}")
                print(f"  Complexity: {result.get('complexity', 'N/A')}")
                print(f"  Response:   {str(response_content)[:200]}...")
                
                # Print tool selection log
                tool_sel = result.get("tool_selection_log")
                if tool_sel:
                    print(f"\n  📋 Tool Selection Log:")
                    print(f"     step       : {tool_sel.get('step', '—')}")
                    print(f"     intent     : {tool_sel.get('intent', '—')}")
                    print(f"     complexity : {tool_sel.get('complexity', '—')}")
                    print(f"     plan       : {tool_sel.get('execution_plan', [])}")
                    print(f"     routed_to  : {tool_sel.get('routed_to', '—')}")
                
                # Print execution plan log
                exec_log = result.get("execution_plan_log") or []
                if exec_log:
                    print(f"\n  ⚙️  Execution Plan Log ({len(exec_log)} step(s)):")
                    for i, entry in enumerate(exec_log, 1):
                        step = entry.get("step", "?")
                        if step == "execute_agent":
                            print(f"     [{i}] execute_agent — tools: {entry.get('tools_selected', [])} ({entry.get('execution_time_seconds', 0)}s)")
                        elif step == "execute_tools":
                            print(f"     [{i}] execute_tools — {entry.get('successful_tools', 0)} ok / {entry.get('failed_tools', 0)} failed ({entry.get('total_execution_time_seconds', 0)}s)")
                            for r in entry.get("tool_results", []):
                                print(f"          • {r.get('tool')} [{r.get('status')}] {r.get('execution_time', 0)}s")
                        elif step == "execute_sequential_tools":
                            print(f"     [{i}] execute_sequential_tools — plan: {entry.get('execution_plan', [])} ({entry.get('total_execution_time_seconds', 0)}s)")
                            for d in entry.get("step_details", []):
                                icon = "✓" if d.get("completed") else "✗"
                                print(f"          {icon} {d.get('tool')}")
            else:
                print(f"\n✗ No response received")
            
            # Update trace
            if root_trace:
                tracer.update_trace(
                    root_trace,
                    output={"response": str(response_content)[:200]},
                    metadata={"test_status": "success"}
                )
            
            tracer.flush()
            
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            logger.error(f"Test case {idx} failed", exc_info=True)
    
    print(f"\n{'=' * 80}")
    print("Testing Complete!")
    print(f"{'=' * 80}")
    
    if tracer.enabled:
        print("\n✓ Check Langfuse dashboard for detailed traces")
    else:
        print("\n⚠ Langfuse tracing not enabled - set credentials in .env")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
