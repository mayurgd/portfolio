"""
Run evaluation experiment using Langfuse SDK.

Follows Langfuse experiment pattern:
https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk

This utility:
1. Fetches dataset from Langfuse
2. Runs the agent on each dataset item (using MCP tools via graph_v2)
3. Captures traces and observations in Langfuse (tagged with the experiment name)

Scoring is intentionally NOT done here — run eval_ragas.py / eval_deepeval.py
afterwards, filtering by the experiment tag, to evaluate the captured traces.

Usage:
    # Start MCP server first:
    python -m agent_eval.mcp_server.server

    # Run experiment with defaults (user=eval-tester, env=eval, session=<timestamp>):
    python -m evaluation.run_experiment --dataset-name "tool-call-eval" --experiment-name "exp-001"

    # Override user / session / environment:
    python -m evaluation.run_experiment \
        --dataset-name "tool-call-eval" \
        --experiment-name "exp-001" \
        --user "alice" \
        --session "session-42" \
        --environment "staging"

    # Then evaluate the captured traces:
    python -m evaluation.eval_ragas    --tag "exp-001"
    python -m evaluation.eval_deepeval --tag "exp-001"
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langfuse.client import Langfuse

from agent_eval.config import get_langfuse_config, get_mcp_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def extract_tool_calls_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from agent messages."""
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "args": tc.get("args", {})})
    return tool_calls


async def _load_mcp_tools() -> list:
    """Load tools from the MCP server."""
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
    logger.info(f"Loaded {len(tools)} tools from MCP server")
    return tools


async def run_experiment_async(
    dataset_name: str,
    experiment_name: str,
    user: str = "eval-tester",
    session: Optional[str] = None,
    environment: str = "eval",
):
    """Run evaluation experiment (async).

    Args:
        dataset_name:    Langfuse dataset name to fetch items from.
        experiment_name: Tag applied to every trace; used to filter later.
        user:            Langfuse user_id on each trace (default: "eval-tester").
        session:         Langfuse session_id shared across all traces in this run.
                         Defaults to an ISO-8601 UTC timestamp so each run gets
                         its own session automatically.
        environment:     Deployment environment label stored in trace metadata
                         (default: "eval").

    For each dataset item:
      - Creates a Langfuse trace tagged with ``experiment_name``
      - Stores ``expected_tool_calls`` in trace metadata so eval scripts can
        find the ground truth without re-fetching the dataset
      - Runs the agent and records actual tool calls + final response as the
        trace output
      - Does NOT score — scoring is delegated to eval_ragas / eval_deepeval
    """
    if session is None:
        session = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = get_langfuse_config()
    if not config.enabled:
        logger.error("Langfuse not configured")
        sys.exit(1)

    try:
        import httpx
        if config.ssl_verify is False:
            import os
            import urllib3
            os.environ.setdefault("CURL_CA_BUNDLE", "")
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        httpx_client = httpx.Client(verify=config.ssl_verify)
        client = Langfuse(
            public_key=config.public_key,
            secret_key=config.secret_key,
            host=config.host,
            httpx_client=httpx_client,
        )
        logger.info(f"Connected to Langfuse at {config.host}")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        sys.exit(1)

    try:
        dataset = client.get_dataset(dataset_name)
        items = list(dataset.items)
        logger.info(f"Loaded {len(items)} items from dataset '{dataset_name}'")
    except Exception as e:
        error_str = str(e).lower()
        if "not found" in error_str or "404" in error_str:
            logger.error(
                f"Dataset '{dataset_name}' not found in Langfuse.\n"
                f"  → Upload it first with:\n"
                f"      python -m evaluation.load_dataset --dataset-name \"{dataset_name}\"\n"
                f"  → Then re-run this command."
            )
        else:
            logger.error(f"Failed to fetch dataset '{dataset_name}': {e}")
        sys.exit(1)

    try:
        tools = await _load_mcp_tools()
    except Exception as e:
        logger.error(f"Failed to load MCP tools (is the MCP server running?): {e}")
        sys.exit(1)

    from agent_eval.agent.graph_v2 import build_graph_v2
    from agent_eval.llm.factory import create_model

    model = create_model()
    graph = build_graph_v2(model, tools)
    logger.info("Initialized agent graph (graph_v2) with MCP tools")

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"  user={user}  session={session}  environment={environment}")
    logger.info("Traces will be tagged with the experiment name for later evaluation.")
    success_count = 0

    for item in items:
        item_id = item.id
        user_input = item.input
        expected_output = item.expected_output or {}
        expected_tool_calls = expected_output.get("tool_calls", [])
        item_metadata = item.metadata or {}

        logger.info(f"\nProcessing: {item_id}")

        trace = client.trace(
            name=f"{experiment_name}-{item_id}",
            input=user_input,
            user_id=user,
            session_id=session,
            metadata={
                "experiment": experiment_name,
                "dataset_name": dataset_name,
                "dataset_item_id": item_id,
                "expected_tool_calls": expected_tool_calls,
                "category": item_metadata.get("category", ""),
                "description": item_metadata.get("description", ""),
                "interface": "run_experiment",
                "environment": environment,
                "tools_available_count": len(tools),
                "tools_available": [t.name for t in tools],
            },
            tags=[experiment_name, environment],
        )

        try:
            run_config: Dict[str, Any] = {"configurable": {"langfuse_span": trace}}
            result = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "intent": None,
                    "complexity": None,
                    "tool_results": {},
                    "execution_plan": [],
                    "tool_selection_log": None,
                    "execution_plan_log": [],
                },
                config=run_config,  # type: ignore[arg-type]
            )
            messages = result["messages"]

            actual_tool_calls = extract_tool_calls_from_messages(messages)
            agent_response = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    agent_response = str(msg.content)
                    break

            intent: str = result.get("intent") or "GENERIC"
            complexity: str = result.get("complexity") or "SINGLE_TOOL"
            execution_plan_log: List[Any] = list(result.get("execution_plan_log") or [])
            tools_called: List[str] = []
            for entry in execution_plan_log:
                if entry.get("step") in ("execute_tools", "execute_sequential_tools"):
                    tools_called.extend(entry.get("tools_executed", []))

            trace.update(
                output={"tool_calls": actual_tool_calls, "response": agent_response},
                metadata={
                    "intent": intent,
                    "complexity": complexity,
                    "tools_called": tools_called,
                    "tools_called_count": len(tools_called),
                    "actual_tool_call_count": len(actual_tool_calls),
                    "expected_tool_call_count": len(expected_tool_calls),
                    "response_length": len(agent_response),
                },
            )

            logger.info(
                f"  ✓ intent={intent} complexity={complexity} "
                f"tool_calls={len(actual_tool_calls)} (expected {len(expected_tool_calls)})"
            )
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            trace.update(output={"error": str(e)})

    client.flush()

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Captured traces for {success_count}/{len(items)} items")
    logger.info(f"Experiment tag: {experiment_name}")
    logger.info(f"{'='*60}")
    logger.info("")
    logger.info("Next steps — evaluate the captured traces:")
    logger.info(f"  python -m evaluation.eval_ragas    --tag \"{experiment_name}\"")
    logger.info(f"  python -m evaluation.eval_deepeval --tag \"{experiment_name}\"")


def run_experiment(
    dataset_name: str,
    experiment_name: str,
    user: str = "eval-tester",
    session: Optional[str] = None,
    environment: str = "eval",
):
    """Run evaluation experiment (sync wrapper)."""
    import asyncio
    asyncio.run(run_experiment_async(
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        user=user,
        session=session,
        environment=environment,
    ))


def main():
    """Main function for evaluation experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Run agent against a Langfuse dataset and capture traces. "
            "Use eval_ragas / eval_deepeval afterwards to score the traces."
        )
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name in Langfuse")
    parser.add_argument(
        "--experiment-name", required=True,
        help="Experiment name — used as the trace tag for later evaluation filtering",
    )
    parser.add_argument(
        "--user",
        default="eval-tester",
        help="Langfuse user_id set on every trace (default: eval-tester)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Langfuse session_id shared across all traces in this run. "
            "Defaults to current UTC timestamp (YYYYMMDDTHHMMSSZ) so each run "
            "gets its own session automatically."
        ),
    )
    parser.add_argument(
        "--environment",
        default="eval",
        help="Environment label stored in trace metadata (default: eval)",
    )

    args = parser.parse_args()
    run_experiment(
        dataset_name=args.dataset_name,
        experiment_name=args.experiment_name,
        user=args.user,
        session=args.session,
        environment=args.environment,
    )


if __name__ == "__main__":
    main()
