"""eval_ragas.py — Evaluate LLM agent traces using RAGAS tool-call metrics.

Fetches agent traces from Langfuse (via evaluation/utils.py), formats them into
the RAGAS MultiTurnSample schema, and scores them with three dedicated functions:

  - evaluate_tool_call_accuracy_strict()   — exact order + name match
  - evaluate_tool_call_accuracy_flexible() — name match, order-insensitive
  - evaluate_tool_call_f1()                — F1 score on tool names

Run end-to-end:
    python -m evaluation.eval_ragas
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langfuse.client import Langfuse
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics._tool_call_accuracy import ToolCallAccuracy
from ragas.metrics._tool_call_f1 import ToolCallF1

from evaluation.utils import fetch_traces_from_langfuse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAGAS sample builder
# ---------------------------------------------------------------------------

def _build_ragas_sample(trace: Dict[str, Any]) -> MultiTurnSample:
    """Convert a single Langfuse trace dict into a RAGAS MultiTurnSample.

    Tool args are stripped so scoring is name-only — the agent generates SQL
    dynamically, so exact arg matching would always score 0.

    RAGAS MultiTurnSample schema:
      user_input:            list of message objects representing the conversation turn
        - HumanMessage       the user's question
        - AIMessage          the agent's response; tool_calls holds what was called
        - ToolMessage        placeholder result returned by the tool
        - AIMessage          the agent's final text answer
      reference_tool_calls:  list of ToolCall objects representing ground truth
    """
    actual_tool_calls = trace["actual_tool_calls"]    # [{"name": ..., "args": ...}, ...]
    expected_tool_calls = trace["expected_tool_calls"]  # ground truth from Langfuse metadata

    # Build the conversation turn as a message list
    messages: List[Any] = [HumanMessage(content=trace["user_input"])]

    # AIMessage carries the tool calls the agent actually made (args stripped to {})
    messages.append(
        AIMessage(
            content="",
            tool_calls=[ToolCall(name=tc["name"], args={}) for tc in actual_tool_calls],
        )
    )
    # ToolMessage is required by RAGAS to represent the tool result in the turn
    messages.append(ToolMessage(content="<tool-result>"))

    # Final AIMessage is the agent's text answer
    messages.append(AIMessage(content=trace["agent_response"]))

    # reference_tool_calls is the ground truth — also name-only (args stripped)
    reference_tool_calls = [
        ToolCall(name=tc["name"], args={}) for tc in expected_tool_calls
    ]

    return MultiTurnSample(user_input=messages, reference_tool_calls=reference_tool_calls)


# ---------------------------------------------------------------------------
# Metric evaluation functions — one per RAGAS metric
# ---------------------------------------------------------------------------

def evaluate_tool_call_accuracy_strict(samples: List[MultiTurnSample]) -> Tuple[float, List[float]]:
    """Score tool call accuracy in strict mode: both tool name AND order must match.

    Follows the RAGAS docs pattern — calls metric.multi_turn_ascore(sample) per trace.
    Uses ToolCallAccuracy with strict_order=True (the default).

    Args:
        samples: List of RAGAS MultiTurnSample objects, one per Langfuse trace.

    Returns:
        Tuple of (mean_score, per_sample_scores). per_sample_scores is aligned
        with the input list (one float per trace, 0.0 – 1.0).
    """
    # strict_order=True: order-sensitive exact name match
    metric = ToolCallAccuracy(strict_order=True)
    # Score each sample individually — exact pattern from RAGAS docs
    per_sample = [asyncio.run(metric.multi_turn_ascore(s)) for s in samples]
    mean_score = sum(per_sample) / len(per_sample) if per_sample else 0.0
    return mean_score, per_sample


def evaluate_tool_call_accuracy_flexible(samples: List[MultiTurnSample]) -> Tuple[float, List[float]]:
    """Score tool call accuracy in flexible mode: tool names must match, order ignored.

    Follows the RAGAS docs pattern — calls metric.multi_turn_ascore(sample) per trace.
    Uses ToolCallAccuracy with strict_order=False.

    Args:
        samples: List of RAGAS MultiTurnSample objects, one per Langfuse trace.

    Returns:
        Tuple of (mean_score, per_sample_scores). per_sample_scores is aligned
        with the input list (one float per trace, 0.0 – 1.0).
    """
    # strict_order=False: ignores call order — only checks that the right names are present
    metric = ToolCallAccuracy(strict_order=False)
    per_sample = [asyncio.run(metric.multi_turn_ascore(s)) for s in samples]
    mean_score = sum(per_sample) / len(per_sample) if per_sample else 0.0
    return mean_score, per_sample


def evaluate_tool_call_f1(samples: List[MultiTurnSample]) -> Tuple[float, List[float]]:
    """Score tool calls using F1: harmonic mean of precision and recall on tool names.

    Follows the RAGAS docs pattern — calls metric.multi_turn_ascore(sample) per trace.
    Uses ToolCallF1. Partial credit when the agent calls some but not all expected tools.

    Args:
        samples: List of RAGAS MultiTurnSample objects, one per Langfuse trace.

    Returns:
        Tuple of (mean_score, per_sample_scores). per_sample_scores is aligned
        with the input list (one float per trace, 0.0 – 1.0).
    """
    metric = ToolCallF1()
    per_sample = [asyncio.run(metric.multi_turn_ascore(s)) for s in samples]
    mean_score = sum(per_sample) / len(per_sample) if per_sample else 0.0
    return mean_score, per_sample


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _write_scores_to_langfuse(
    traces: List[Dict[str, Any]],
    strict_scores: List[float],
    flexible_scores: List[float],
    f1_scores: List[float],
) -> None:
    """Write per-trace RAGAS scores back to Langfuse so they appear in the portal.

    Each trace gets three scores written:
      - ragas_tool_call_accuracy_strict
      - ragas_tool_call_accuracy_flexible
      - ragas_tool_call_f1
    """
    import httpx
    from agent_eval.config import get_langfuse_config

    config = get_langfuse_config()
    httpx_client = httpx.Client(verify=config.ssl_verify)
    client = Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        host=config.host,
        httpx_client=httpx_client,
    )

    for i, trace in enumerate(traces):
        trace_id = trace["trace_id"]
        try:
            client.score(trace_id=trace_id, name="ragas_tool_call_accuracy_strict",   value=strict_scores[i])
            client.score(trace_id=trace_id, name="ragas_tool_call_accuracy_flexible", value=flexible_scores[i])
            client.score(trace_id=trace_id, name="ragas_tool_call_f1",                value=f1_scores[i])
            logger.info(
                f"  Scored trace {trace_id}: "
                f"strict={strict_scores[i]:.2f}  flexible={flexible_scores[i]:.2f}  f1={f1_scores[i]:.2f}"
            )
        except Exception as e:
            logger.warning(f"  Failed to write scores for trace {trace_id}: {e}")

    client.flush()


def orchestrate_evaluation(tag: Optional[str] = None, limit: int = 20) -> None:
    """Fetch traces from Langfuse, run all three RAGAS metrics, write scores, and print results.

    Steps:
      1. Fetch and format agent traces from Langfuse into RAGAS-ready samples.
      2. Build a single EvaluationDataset shared across all metric functions.
      3. Call each metric function sequentially and collect per-trace + mean scores.
      4. Write per-trace scores back to Langfuse (visible in the portal).
      5. Print a consolidated results summary.

    Args:
        tag:   Langfuse tag to filter traces (e.g. "eval-run-1"). None = most recent.
        limit: Max number of traces to fetch (default: 20).
    """
    # Step 1 — fetch traces from Langfuse and format into RAGAS input schema
    logger.info("Fetching traces from Langfuse…")
    traces = fetch_traces_from_langfuse(tag=tag, limit=limit)

    if not traces:
        logger.warning("No scoreable traces found — nothing to evaluate.")
        return

    logger.info(f"Building RAGAS samples from {len(traces)} trace(s)…")

    # Step 2 — build one MultiTurnSample per trace (shared list reused by all three metrics)
    samples = [_build_ragas_sample(t) for t in traces]

    # Step 3 — run each metric function sequentially; each scores sample-by-sample per RAGAS docs
    logger.info("Running evaluate_tool_call_accuracy_strict…")
    score_strict, strict_per_trace = evaluate_tool_call_accuracy_strict(samples)

    logger.info("Running evaluate_tool_call_accuracy_flexible…")
    score_flexible, flexible_per_trace = evaluate_tool_call_accuracy_flexible(samples)

    logger.info("Running evaluate_tool_call_f1…")
    score_f1, f1_per_trace = evaluate_tool_call_f1(samples)

    # Step 4 — write per-trace scores back to Langfuse so they appear in the portal
    logger.info("Writing per-trace scores to Langfuse…")
    _write_scores_to_langfuse(traces, strict_per_trace, flexible_per_trace, f1_per_trace)

    # Step 5 — print consolidated results summary
    print("\n" + "=" * 60)
    print("  RAGAS Tool Call Evaluation Results")
    print("=" * 60)
    print(f"  Traces evaluated           : {len(traces)}")
    print(f"  tool_call_accuracy_strict  : {score_strict:.4f}")
    print(f"  tool_call_accuracy_flexible: {score_flexible:.4f}")
    print(f"  tool_call_f1               : {score_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    orchestrate_evaluation()


