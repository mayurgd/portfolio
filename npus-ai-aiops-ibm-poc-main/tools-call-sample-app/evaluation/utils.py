"""
Shared utilities for evaluation scripts.

Provides:
  - call_with_retry: retry helper with exponential backoff for Langfuse API calls
    that may hit rate limits (HTTP 429).
  - fetch_traces_from_langfuse: fetches agent traces from Langfuse and formats
    them into a framework-agnostic list of trace dicts ready for evaluation.
    Used by both eval_ragas.py and eval_deepeval.py.
"""

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default inter-call pause inserted between successive Langfuse API calls
# to stay well below the rate limit.  Can be overridden per call-site.
DEFAULT_INTER_CALL_DELAY: float = 1.0  # seconds

_RATE_LIMIT_STATUS = 429


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception represents a Langfuse 429 rate-limit error."""
    # langfuse.api.core.api_error.ApiError carries status_code
    status = getattr(exc, "status_code", None)
    if status == _RATE_LIMIT_STATUS:
        logger.debug(
            f"[rate-limit-check] 429 detected via status_code attribute | "
            f"type={type(exc).__name__} status_code={status}"
        )
        return True
    # Fallback: check string representation (e.g. "status_code: 429")
    msg = str(exc)
    matched = "429" in msg and ("rate limit" in msg.lower() or "status_code" in msg.lower())
    if matched:
        logger.debug(
            f"[rate-limit-check] 429 detected via string match | "
            f"type={type(exc).__name__} msg={msg[:200]}"
        )
    else:
        logger.debug(
            f"[rate-limit-check] NOT a rate-limit error | "
            f"type={type(exc).__name__} status_code={status!r} msg={msg[:200]}"
        )
    return matched


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 2,
    base_delay: float = 10.0,
    max_delay: float = 120.0,
    inter_call_delay: float = DEFAULT_INTER_CALL_DELAY,
    label: str = "Langfuse API call",
) -> T:
    """Call ``fn`` and retry on 429 rate-limit errors with exponential backoff.

    A short ``inter_call_delay`` pause is inserted *before* every call
    (including the first) to reduce the chance of hitting the rate limit at all.

    Args:
        fn:                Zero-argument callable that performs the API call.
        max_retries:       Maximum number of retry attempts after the first failure.
        base_delay:        Initial wait time in seconds after a 429 (doubles each retry).
        max_delay:         Cap on the wait time between retries.
        inter_call_delay:  Seconds to sleep before every call to pace requests.
                           Set to 0 to disable.
        label:             Human-readable description used in log messages.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        The last exception if all retries are exhausted, or immediately for
        non-rate-limit errors.
    """
    attempt = 0
    delay = base_delay

    while True:
        # Pace requests to avoid hitting the rate limit in the first place
        if inter_call_delay > 0:
            time.sleep(inter_call_delay)

        try:
            return fn()
        except Exception as exc:
            logger.debug(
                f"[call_with_retry] exception caught | label={label!r} "
                f"type={type(exc).__qualname__} "
                f"status_code={getattr(exc, 'status_code', 'N/A')!r} "
                f"msg={str(exc)[:300]}"
            )
            if not _is_rate_limit_error(exc):
                raise  # Non-rate-limit errors propagate immediately

            attempt += 1
            if attempt > max_retries:
                logger.error(
                    f"  ✗ {label} — rate limit exceeded after {max_retries} retries: {exc}"
                )
                raise

            wait = min(delay, max_delay)
            logger.info(
                f"  ⏳ {label} — rate limit hit (429), retrying in {wait:.1f}s "
                f"(attempt {attempt}/{max_retries})…"
            )
            time.sleep(wait)
            delay = min(delay * 2, max_delay)




# ---------------------------------------------------------------------------
# Langfuse trace fetching — shared by eval_ragas.py and eval_deepeval.py
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_observations(observations: List[Any]) -> List[Dict[str, Any]]:
    """Extract actual tool calls from Langfuse trace observations.

    Tries observation types in priority order to avoid double-counting:
      1. GENERATION output with ``tool_calls`` (most complete, has args)
      2. SPAN named ``tool:<name>``
      3. SPAN named ``tool_<name>`` (legacy)
      4. SPAN with ``metadata["tool_name"]``
      5. SPAN ``execute_tools`` input list (last resort)
    """
    # Priority 1: GENERATION with tool_calls in output
    from_generation: List[Dict[str, Any]] = []
    for obs in observations:
        obs_type = getattr(obs, "type", "") or ""
        obs_output = getattr(obs, "output", None)
        if obs_type == "GENERATION" and isinstance(obs_output, dict) and "tool_calls" in obs_output:
            for tc in obs_output["tool_calls"]:
                from_generation.append({"name": tc.get("name", ""), "args": tc.get("args", {})})
    if from_generation:
        return from_generation

    # Priority 2: SPAN named "tool:<name>"
    from_tool_spans: List[Dict[str, Any]] = []
    for obs in observations:
        obs_type = getattr(obs, "type", "") or ""
        obs_name = getattr(obs, "name", "") or ""
        obs_input = getattr(obs, "input", None)
        if obs_type == "SPAN" and obs_name.startswith("tool:"):
            tool_name = obs_name[len("tool:"):]
            from_tool_spans.append({"name": tool_name, "args": obs_input if isinstance(obs_input, dict) else {}})
    if from_tool_spans:
        return from_tool_spans

    # Priority 3: SPAN named "tool_<name>" (legacy)
    from_legacy_spans: List[Dict[str, Any]] = []
    for obs in observations:
        obs_type = getattr(obs, "type", "") or ""
        obs_name = getattr(obs, "name", "") or ""
        obs_input = getattr(obs, "input", None)
        if obs_type == "SPAN" and obs_name.startswith("tool_") and obs_name not in ("tool_span", "tool_executor"):
            tool_name = obs_name[len("tool_"):]
            from_legacy_spans.append({"name": tool_name, "args": obs_input if isinstance(obs_input, dict) else {}})
    if from_legacy_spans:
        return from_legacy_spans

    # Priority 4: SPAN with metadata["tool_name"]
    from_metadata: List[Dict[str, Any]] = []
    for obs in observations:
        obs_type = getattr(obs, "type", "") or ""
        obs_input = getattr(obs, "input", None)
        obs_metadata = getattr(obs, "metadata", None) or {}
        if obs_type == "SPAN" and isinstance(obs_metadata, dict) and obs_metadata.get("tool_name"):
            from_metadata.append({"name": obs_metadata["tool_name"], "args": obs_input if isinstance(obs_input, dict) else {}})
    if from_metadata:
        return from_metadata

    # Priority 5: execute_tools SPAN input list (last resort)
    for obs in observations:
        obs_type = getattr(obs, "type", "") or ""
        obs_name = getattr(obs, "name", "") or ""
        obs_input = getattr(obs, "input", None)
        if obs_type == "SPAN" and obs_name == "execute_tools" and isinstance(obs_input, list):
            result = [{"name": item["name"], "args": item.get("args", {})} for item in obs_input if isinstance(item, dict) and "name" in item]
            if result:
                return result

    return []


def _get_expected_tool_calls(client: Any, trace: Any) -> Optional[List[Dict[str, Any]]]:
    """Get ground-truth tool calls for a trace. Returns None if not found (trace is skipped).

    Sources checked in order:
      1. ``trace.metadata["expected_tool_calls"]`` — set by run_experiment.py and app.py
      2. Langfuse dataset lookup by ``dataset_item_id`` in trace metadata
    """
    metadata = trace.metadata or {}

    if "expected_tool_calls" in metadata:
        return metadata["expected_tool_calls"]

    if "dataset_item_id" in metadata:
        dataset_item_id = metadata["dataset_item_id"]
        experiment = metadata.get("experiment", "")
        try:
            dataset_names = ["tool-call-eval"] if experiment else []
            for dataset_name in dataset_names:
                try:
                    dataset = client.get_dataset(dataset_name)
                    for item in dataset.items:
                        if item.id == dataset_item_id:
                            expected_output = item.expected_output or {}
                            if "tool_calls" in expected_output:
                                return expected_output["tool_calls"]
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Failed to lookup dataset item {dataset_item_id}: {e}")

    return None


def fetch_traces_from_langfuse(
    tag: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Fetch agent traces from Langfuse and format them for evaluation frameworks.

    Connects to Langfuse using environment variables:
      LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

    For each trace, extracts:
      - user_input:           the human question sent to the agent
      - agent_response:       the agent's final text response
      - actual_tool_calls:    list of {"name": str, "args": dict} dicts
      - expected_tool_calls:  list of {"name": str, "args": dict} dicts (ground truth)

    Traces without expected_tool_calls or with empty expected lists are skipped
    (no-tool-needed queries cannot be scored for tool call accuracy).

    Args:
        tag:   Langfuse tag to filter traces by (e.g. "eval-run-1").
               Pass None to fetch the most recent traces up to ``limit``.
        limit: Maximum number of traces to retrieve (default: 20).

    Returns:
        List of trace dicts ready to be passed into RAGAS or DeepEval metric functions.
        Each dict has keys: trace_id, user_input, agent_response,
        actual_tool_calls, expected_tool_calls.
    """
    # Import here to avoid hard dependency when utils is imported without Langfuse installed
    import httpx
    from langfuse.client import Langfuse
    from agent_eval.config import get_langfuse_config

    config = get_langfuse_config()

    if not config.public_key or not config.secret_key:
        raise EnvironmentError(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in environment or .env file"
        )

    httpx_client = httpx.Client(verify=config.ssl_verify)
    client = Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        host=config.host,
        httpx_client=httpx_client,
    )

    # Fetch traces — optionally filtered by tag
    fetch_kwargs: Dict[str, Any] = {"limit": limit}
    if tag:
        fetch_kwargs["tags"] = [tag]

    try:
        result = call_with_retry(lambda: client.fetch_traces(**fetch_kwargs), label="fetch_traces")
        raw_traces = result.data
    except Exception as e:
        logger.error(f"Failed to fetch traces from Langfuse: {e}")
        return []

    if not raw_traces:
        logger.warning("No traces returned from Langfuse")
        return []

    logger.info(f"Fetched {len(raw_traces)} trace(s) from Langfuse")

    formatted: List[Dict[str, Any]] = []

    for trace in raw_traces:
        trace_id = trace.id

        # Fetch observations (spans/generations) for this trace to extract tool calls
        try:
            obs_response = call_with_retry(
                lambda tid=trace_id: client.fetch_observations(trace_id=tid),
                label=f"fetch_observations({trace_id})",
                inter_call_delay=0,
            )
            observations = obs_response.data
        except Exception as e:
            logger.warning(f"Skipping trace {trace_id} — could not fetch observations: {e}")
            continue

        # Extract the sequence of tool calls the agent actually made
        actual_tool_calls = _extract_tool_calls_from_observations(observations)

        if not actual_tool_calls:
            logger.debug(f"Skipping trace {trace_id} — no tool calls found in observations")
            continue

        # Get ground-truth tool calls from trace metadata (set by run_experiment.py / app.py)
        expected_tool_calls = _get_expected_tool_calls(client, trace)

        if expected_tool_calls is None:
            logger.debug(f"Skipping trace {trace_id} — no expected_tool_calls in metadata")
            continue

        if len(expected_tool_calls) == 0:
            logger.debug(f"Skipping trace {trace_id} — expected_tool_calls is empty (no-tool query)")
            continue

        # Extract the human question and agent's final answer from the trace
        user_input = trace.input if isinstance(trace.input, str) else str(trace.input or "")
        agent_response = trace.output if isinstance(trace.output, str) else str(trace.output or "")

        formatted.append({
            "trace_id": trace_id,
            "user_input": user_input,
            "agent_response": agent_response,
            # Actual tool calls: [{"name": "query_customers", "args": {...}}, ...]
            "actual_tool_calls": actual_tool_calls,
            # Expected tool calls: ground truth from trace metadata
            "expected_tool_calls": expected_tool_calls,
        })

    logger.info(f"Formatted {len(formatted)} trace(s) with valid tool call data for evaluation")
    return formatted
