"""eval_deepeval.py — Evaluate LLM agent traces using DeepEval tool-call metrics.

Fetches agent traces from Langfuse (via evaluation/utils.py), formats them into
DeepEval test case objects, and scores them with three dedicated functions:

  - evaluate_tool_correctness()     — ToolCorrectnessMetric: were the right tools called?
  - evaluate_tool_use()             — ToolUseMetric: did the agent use tools appropriately?
  - evaluate_argument_correctness() — ArgumentCorrectnessMetric: were args relevant?

All three metrics are LLM-judged (DeepEval calls the LLM internally for scoring).
LLM judge priority:
  1. NESGEN (default) — uses NESTLE_CLIENT_ID + NESTLE_CLIENT_SECRET
  2. OpenAI (fallback) — uses OPENAI_API_KEY
  3. None — all metrics are skipped gracefully; no scores written to Langfuse

Run end-to-end:
    python -m evaluation.eval_deepeval
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re
import requests
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric, ToolUseMetric
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn
from dotenv import load_dotenv
from langfuse.client import Langfuse

from evaluation.utils import fetch_traces_from_langfuse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NESGEN judge — wraps the NESGEN chat completions API as a DeepEval LLM judge
# ---------------------------------------------------------------------------

class _NESGENJudge(DeepEvalBaseLLM):
    """DeepEval-compatible LLM judge that calls the NESGEN API.

    Mirrors the request logic of NesGenChatModel._make_request():
      - If NESGEN_URL is set → uses the responses endpoint (POST with {"model", "input"})
      - Otherwise           → builds chat completions URL from NESGEN_API_BASE + model

    NESGEN uses client_id / client_secret headers instead of a Bearer token,
    so it cannot be wrapped with GPTModel.
    """

    def __init__(self) -> None:
        self._client_id = os.getenv("NESTLE_CLIENT_ID", "")
        self._client_secret = os.getenv("NESTLE_CLIENT_SECRET", "")
        # NESGEN_URL: direct responses endpoint (takes priority when set)
        self._model_endpoint = os.getenv("NESGEN_URL", "")
        # NESGEN_API_BASE: base for chat completions fallback
        # URL pattern: {api_base}/openai/deployments/{model}/chat/completions
        self._api_base = os.getenv(
            "NESGEN_API_BASE",
            "https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1",
        ).rstrip("/")
        self._model = os.getenv("NESGEN_MODEL", "gpt-4o-mini")
        self._api_version = os.getenv("NESGEN_API_VERSION", "2025-04-01-preview")
        super().__init__()

    def get_model_name(self) -> str:
        return f"nesgen/{self._model}"

    def load_model(self) -> "DeepEvalBaseLLM":
        # No persistent client to load — each call uses requests directly
        return self

    def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        """Send a single-turn prompt to NESGEN and return the text response.

        When DeepEval passes a `schema` kwarg (a Pydantic model class), the prompt
        is augmented with explicit JSON formatting instructions so NESGEN returns
        a valid JSON object matching the schema's fields. DeepEval then parses
        the returned string as JSON.

        Uses NESGEN_URL (responses endpoint) if set, otherwise falls back to
        the chat completions endpoint built from NESGEN_API_BASE + model name.
        Mirrors NesGenChatModel._make_request() exactly.
        """
        # When a schema is provided, append JSON-only instructions to the prompt
        if schema is not None:
            try:
                # Extract field names from the Pydantic schema to guide the model
                schema_fields = list(schema.model_fields.keys()) if hasattr(schema, "model_fields") else []
                fields_hint = ", ".join(f'"{f}"' for f in schema_fields) if schema_fields else "the required fields"
                # Pre-compute example string outside f-string (backslashes not allowed in f-string expressions in Python <3.12)
                example_pairs = ", ".join('"' + f + '": <value>' for f in schema_fields)
                example_str = "{" + example_pairs + "}"
                json_instruction = (
                    f"\n\nIMPORTANT: You MUST respond with ONLY a valid JSON object. "
                    f"No prose, no markdown, no code fences. "
                    f"The JSON must contain exactly these fields: {fields_hint}. "
                    f"Example format: {example_str}"
                )
                prompt = prompt + json_instruction
            except Exception:
                # If schema introspection fails, just add a generic JSON instruction
                prompt = prompt + "\n\nIMPORTANT: Respond with ONLY a valid JSON object. No prose or markdown."

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }

        if self._model_endpoint:
            url = self._model_endpoint
            lower_url = url.lower()
            if "/completions" in lower_url:
                # Chat completions-style endpoint — send messages payload
                payload: Dict[str, Any] = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
            else:
                # Responses endpoint — send input payload
                payload = {
                    "model": self._model,
                    "input": [{"role": "user", "content": prompt}],
                }
            response = requests.post(url, headers=headers, json=payload, timeout=120)
        else:
            # Chat completions endpoint: {api_base}/openai/deployments/{model}/chat/completions
            url = f"{self._api_base}/openai/deployments/{self._model}/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }
            response = requests.post(
                url,
                params={"api-version": self._api_version},
                headers=headers,
                json=payload,
                timeout=120,
            )

        response.raise_for_status()
        data = response.json()

        # Extract text — handle both chat completions and responses endpoint shapes
        # Chat completions: data["choices"][0]["message"]["content"]
        # Responses endpoint: data["output"][0]["content"][0]["text"]
        raw_text = ""
        try:
            raw_text = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            pass
        if not raw_text:
            try:
                raw_text = str(data["output"][0]["content"][0]["text"])
            except (KeyError, IndexError, TypeError):
                pass
        if not raw_text:
            raise ValueError(f"Cannot extract text from NESGEN response: {list(data.keys())}")

        # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped)
            stripped = stripped.strip()

        return stripped

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> str:
        """Async generate — delegates to sync implementation."""
        return self.generate(prompt, schema=schema, **kwargs)


def _get_llm_judge() -> Optional[DeepEvalBaseLLM]:
    """Return the best available LLM judge for ArgumentCorrectnessMetric.

    Priority:
      1. NESGEN (default) — requires NESTLE_CLIENT_ID + NESTLE_CLIENT_SECRET
      2. OpenAI (fallback) — requires OPENAI_API_KEY
      3. None — ArgumentCorrectnessMetric will be skipped gracefully

    Returns a DeepEvalBaseLLM-compatible instance, or None if nothing is configured.
    """
    # --- 1. Try NESGEN first (default judge) ---
    client_id = os.getenv("NESTLE_CLIENT_ID", "")
    client_secret = os.getenv("NESTLE_CLIENT_SECRET", "")
    if client_id and client_secret:
        logger.info("LLM judge: NESGEN (%s)", os.getenv("NESGEN_MODEL", "gpt-4o-mini"))
        return _NESGENJudge()

    # --- 2. Fall back to OpenAI ---
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        logger.info("LLM judge: OpenAI (%s)", model_name)
        kwargs: Dict[str, Any] = {"model": model_name, "api_key": openai_key}
        if base_url:
            kwargs["base_url"] = base_url
        return GPTModel(**kwargs)

    # --- 3. No judge available ---
    return None


# ---------------------------------------------------------------------------
# DeepEval test case builders
# ---------------------------------------------------------------------------

def _build_llm_test_case(trace: Dict[str, Any]) -> LLMTestCase:
    """Convert a single Langfuse trace dict into a DeepEval LLMTestCase.

    LLMTestCase schema used by ToolCorrectnessMetric and ArgumentCorrectnessMetric:
      input:          the user's question sent to the agent
      actual_output:  the agent's final text response
      tools_called:   list of ToolCall objects the agent actually invoked
      expected_tools: list of ToolCall objects representing ground truth (name-only)
    """
    # actual tool calls the agent made — args preserved for ArgumentCorrectnessMetric
    tools_called = [
        ToolCall(name=tc["name"], input_parameters=tc.get("args", {}))
        for tc in trace["actual_tool_calls"]
    ]

    # expected tools from ground truth — name-only (args stripped) for name-based scoring
    expected_tools = [
        ToolCall(name=tc["name"], input_parameters={})
        for tc in trace["expected_tool_calls"]
    ]

    return LLMTestCase(
        input=trace["user_input"],
        actual_output=trace["agent_response"],
        tools_called=tools_called,
        expected_tools=expected_tools,
    )


def _build_conversational_test_case(trace: Dict[str, Any]) -> ConversationalTestCase:
    """Convert a single Langfuse trace dict into a DeepEval ConversationalTestCase.

    Required by ToolUseMetric, which checks whether the agent used tools from the
    available catalogue. The metric is non-LLM — pure name-based comparison.

    ConversationalTestCase schema:
      turns: list of Turn objects representing the conversation
        - Turn(role="user", content=..., tools_called=[...])
    """
    # Single turn: the user message with the tool calls the agent made
    turn = Turn(
        role="user",
        content=trace["user_input"],
        tools_called=[
            ToolCall(name=tc["name"], input_parameters=tc.get("args", {}))
            for tc in trace["actual_tool_calls"]
        ],
    )
    return ConversationalTestCase(turns=[turn])


# ---------------------------------------------------------------------------
# Metric evaluation functions — one per DeepEval metric
# ---------------------------------------------------------------------------

def evaluate_tool_correctness(
    test_cases: List[LLMTestCase],
    llm_judge: Optional[DeepEvalBaseLLM],
) -> Tuple[Optional[float], List[Optional[float]], List[str]]:
    """Score whether the agent called the correct tools (name match, order not considered).

    Uses DeepEval ToolCorrectnessMetric — LLM-judged (DeepEval calls the model internally).
    Skipped gracefully (returns None scores) when no LLM judge is available.

    Args:
        test_cases: List of LLMTestCase objects, one per Langfuse trace.
        llm_judge:  DeepEvalBaseLLM instance (NESGEN or OpenAI), or None to skip.

    Returns:
        Tuple of (mean_score, per_sample_scores, per_sample_reasons).
        Scores are None when skipped.
    """
    if llm_judge is None:
        logger.warning("  Skipping ToolCorrectnessMetric — no LLM judge configured")
        n = len(test_cases)
        return None, [None] * n, ["skipped: no LLM judge configured"] * n

    # should_consider_ordering=False: order-insensitive name match
    metric = ToolCorrectnessMetric(
        model=llm_judge,
        include_reason=True,
        async_mode=False,
        should_consider_ordering=False,
    )

    per_scores: List[Optional[float]] = []
    per_reasons: List[str] = []

    for tc in test_cases:
        try:
            metric.measure(tc, _show_indicator=False)
            per_scores.append(float(metric.score or 0.0))
            per_reasons.append(str(metric.reason))
        except Exception as e:
            logger.warning(f"  ToolCorrectnessMetric failed for a sample: {e}")
            per_scores.append(None)
            per_reasons.append(f"error: {e}")

    valid = [s for s in per_scores if s is not None]
    mean_score: Optional[float] = sum(valid) / len(valid) if valid else None
    return mean_score, per_scores, per_reasons


def evaluate_tool_use(
    conv_test_cases: List[ConversationalTestCase],
    available_tool_names: List[str],
    llm_judge: Optional[DeepEvalBaseLLM],
) -> Tuple[Optional[float], List[Optional[float]], List[str]]:
    """Score whether the agent used tools from the available catalogue appropriately.

    Uses DeepEval ToolUseMetric — LLM-judged (DeepEval calls the model internally for
    both tool selection and argument correctness scoring).
    Skipped gracefully (returns None scores) when no LLM judge is available.

    Args:
        conv_test_cases:      List of ConversationalTestCase objects, one per trace.
        available_tool_names: Names of all tools the agent had access to.
        llm_judge:            DeepEvalBaseLLM instance (NESGEN or OpenAI), or None to skip.

    Returns:
        Tuple of (mean_score, per_sample_scores, per_sample_reasons).
        Scores are None when skipped.
    """
    if llm_judge is None:
        logger.warning("  Skipping ToolUseMetric — no LLM judge configured")
        n = len(conv_test_cases)
        return None, [None] * n, ["skipped: no LLM judge configured"] * n

    # Build ToolCall objects for the available tool catalogue (name-only, no args)
    available_tools = [ToolCall(name=name, input_parameters={}) for name in available_tool_names]

    # Pass the judge explicitly — ToolUseMetric calls the LLM for scoring
    metric = ToolUseMetric(
        available_tools=available_tools,
        model=llm_judge,
        include_reason=True,
        async_mode=False,
    )

    per_scores: List[Optional[float]] = []
    per_reasons: List[str] = []

    for tc in conv_test_cases:
        try:
            metric.measure(tc, _show_indicator=False)
            per_scores.append(float(metric.score or 0.0))
            per_reasons.append(str(metric.reason))
        except Exception as e:
            logger.warning(f"  ToolUseMetric failed for a sample: {e}")
            per_scores.append(None)
            per_reasons.append(f"error: {e}")

    valid = [s for s in per_scores if s is not None]
    mean_score: Optional[float] = sum(valid) / len(valid) if valid else None
    return mean_score, per_scores, per_reasons


def evaluate_argument_correctness(
    test_cases: List[LLMTestCase],
    llm_judge: Optional[DeepEvalBaseLLM],
) -> Tuple[float, List[float], List[str]]:
    """Score whether the tool call arguments were relevant to the user's input.

    Uses DeepEval ArgumentCorrectnessMetric — LLM-judged, requires OPENAI_API_KEY.
    Does NOT compare against reference SQL; it validates argument relevance to the query.
    Returns all-zero scores with a skip notice if no LLM judge is available.

    Args:
        test_cases: List of LLMTestCase objects, one per Langfuse trace.
        llm_judge:  DeepEvalBaseLLM instance to use as the judge, or None to skip.

    Returns:
        Tuple of (mean_score, per_sample_scores, per_sample_reasons).
    """
    if llm_judge is None:
        logger.warning("  Skipping ArgumentCorrectnessMetric — no LLM judge configured")
        n = len(test_cases)
        return 0.0, [0.0] * n, ["skipped: no LLM judge configured"] * n

    metric = ArgumentCorrectnessMetric(
        model=llm_judge,
        include_reason=True,
        async_mode=False,
    )

    per_scores: List[float] = []
    per_reasons: List[str] = []

    for tc in test_cases:
        try:
            metric.measure(tc, _show_indicator=False)
            per_scores.append(float(metric.score or 0.0))
            per_reasons.append(str(metric.reason))
        except Exception as e:
            logger.warning(f"  ArgumentCorrectnessMetric failed for a sample: {e}")
            per_scores.append(0.0)
            per_reasons.append(f"error: {e}")

    mean_score = sum(per_scores) / len(per_scores) if per_scores else 0.0
    return mean_score, per_scores, per_reasons


# ---------------------------------------------------------------------------
# Langfuse score write-back
# ---------------------------------------------------------------------------

def _write_scores_to_langfuse(
    traces: List[Dict[str, Any]],
    correctness_scores: Sequence[Optional[float]],
    tool_use_scores: Sequence[Optional[float]],
    argument_scores: Sequence[Optional[float]],
) -> None:
    """Write per-trace DeepEval scores back to Langfuse so they appear in the portal.

    Only writes a score when it is not None — skipped metrics produce no Langfuse entry
    so they don't appear as misleading 0.0 values in the portal.

    Scores written (when available):
      - deepeval_tool_correctness
      - deepeval_tool_use
      - deepeval_argument_correctness
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
            # Only write scores that are not None — None means the metric was skipped
            written: Dict[str, float] = {}
            if correctness_scores[i] is not None:
                v = float(correctness_scores[i])  # type: ignore[arg-type]
                client.score(trace_id=trace_id, name="deepeval_tool_correctness", value=v)
                written["correctness"] = v
            if tool_use_scores[i] is not None:
                v = float(tool_use_scores[i])  # type: ignore[arg-type]
                client.score(trace_id=trace_id, name="deepeval_tool_use", value=v)
                written["tool_use"] = v
            if argument_scores[i] is not None:
                v = float(argument_scores[i])  # type: ignore[arg-type]
                client.score(trace_id=trace_id, name="deepeval_argument_correctness", value=v)
                written["arg_correctness"] = v

            if written:
                parts = "  ".join(f"{k}={v:.2f}" for k, v in written.items())
                logger.info(f"  Scored trace {trace_id}: {parts}")
            else:
                logger.info(f"  No scores written for trace {trace_id} (all metrics skipped)")
        except Exception as e:
            logger.warning(f"  Failed to write scores for trace {trace_id}: {e}")

    client.flush()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def orchestrate_evaluation(tag: Optional[str] = None, limit: int = 20) -> None:
    """Fetch traces from Langfuse, run all three DeepEval metrics, write scores, and print results.

    Steps:
      1. Fetch and format agent traces from Langfuse into framework-agnostic dicts.
      2. Build LLMTestCase and ConversationalTestCase objects for each trace.
      3. Derive the available tool catalogue from the union of all expected tool names.
      4. Call each metric function sequentially and collect per-trace + mean scores.
      5. Write per-trace scores back to Langfuse (visible in the portal).
      6. Print a consolidated results summary.

    Args:
        tag:   Langfuse tag to filter traces (e.g. "eval-run-1"). None = most recent.
        limit: Max number of traces to fetch (default: 20).
    """
    # Step 1 — fetch traces from Langfuse and format into framework-agnostic dicts
    logger.info("Fetching traces from Langfuse…")
    traces = fetch_traces_from_langfuse(tag=tag, limit=limit)

    if not traces:
        logger.warning("No scoreable traces found — nothing to evaluate.")
        return

    logger.info(f"Building DeepEval test cases from {len(traces)} trace(s)…")

    # Step 2 — build one LLMTestCase and one ConversationalTestCase per trace
    llm_test_cases = [_build_llm_test_case(t) for t in traces]
    conv_test_cases = [_build_conversational_test_case(t) for t in traces]

    # Step 3 — derive available tool catalogue from all expected tool names across traces
    # ToolUseMetric checks that the agent only called tools from this set
    available_tool_names: List[str] = sorted({
        tc["name"]
        for t in traces
        for tc in t["expected_tool_calls"]
    })
    logger.info(f"Available tool catalogue ({len(available_tool_names)}): {available_tool_names}")

    # Step 4 — resolve LLM judge (all three metrics are LLM-judged)
    # Priority: NESGEN (default) → OpenAI (fallback) → None (skip all metrics)
    llm_judge = _get_llm_judge()
    if llm_judge:
        logger.info("LLM judge: %s — running all three metrics", llm_judge.get_model_name())
    else:
        logger.warning(
            "No LLM judge configured — all metrics will be skipped. "
            "Set NESTLE_CLIENT_ID/SECRET (NESGEN) or OPENAI_API_KEY (OpenAI)."
        )

    # Step 5 — run each metric function sequentially; each scores sample-by-sample
    logger.info("Running evaluate_tool_correctness…")
    score_correctness, correctness_per_trace, _ = evaluate_tool_correctness(llm_test_cases, llm_judge)

    logger.info("Running evaluate_tool_use…")
    score_tool_use, tool_use_per_trace, _ = evaluate_tool_use(conv_test_cases, available_tool_names, llm_judge)

    logger.info("Running evaluate_argument_correctness…")
    score_argument, argument_per_trace, _ = evaluate_argument_correctness(llm_test_cases, llm_judge)

    # Step 6 — write per-trace scores back to Langfuse (None scores are not written)
    logger.info("Writing per-trace scores to Langfuse…")
    _write_scores_to_langfuse(traces, correctness_per_trace, tool_use_per_trace, argument_per_trace)

    # Step 7 — print consolidated results summary
    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "skipped"

    print("\n" + "=" * 60)
    print("  DeepEval Tool Call Evaluation Results")
    print("=" * 60)
    print(f"  Traces evaluated           : {len(traces)}")
    print(f"  tool_correctness           : {_fmt(score_correctness)}")
    print(f"  tool_use                   : {_fmt(score_tool_use)}")
    print(f"  argument_correctness       : {_fmt(score_argument)}")
    print("=" * 60)


if __name__ == "__main__":
    orchestrate_evaluation()

# Made with Bob
