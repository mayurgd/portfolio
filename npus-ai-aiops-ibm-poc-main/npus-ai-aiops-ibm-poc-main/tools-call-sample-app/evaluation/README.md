# Evaluation Framework

End-to-end evaluation pipeline for the **Bakehouse Data Agent** — measures tool-call accuracy using [RAGAS](https://docs.ragas.io/) and [DeepEval](https://docs.confident-ai.com/), with scores written back to [Langfuse](https://langfuse.com/).

---

## How It Works

The pipeline has **three distinct phases**:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────────────┐
│  load_dataset   │────▶│  run_experiment  │────▶│  eval_ragas / eval_deepeval  │
│  (one-time)     │     │  (per experiment)│     │  (score the traces)          │
└─────────────────┘     └──────────────────┘     └──────────────────────────────┘
  Uploads dataset.json    Runs agent on each        Fetches traces by tag,
  to Langfuse as a        dataset item, creates     computes metrics, writes
  named dataset.          tagged Langfuse traces.   scores back to Langfuse.
```

**`run_experiment.py` does NOT score** — it only runs the agent and captures traces.  
**`eval_ragas.py` and `eval_deepeval.py` do the scoring** — they read those traces and write metric scores back.

---

## Files

| File | Purpose |
|------|---------|
| `dataset.json` | 42 test cases in portable JSON (source of truth for Langfuse upload) |
| `load_dataset.py` | Uploads `dataset.json` to Langfuse as a named dataset |
| `run_experiment.py` | Runs the agent on each dataset item; creates tagged Langfuse traces |
| `eval_ragas.py` | Scores existing traces with RAGAS metrics; writes scores to Langfuse |
| `eval_deepeval.py` | Scores existing traces with DeepEval metrics; writes scores to Langfuse |
| `utils.py` | Shared retry/helper utilities |

---

## Prerequisites

1. MCP server must be running before `run_experiment`:
   ```bash
   uv run python -m agent_eval.mcp_server.server
   ```

2. `.env` must have valid credentials:
   ```
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_SECRET_KEY=...
   LANGFUSE_HOST=...
   OPENAI_API_KEY=...   # or Nestle LLM credentials
   ```

---

## Step-by-Step Usage

### Step 1 — Load dataset (one-time setup)

Uploads `dataset.json` to Langfuse as a named dataset. Re-run whenever `dataset.json` changes.

```bash
uv run python -m evaluation.load_dataset --dataset-name "tool-call-eval"
```

### Step 2 — Run experiment (captures traces)

Runs the agent on every dataset item and creates Langfuse traces tagged with `--experiment-name`.  
Does **not** score — scoring is done in Step 3.

```bash
# Start MCP server first (separate terminal)
uv run python -m agent_eval.mcp_server.server

# Run experiment (all defaults: user=eval-tester, env=eval, session=<timestamp>)
uv run python -m evaluation.run_experiment \
  --dataset-name "tool-call-eval" \
  --experiment-name "exp-001"

# Override user / session / environment
uv run python -m evaluation.run_experiment \
  --dataset-name "tool-call-eval" \
  --experiment-name "exp-001" \
  --user "alice" \
  --session "session-42" \
  --environment "staging"
```

Each trace is tagged with `exp-001` and stores the following metadata for evaluators and dashboards:

| Metadata field | Set at | Description |
|---|---|---|
| `experiment` | trace creation | Experiment name (same as tag) |
| `dataset_name` | trace creation | Langfuse dataset name |
| `dataset_item_id` | trace creation | Dataset item ID (used as fallback by eval scripts) |
| `expected_tool_calls` | trace creation | Ground-truth tool calls from `dataset.json` |
| `category` | trace creation | Test case category (e.g. `perfect_match`, `multi_tool`) |
| `description` | trace creation | Human-readable description of the test case |
| `interface` | trace creation | Always `"run_experiment"` (vs `"streamlit_ui"` for app.py) |
| `tools_available` | trace creation | List of tool names loaded from MCP server |
| `tools_available_count` | trace creation | Number of tools loaded |
| `intent` | after agent run | `GENERIC` or `SPECIALIZED` (from `route_intent` node) |
| `complexity` | after agent run | `SINGLE_TOOL` or `MULTI_TOOL` (from `decompose_query` node) |
| `tools_called` | after agent run | Ordered list of tools actually executed |
| `tools_called_count` | after agent run | Number of tools actually executed |
| `actual_tool_call_count` | after agent run | Tool calls extracted from agent messages |
| `expected_tool_call_count` | after agent run | Number of expected tool calls from dataset |
| `response_length` | after agent run | Character length of the agent's final response |

### Step 3 — Score the traces

Run either or both evaluators against the experiment tag:

```bash
# RAGAS metrics
uv run python -m evaluation.eval_ragas --tag "exp-001"

# DeepEval metrics
uv run python -m evaluation.eval_deepeval --tag "exp-001"
```

Scores are written back to each Langfuse trace. View them in the Langfuse dashboard by filtering on tag `exp-001`.

---

## All CLI Options

### `load_dataset.py`

```bash
uv run python -m evaluation.load_dataset \
  --dataset-name "tool-call-eval"   # Name of the dataset in Langfuse
  [--dataset-file "dataset.json"]   # Path to JSON file (default: dataset.json)
```

### `run_experiment.py`

```bash
uv run python -m evaluation.run_experiment \
  --dataset-name "tool-call-eval"   # Dataset name in Langfuse (required)
  --experiment-name "exp-001"       # Tag applied to all traces (required)
  [--user "eval-tester"]            # Langfuse user_id on every trace (default: eval-tester)
  [--session "session-42"]          # Langfuse session_id shared across all traces
                                    # (default: UTC timestamp YYYYMMDDTHHMMSSZ)
  [--environment "eval"]            # Environment label in trace metadata (default: eval)
```

### `eval_ragas.py`

```bash
uv run python -m evaluation.eval_ragas \
  [--tag "exp-001"]                          # Filter traces by tag
  [--trace-ids trace-123 trace-456]          # Evaluate specific trace IDs
  [--from-date 2026-01-01]                   # Start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
  [--to-date   2026-01-31]                   # End date
  [--limit 100]                              # Max traces to fetch (default: 100)
  [--re-evaluate]                            # Re-score already-scored traces
  [--exit-on-error]                          # Exit code 1 if any evaluation fails
```

### `eval_deepeval.py`

```bash
uv run python -m evaluation.eval_deepeval \
  [--tag "exp-001"]                          # Filter traces by tag
  [--trace-ids trace-123 trace-456]          # Evaluate specific trace IDs
  [--from-date 2026-01-01]                   # Start date
  [--to-date   2026-01-31]                   # End date
  [--limit 100]                              # Max traces to fetch (default: 100)
  [--re-evaluate]                            # Re-score already-scored traces
  [--exit-on-error]                          # Exit code 1 if any evaluation fails
```

---

## Metrics Reference

### RAGAS (written by `eval_ragas.py`)

| Score name | Range | What it measures |
|---|---|---|
| `ragas_tool_call_accuracy` | 0.0 – 1.0 | Strict order match on **tool names** (args stripped — RAGAS `ToolCallAccuracy`) |
| `ragas_tool_call_f1` | 0.0 – 1.0 | F1 on **tool names** only, order-independent (args stripped — RAGAS `ToolCallF1`) |

> **Why args are stripped for RAGAS:** The agent generates SQL dynamically. Exact string matching against the reference SQL in `dataset.json` would always score 0. RAGAS metrics here evaluate *which tools were selected and in what order* — argument quality is assessed by DeepEval's `ArgumentCorrectnessMetric`.

### DeepEval (written by `eval_deepeval.py`)

| Score name | Range | LLM call? | What it measures |
|---|---|---|---|
| `deepeval_tool_correctness` | 0.0 – 1.0 | No | Tool name match — proportion of expected tools correctly called |
| `deepeval_argument_correctness` | 0.0 – 1.0 | **Yes** (OpenAI gpt-4o) | LLM-judged: are the tool call arguments relevant and correct for the user's question? |

> **What `deepeval_argument_correctness` actually measures:** This is a standard DeepEval `ArgumentCorrectnessMetric`. The LLM judge asks *"Given the user's input, are the SQL arguments the agent passed relevant and correct for that question?"* — it does **not** compare against the reference SQL in `dataset.json`. It validates argument relevance to the query, not identity with a reference. This is why it scores 1.0 when the agent generates sensible SQL. Requires `OPENAI_API_KEY` — one LLM call per evaluated trace.

> **Why `tool_correctness_with_args` was removed:** `ToolCorrectnessMetric(evaluation_params=[INPUT_PARAMETERS])` does exact dict comparison of `input_parameters`. The agent generates SQL dynamically so it never matches the reference SQL byte-for-byte — this metric always scored 0 and was misleading.

> **Skipped traces:** Traces where `expected_tool_calls` is empty (no-tool-needed queries) are automatically skipped by both evaluators — tool-call metrics are not meaningful for them.

---

## Dataset (`dataset.json`)

42 test cases covering all evaluation scenarios:

| Category | Count | What it tests |
|---|---|---|
| `perfect_match` | 7 | Single tool, correct name + SQL args — all metrics should be 1.0 |
| `real_world` | 7 | Realistic business queries (aggregations, joins, date ranges) |
| `argument_extraction` | 6 | SQL generation from natural language (LIKE, GROUP BY, BETWEEN, etc.) |
| `multi_tool` | 6 | Two or three tools in sequence |
| `no_tool_needed` | 4 | General/meta questions — agent should NOT call any tool |
| `ambiguous` | 3 | Vague requests — agent should ask for clarification |
| `conversational` | 3 | Casual phrasing with embedded data requests |
| `edge_case` | 3 | Unusual inputs (direct SQL, minimal requests, text search) |
| `wrong_order` | 2 | Explicit ordering instructions — tests execution order |
| `partial_completion` | 1 | Three-tool request — tests whether agent misses any tool |

**Available tools** (all take a single `query: str` SQL parameter):

| Tool | Table |
|---|---|
| `query_customer_reviews` | `samples.bakehouse.media_customer_reviews` |
| `query_gold_reviews_chunked` | `samples.bakehouse.media_gold_reviews_chunked` |
| `query_customers` | `samples.bakehouse.sales_customers` |
| `query_franchises` | `samples.bakehouse.sales_franchises` |
| `query_suppliers` | `samples.bakehouse.sales_suppliers` |
| `query_transactions` | `samples.bakehouse.sales_transactions` |

---

## How Expected Tool Calls Are Resolved

Both `eval_ragas.py` and `eval_deepeval.py` use the same fallback chain:

1. **`trace.metadata["expected_tool_calls"]`** ← set by `run_experiment.py` and `app.py` (primary source)
2. **Langfuse dataset item** via `trace.metadata["dataset_item_id"]` ← fallback for dataset runs

If neither resolves, the trace is skipped with a warning.

---

## Typical Workflows

### Developer — manual evaluation run

```bash
# Terminal 1: start MCP server
uv run python -m agent_eval.mcp_server.server

# Terminal 2: run evaluation pipeline
uv run python -m evaluation.load_dataset --dataset-name "tool-call-eval"
uv run python -m evaluation.run_experiment --dataset-name "tool-call-eval" --experiment-name "exp-001"
uv run python -m evaluation.eval_ragas    --tag "exp-001"
uv run python -m evaluation.eval_deepeval --tag "exp-001"
```

### Re-evaluate existing traces with updated metrics

```bash
uv run python -m evaluation.eval_ragas    --tag "exp-001" --re-evaluate
uv run python -m evaluation.eval_deepeval --tag "exp-001" --re-evaluate
```

### Evaluate by date range

```bash
uv run python -m evaluation.eval_ragas --from-date 2026-01-01 --to-date 2026-01-31
```

### CI/CD (GitHub Actions)

```yaml
- name: Start MCP server
  run: uv run python -m agent_eval.mcp_server.server &
  
- name: Run experiment
  run: |
    uv run python -m evaluation.run_experiment \
      --dataset-name "tool-call-eval" \
      --experiment-name "ci-${{ github.run_id }}" \
      --user "ci-bot" \
      --environment "ci"

- name: Score with RAGAS
  run: |
    uv run python -m evaluation.eval_ragas \
      --tag "ci-${{ github.run_id }}" --exit-on-error

- name: Score with DeepEval
  run: |
    uv run python -m evaluation.eval_deepeval \
      --tag "ci-${{ github.run_id }}" --exit-on-error
  env:
    LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
    LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No expected tool calls found — skipping` | Trace metadata missing `expected_tool_calls` | Re-run `run_experiment.py`; ensure dataset is loaded |
| `Failed to fetch dataset` | Dataset not in Langfuse | Run `load_dataset.py` first |
| `Failed to load MCP tools` | MCP server not running | Start server: `uv run python -m agent_eval.mcp_server.server` |
| `No traces found` | Wrong tag or date range | Check Langfuse dashboard for correct tag/IDs |
| SSL errors | Self-signed cert | Add `LANGFUSE_SSL_VERIFY=false` to `.env` |
| `Import errors` | Missing dependencies | Run `uv sync` |
| `ArgumentCorrectnessMetric failed` / auth errors in `eval_deepeval` | Missing `OPENAI_API_KEY` | Add `OPENAI_API_KEY=...` to `.env` — `deepeval_argument_correctness` uses an LLM judge (one call per trace) |
| `ragas_tool_call_accuracy = 0` for all traces | Stale scores from before args-stripping fix | Re-run with `--re-evaluate` flag |

---

## Resources

- [Langfuse Experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)
- [RAGAS ToolCallAccuracy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/tool_call_accuracy/)
- [DeepEval ToolCorrectnessMetric](https://deepeval.com/docs/metrics-tool-correctness)
- [DeepEval ArgumentCorrectnessMetric](https://deepeval.com/docs/metrics-argument-correctness)