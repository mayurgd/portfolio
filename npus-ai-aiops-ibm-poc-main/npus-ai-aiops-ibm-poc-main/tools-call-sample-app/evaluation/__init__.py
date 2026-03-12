"""
Evaluation framework — dataset (dataset.json), RAGAS, DeepEval, and runner.

Standalone top-level package. Depends on agent_eval.config for Langfuse
configuration, and on agent_eval.agent / agent_eval.tools / agent_eval.llm
at runtime (run_experiment.py only).

Dataset:
  evaluation/dataset.json  — ground-truth tool call scenarios (Bakehouse agent)
  evaluation/load_dataset.py — uploads dataset.json to Langfuse

Evaluation scripts:
  evaluation/eval_ragas.py    — RAGAS ToolCallAccuracy + ToolCallF1 (name-only)
  evaluation/eval_deepeval.py — DeepEval ToolCorrectness + ArgumentCorrectness
"""
