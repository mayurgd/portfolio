# RAG Evaluation Template - Copilot Instructions

This repository provides a template for evaluating RAG (Retrieval-Augmented Generation) pipelines using LangFuse for tracing and RAGAS for metrics.

## Project Structure

- `cookbooks/` — Jupyter notebooks with evaluation experiments
- `docs/` — Setup guides for LangFuse and OpenAI/Azure OpenAI

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | `PascalCase` | `DataProcessor`, `AzureOpenAILLM` |
| Functions | `snake_case` | `fetch_data`, `process_records` |
| Methods | `snake_case` | `get_items`, `run_query` |
| Variables | `snake_case` | `user_input`, `max_retries` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_TIMEOUT`, `MAX_RETRIES` |
| Private | `_leading_underscore` | `_internal_cache`, `_parse_response` |

## Error Handling

Provide clear, actionable error messages with context:

- Print user-friendly messages to stderr
- Support `--verbose` flag for detailed tracebacks
- Handle optional dependencies gracefully with fallbacks

## Code Style

- **Line length**: 120 characters maximum
- **Quotes**: Use double quotes for strings
- **Trailing commas**: Include in multi-line collections
- **Blank lines**: Two between top-level definitions, one between methods

## Additional Instructions

Python-specific conventions (imports, type hints, docstrings, testing patterns, LangChain/LangFuse patterns) are loaded automatically when editing `.py` or `.ipynb` files via `.github/instructions/python.instructions.md`.
