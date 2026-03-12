---
applyTo: "**/*.{py,ipynb}"
---

# Python Best Practices

Follow these conventions when generating or modifying Python code in this repository.

## Import Organization

Always organize imports in this order, with blank lines between groups:

```python
"""Module docstring."""

from __future__ import annotations  # Always first

# 1. Standard library imports
import json
import os
from pathlib import Path
from typing import Any

# 2. Third-party imports
import pandas as pd
from pydantic import BaseModel

# 3. Local imports (relative or absolute)
from rag_eval.utils.env import env
```

Use `from __future__ import annotations` at the top of every Python file to enable postponed annotation evaluation.

## Type Hints

Use comprehensive type hints throughout:

```python
# Use modern union syntax (Python 3.10+)
def process(value: str | None = None) -> dict[str, Any]:
    ...

# Use lowercase generic types (Python 3.9+)
def get_items() -> list[dict[str, Any]]:
    ...

# Type hint function signatures completely
def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    ...
```

## Docstrings

Use Google-style docstrings:

```python
def fetch_data(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch JSON data from a remote URL.

    Args:
        url: The endpoint URL to fetch from.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        requests.RequestException: If the request fails.
        ValueError: If the response is not valid JSON.
    """
```

For classes:

```python
class DataProcessor:
    """Process and transform data records.

    Attributes:
        batch_size: Number of records to process at once.
        strict_mode: Whether to raise on invalid records.
    """

    def __init__(self, batch_size: int = 100, strict_mode: bool = False) -> None:
        self.batch_size = batch_size
        self.strict_mode = strict_mode
```

## Design Patterns

### Protocol-Based Interfaces

Use `typing.Protocol` for interface definitions (structural typing):

```python
from typing import Protocol

class Retriever(Protocol):
    """Interface for document retrievers."""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        ...
```

Implementations don't need to explicitly inherit—just implement the methods.

### Composition Over Inheritance

Prefer composing objects rather than deep inheritance hierarchies:

```python
class Pipeline:
    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self.retriever = retriever
        self.llm = llm
```

### Lazy Imports for Heavy Dependencies

Defer imports of heavy dependencies to function scope:

```python
def process_with_pandas(data: list[dict]) -> None:
    import pandas as pd  # Deferred import
    df = pd.DataFrame(data)
    ...
```

### Explicit Public API

Use `__all__` in `__init__.py` to declare public interfaces:

```python
__all__ = [
    "Pipeline",
    "Retriever",
    "LLM",
]
```

## Testing

### Descriptive Fixtures

Use fixtures with clear docstrings:

```python
@pytest.fixture
def mock_retriever() -> MockRetriever:
    """Provide a mock retriever for testing."""
    return MockRetriever()
```

### Protocol-Compliant Mocks

Create mocks that implement the Protocol interface with call tracking:

```python
class MockLLM:
    """Mock LLM implementing the LLM protocol."""

    def __init__(self, response: str = "mock response") -> None:
        self.response = response
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self.response
```

### Descriptive Test Names

Name tests clearly: `test_<what_is_being_tested>`:

```python
def test_retriever_returns_top_k_results() -> None:
    """Verify retriever respects the top_k parameter."""
    ...
```

## Configuration

### Environment Variables

Use a helper function for environment variable access:

```python
from dotenv import load_dotenv
load_dotenv()

def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val
```

### JSON Configuration Files

Store configuration in JSON files, not hardcoded:

```python
config = json.loads(Path("configs/settings.json").read_text())
```

## LangChain v1.x Patterns

This project uses LangChain v1.x. Follow these patterns:

### Import Structure

Use `langchain_core` for base classes and `langchain_openai` for OpenAI/Azure integrations:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import AzureChatOpenAI, ChatOpenAI
```

### LCEL Chain Composition

Build chains using the pipe operator (`|`) for LangChain Expression Language:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}"),
])
chain = prompt | llm
```

### Callback Configuration

Pass callbacks via `config` parameter, not as direct arguments:

```python
from langfuse.langchain import CallbackHandler

handler = CallbackHandler()
result = chain.invoke(
    {"question": "What is AI?"},
    config={"callbacks": [handler]},
)
```

### Azure OpenAI Configuration

Configure Azure OpenAI via environment variables:

```python
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
)
```

## LangFuse Trace Association

Properly associate scores with traces using these patterns:

### Capturing Trace ID

Use `langfuse_context` to get the current trace ID inside `@observe()` decorated functions:

```python
from langfuse import observe
from langfuse.decorators import langfuse_context

@observe()
def traced_function():
    trace_id = langfuse_context.get_current_trace_id()
    # ... perform work ...
    return {"result": result, "trace_id": trace_id}
```

### Recording Scores

Always pass `trace_id` when recording scores:

```python
lf = get_langfuse_client()
lf.score(
    trace_id=trace_id,  # Required - never pass None
    name="faithfulness",
    value=0.95,
    comment="High faithfulness to context",
)
```

### Score Recording Helper

Use the `record_score` helper for consistent score recording:

```python
from rag_eval.tracing_setup import record_score

record_score(
    name="ragas_faithfulness",
    value=result.faithfulness,
    trace_id=trace_id,
    comment="RAGAS faithfulness metric",
)
```
