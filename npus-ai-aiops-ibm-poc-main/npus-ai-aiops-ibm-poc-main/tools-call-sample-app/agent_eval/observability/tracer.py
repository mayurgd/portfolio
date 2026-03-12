"""Langfuse Python-SDK v2 tracing utilities (manual instrumentation).

Uses the **Langfuse Python SDK v2** directly — no LangChain callback handler.
This gives full control over which operations are traced and how they appear
in the Langfuse dashboard.

Key v2 concepts:
  • A *trace* is created with ``client.trace()``.
  • ``trace.span()`` for nested spans.
  • ``trace.generation()`` for LLM calls.
  • ``trace.update()`` to set output / metadata.
  • ``trace.score()`` to attach scores.

Usage in the agent graph::

    tracer = get_tracer()
    root = tracer.start_trace(name="chat", session_id="s1",
                              user_id="demo", input=user_msg)

    # … invoke the graph with root span in config …
    result = graph.invoke({"messages": messages},
                          config={"configurable": {"langfuse_span": root}})

    tracer.update_trace(root, output=result_text)
    tracer.flush()
"""

import logging
from typing import Any, Dict, List, Optional

from agent_eval.config import get_langfuse_config

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Manages a Langfuse client and provides helpers for controlled tracing.

    All methods are safe to call even when Langfuse is not configured — they
    simply return ``None`` or no-op.
    """

    def __init__(self) -> None:
        self._config = get_langfuse_config()
        self._client = None

        if self._config.enabled:
            try:
                import httpx
                from langfuse import Langfuse

                if self._config.ssl_verify is False:
                    import os
                    import urllib3

                    logger.warning(
                        "Langfuse: SSL verification DISABLED (LANGFUSE_SSL_VERIFY=false). "
                        "Only use this behind a trusted corporate proxy."
                    )
                    os.environ.setdefault("CURL_CA_BUNDLE", "")
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                elif isinstance(self._config.ssl_verify, str):
                    logger.info("Langfuse: using custom CA bundle: %s", self._config.ssl_verify)

                httpx_client = httpx.Client(verify=self._config.ssl_verify)

                # Disable SDK self-instrumentation via OTEL — local Langfuse instances
                # often lack the OTEL endpoint, causing "Internal server error".
                import os as _os
                _os.environ.setdefault("LANGFUSE_SDK_INTEGRATION_TRACING_ENABLED", "false")

                self._client = Langfuse(
                    public_key=self._config.public_key,
                    secret_key=self._config.secret_key,
                    host=self._config.host,
                    httpx_client=httpx_client,
                )
                logger.info(
                    "Langfuse tracing enabled (%s, ssl_verify=%s)",
                    self._config.host,
                    self._config.ssl_verify,
                )
            except Exception as e:
                logger.warning("Failed to initialise Langfuse client: %s", str(e), exc_info=True)
                self._client = None
        else:
            logger.info("Langfuse tracing disabled — credentials not configured")

    @property
    def enabled(self) -> bool:
        """Return ``True`` when a live Langfuse client is available."""
        return self._client is not None

    def start_trace(
        self,
        name: str,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        input: Optional[Any] = None,
        metadata: Optional[Any] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a Langfuse trace.

        Returns the trace object (StatefulTraceClient), or ``None`` if disabled.
        """
        if self._client is None:
            return None
        try:
            trace = self._client.trace(  # type: ignore
                name=name,
                input=input,
                metadata=metadata,
                session_id=session_id,
                user_id=user_id,
                tags=tags,
            )
            return trace
        except Exception as e:
            logger.warning("Failed to start Langfuse trace: %s", str(e), exc_info=True)
            return None

    def update_trace(self, trace, **kwargs: Any) -> None:
        """Update trace-level attributes (output, metadata, etc.)."""
        if trace is None:
            return
        try:
            trace.update(**kwargs)
        except Exception:
            logger.warning("Failed to update Langfuse trace", exc_info=True)

    def start_span(self, parent, name: str, **kwargs: Any):
        """Create a span under a trace or another span. Returns a StatefulSpanClient."""
        if parent is None:
            return None
        try:
            return parent.span(name=name, **kwargs)
        except Exception:
            logger.warning("Failed to start Langfuse span", exc_info=True)
            return None

    def end_span(self, span, *, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update output/metadata and end a span. Traces are flushed automatically."""
        if span is None:
            return
        try:
            update_kwargs: Dict[str, Any] = {}
            if output is not None:
                update_kwargs["output"] = output
            if metadata:
                update_kwargs["metadata"] = metadata
            if update_kwargs:
                span.update(**update_kwargs)
            
            if hasattr(span, 'end') and callable(getattr(span, 'end')):
                span.end()
        except Exception:
            logger.warning("Failed to end Langfuse span", exc_info=True)

    def start_generation(self, parent, name: str, **kwargs: Any):
        """Start an LLM generation under a trace or span. Returns a StatefulGenerationClient."""
        if parent is None:
            return None
        try:
            return parent.generation(name=name, **kwargs)
        except Exception:
            logger.warning("Failed to start Langfuse generation", exc_info=True)
            return None

    def end_generation(
        self,
        generation,
        *,
        output: Any = None,
        usage_details: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update output / usage / metadata for a generation and end it."""
        if generation is None:
            return
        try:
            update_kwargs: Dict[str, Any] = {}
            if output is not None:
                update_kwargs["output"] = output
            if usage_details:
                update_kwargs["usage"] = usage_details
            if metadata:
                update_kwargs["metadata"] = metadata
            if update_kwargs:
                generation.update(**update_kwargs)
            generation.end()
        except Exception:
            logger.warning("Failed to end Langfuse generation", exc_info=True)

    def score_trace(
        self,
        trace,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ) -> None:
        """Attach a score to a trace."""
        if trace is None or self._client is None:
            return
        try:
            if hasattr(trace, 'score'):
                trace.score(name=name, value=value, comment=comment)
            else:
                logger.warning("Trace object does not have score() method")
        except Exception:
            logger.warning("Failed to score Langfuse trace", exc_info=True)

    def flush(self) -> None:
        """Flush any buffered events to Langfuse."""
        if self._client is not None:
            self._client.flush()

    def shutdown(self) -> None:
        """Flush and release resources."""
        if self._client is not None:
            self._client.flush()
            self._client.shutdown()


_TRACER: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    """Return the module-level singleton ``LangfuseTracer``."""
    global _TRACER
    if _TRACER is None:
        _TRACER = LangfuseTracer()
    return _TRACER


def check_observability() -> tuple[bool, str]:
    """Check if Langfuse is configured and accessible. Returns (ok, message)."""
    config = get_langfuse_config()

    if not config.enabled:
        return False, "Langfuse not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env"

    logger.info("Langfuse credentials configured. Testing connectivity...")

    try:
        global _TRACER
        _TRACER = None
        tracer = get_tracer()

        if not tracer.enabled:
            return False, "Langfuse tracer failed to initialize. Check credentials and network."

        logger.info("Creating test trace...")
        root = tracer.start_trace(
            name="connectivity_check",
            user_id="test_user",
            session_id="test_session",
            input={"test": "connectivity check"},
            metadata={"source": "tracer.py", "type": "health_check"},
            tags=["health_check", "connectivity_test"]
        )
        
        if root is None:
            return False, "Failed to create test trace"

        child_span = tracer.start_span(
            root,
            name="test_span",
            input={"message": "Testing Langfuse connectivity"}
        )
        tracer.end_span(child_span, output={"status": "success"})

        generation = tracer.start_generation(
            root,
            name="test_generation",
            model="test-model",
            input=[{"role": "user", "content": "test"}]
        )
        tracer.end_generation(generation, output="test response")
        tracer.update_trace(root, output={"status": "connectivity_check_passed"})
        tracer.end_span(root)
        tracer.flush()

        logger.info("✓ Successfully connected to Langfuse")
        return True, f"✓ Successfully connected to Langfuse at {config.host}"
        
    except Exception as e:
        error_msg = f"✗ Failed to connect to Langfuse: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def capture_agent_step(
    parent_span,
    step_name: str,
    input_data: Any,
    output_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a single agent step as a Langfuse span."""
    tracer = get_tracer()
    if not tracer.enabled or parent_span is None:
        return
    
    try:
        span = tracer.start_span(
            parent_span,
            name=step_name,
            input=input_data,
            metadata=metadata or {}
        )
        tracer.end_span(span, output=output_data)
    except Exception:
        logger.warning(f"Failed to capture agent step: {step_name}", exc_info=True)


def capture_llm_call(
    parent_span,
    model: str,
    input_messages: List[Dict[str, str]],
    output_text: str,
    usage: Optional[Dict[str, int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture an LLM call as a Langfuse generation."""
    tracer = get_tracer()
    if not tracer.enabled or parent_span is None:
        return
    
    try:
        generation = tracer.start_generation(
            parent_span,
            name=f"llm_call_{model}",
            model=model,
            input=input_messages,
            metadata=metadata or {}
        )
        
        tracer.end_generation(
            generation,
            output=output_text,
            usage_details=usage
        )
    except Exception:
        logger.warning(f"Failed to capture LLM call: {model}", exc_info=True)


def capture_tool_call(
    parent_span,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a tool call as a Langfuse span."""
    tracer = get_tracer()
    if not tracer.enabled or parent_span is None:
        return
    
    try:
        span = tracer.start_span(
            parent_span,
            name=f"tool_{tool_name}",
            input=tool_input,
            metadata={**(metadata or {}), "tool_type": "function_call"}
        )
        tracer.end_span(span, output=tool_output)
    except Exception:
        logger.warning(f"Failed to capture tool call: {tool_name}", exc_info=True)


def capture_retrieval(
    parent_span,
    query: str,
    documents: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture a document retrieval operation as a Langfuse span."""
    tracer = get_tracer()
    if not tracer.enabled or parent_span is None:
        return
    
    try:
        span = tracer.start_span(
            parent_span,
            name="retrieval",
            input={"query": query},
            metadata={
                **(metadata or {}),
                "num_documents": len(documents),
                "operation_type": "retrieval"
            }
        )
        tracer.end_span(span, output={"documents": documents})
    except Exception:
        logger.warning("Failed to capture retrieval", exc_info=True)


def capture_evaluation_score(
    span,
    metric_name: str,
    score: float,
    comment: Optional[str] = None,
) -> None:
    """Attach an evaluation score to a Langfuse trace."""
    tracer = get_tracer()
    if not tracer.enabled or span is None:
        return
    
    tracer.score_trace(span, name=metric_name, value=score, comment=comment)


def main():
    """Test Langfuse observability connectivity."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Langfuse Observability Check")
    print("=" * 60)

    config = get_langfuse_config()
    print(f"\nHost: {config.host}")
    print(f"SSL Verify: {config.ssl_verify}")
    print("-" * 60)

    print("\n1. Checking Langfuse configuration...")
    if not config.enabled:
        print("   ✗ Langfuse not configured")
        print("   Please set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")
        sys.exit(1)
    
    print(f"   ✓ Public key: {config.public_key[:20]}...")
    print(f"   ✓ Secret key: {config.secret_key[:20]}...")

    print("\n2. Testing Langfuse connectivity...")
    conn_ok, conn_msg = check_observability()
    print(f"   {conn_msg}")
    
    print("\n" + "=" * 60)
    if conn_ok:
        print("✅ Langfuse observability check PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ Langfuse observability check FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
