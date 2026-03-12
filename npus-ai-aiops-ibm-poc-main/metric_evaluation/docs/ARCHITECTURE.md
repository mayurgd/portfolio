# 🏗️ Architecture Overview

This document explains the architecture and design of the RAG Evaluation Framework.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     RAG Evaluation System                         │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴────────────────────┐
              │                                    │
         ┌────▼────┐                    ┌─────────▼─────────┐
         │   RAG   │                    │   Dual Evaluator  │
         │Pipeline │                    │   RAGAS+DeepEval  │
         └────┬────┘                    └─────────┬─────────┘
              │                                   │
    ┌─────────┼─────────┐              ┌─────────┼──────────┐
    │         │         │              │                    │
┌───▼───┐ ┌──▼──┐ ┌────▼────┐   ┌────▼─────┐      ┌──────▼──────┐
│Ingest │ │Retr │ │Generate │   │  RAGAS   │      │  DeepEval   │
│       │ │ieve │ │(NESGEN) │   │ 6 Metrics│      │14+ Metrics  │
└───┬───┘ └──┬──┘ └────┬────┘   └────┬─────┘      │+Safety      │
    │        │         │              │            └──────┬──────┘
    │   ┌────▼─────────▼────┐         │                   │
    │   │   Vector Store    │         │                   │
    │   │   (ChromaDB)      │         │                   │
    │   └───────────────────┘         │                   │
    │                                 │                   │
    └─────────────────┬───────────────┴───────────────────┘
                      │
               ┌──────▼──────┐
               │  Langfuse   │
               │  Tracking + │
               │Storage/Logs │
               └─────────────┘
```

## Component Details

### 1. RAG Pipeline (`src/rag_pipeline.py`)

The core RAG system with three main operations:

#### Ingestion
```python
documents → chunk → embed → store in ChromaDB
```
- **Text Splitter**: RecursiveCharacterTextSplitter
- **Chunk Size**: 500 tokens (configurable)
- **Overlap**: 50 tokens (configurable)
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB

#### Retrieval
```python
query → embed → similarity search → top-k contexts
```
- **Top-K**: 5 contexts (configurable)
- **Similarity**: Cosine similarity via ChromaDB
- **Threshold**: Configurable filtering

#### Generation
```python
query + contexts → prompt → NESGEN LLM → answer
```
- **Model**: NESGEN gpt-4.1 (Nestle Enterprise GenAI)
- **Temperature**: 0.0 for consistency
- **Prompt**: Context-grounded instruction prompt
- **API**: Custom NESGEN API wrapper for enterprise deployment

### 2. Evaluation Metrics

The framework now supports **dual evaluation** using both RAGAS and DeepEval:

#### RAGAS Evaluator (`src/evaluation_metrics.py`)
Uses RAGAS framework to compute 6 metrics:

**Category 1: Quality Metrics**
- Faithfulness (groundedness)
- Answer Correctness

**Category 2: Relevancy Metrics**
- Answer Relevancy
- Context Relevancy

**Category 3: Retrieval Metrics**
- Context Precision
- Context Recall

**Category 4: Similarity Metrics**
- Answer Similarity

#### DeepEval Evaluator (`src/deepeval_metrics.py`)
Provides comprehensive evaluation with 14+ metrics including safety:

**Core RAG Metrics**
- Faithfulness
- Answer Relevancy
- Contextual Precision
- Contextual Recall

**Safety & Quality Metrics**
- Hallucination Detection
- Bias Detection
- Toxicity Detection

**Key Features:**
- Custom NESGEN model wrapper (`NESGENDeepEvalModel`)
- Faster evaluation performance
- Better customization options
- Built-in testing framework
- Batch evaluation support

#### Production Evaluator (`src/evaluation/evaluator.py`)
Enterprise-grade async evaluation with:
- Rate limiting (100 API calls/minute)
- Retry logic (3 attempts with exponential backoff)
- Concurrent processing with asyncio
- Progress tracking
- Result persistence and aggregation
- Storage handler for local/cloud storage

#### Evaluation Flow
```
evaluation_data → prepare_dataset → 
  ├─ RAGAS.evaluate() → 6 scores
  └─ DeepEval.evaluate_batch() → 14+ scores (with safety)
    → aggregate results → log to Langfuse → persist to storage
```

### 3. Langfuse Integration

#### Tracking Layers

**Level 1: Traces**
- Each RAG query creates a trace
- Unique ID for tracking
- Session grouping support

**Level 2: Spans**
- Retrieval span (with contexts)
- Generation span (with prompt/response)
- Evaluation span (with scores from both frameworks)

**Level 3: Scores**
- All 6 RAGAS evaluation metrics
- All 14+ DeepEval metrics (including safety)
- Custom scores support
- Trend analysis and comparison

**Level 4: Metadata**
- User IDs
- Session IDs
- Tags and custom fields
- Framework identifiers (RAGAS vs DeepEval)

**Fallback Mechanism:**
- Automatic file-based storage when Langfuse unavailable
- Credentials validation via `config.langfuse.is_available()`
- Results saved to `evaluation_results/` directory

### 5. Utility Modules (`src/utils/`)

#### Logger (`logger.py`)
- Structured logging with timestamps
- File and console output
- Configurable log levels
- Context-aware formatting

#### Rate Limiter (`rate_limiter.py`)
- Token bucket algorithm
- 100 API calls per minute limit
- Prevents API throttling
- Used in ProductionEvaluator

#### Retry Handler (`retry_handler.py`)
- Exponential backoff strategy
- 3 retry attempts (1s, 2s, 4s delays)
- Handles transient API failures
- Automatic error recovery

### 4. Configuration (`config.py`)

Centralized configuration using Pydantic:

```python
config
├── nesgen (NESGEN-specific configuration)
│   ├── client_id
│   ├── client_secret
│   ├── api_base
│   ├── model (gpt-4.1)
│   ├── api_version
│   ├── temperature
│   ├── embedding_model (text-embedding-3-small)
│   ├── embedding_api_base
│   ├── embedding_api_version
│   └── openai_api_key_fallback (for embeddings)
├── langfuse
│   ├── secret_key
│   ├── public_key
│   ├── host
│   ├── enabled
│   └── is_available() (credential validation)
├── rag
│   ├── chunk_size
│   ├── chunk_overlap
│   ├── top_k
│   └── retrieval_threshold
└── evaluation
    ├── ragas_metrics (list of 6 RAGAS metrics)
    ├── deepeval_metrics (list of 7+ DeepEval metrics)
    └── batch_size
```

## Data Flow

### Query Flow
```
1. User Question
   ↓
2. Embed Question (OpenAI Embeddings)
   ↓
3. Retrieve Contexts (ChromaDB)
   ↓
4. Format Prompt (Question + Contexts)
   ↓
5. Generate Answer (GPT-4)
   ↓
6. Return Result {question, answer, contexts}
   ↓
7. Log to Langfuse (async)
```

### Evaluation Flow
```
1. Collect Evaluation Data
   {questions, answers, contexts, ground_truths}
   ↓
2. Prepare Datasets
   ├─ RAGAS Dataset
   └─ DeepEval Test Cases
   ↓
3. Run Dual Evaluation
   ├─ RAGAS Metrics (6 metrics)
   │   ├── Faithfulness
   │   ├── Answer Relevancy
   │   ├── Context Precision
   │   ├── Context Recall
   │   ├── Answer Similarity
   │   └── Answer Correctness
   │
   └─ DeepEval Metrics (14+ metrics)
       ├── Core RAG Metrics
       │   ├── Faithfulness
       │   ├── Answer Relevancy
       │   ├── Contextual Precision
       │   └── Contextual Recall
       └── Safety Metrics
           ├── Hallucination Detection
           ├── Bias Detection
           └── Toxicity Detection
   ↓
4. Aggregate & Compare Scores
   ↓
5. Log to Langfuse
   ├── Create Traces
   ├── Add Scores (RAGAS + DeepEval)
   └── Add Metadata
   ↓
6. Persist Results
   ├── Save to CSV
   ├── Local file storage (fallback)
   └── Return aggregated results
```

## Key Design Decisions

### Why ChromaDB?
- In-memory for fast prototyping, persistent storage option
- **Production Alternatives**: Pinecone, Weaviate, Qdrant

### Why Dual Evaluation (RAGAS + DeepEval)?
- **RAGAS**: Academically validated, research-backed, comprehensive retrieval analysis
- **DeepEval**: Faster evaluation, 14+ metrics, safety metrics (bias, toxicity, hallucination)
- **Strategy**: Use RAGAS for research/academic validation, DeepEval for production and safety-critical applications
- **Best Practice**: Use BOTH for comprehensive validation

### Why Langfuse?
- Purpose-built for LLM observability, excellent tracing
- Supports both RAGAS and DeepEval metric tracking
- **Alternatives**: LangSmith, Weights & Biases

### Why NESGEN?
- Enterprise-grade API specifically for Nestle GenAI
- Custom wrapper (`NESGENDeepEvalModel`) for DeepEval compatibility
- Centralized authentication and access control
- **Alternatives**: OpenAI, Anthropic (require code modifications)

## Extension Points

### Adding New Metrics

**For RAGAS:**
```python
# src/evaluation_metrics.py
from ragas.metrics import custom_metric

self.available_metrics = {
    # ... existing metrics
    "custom_metric": custom_metric
}
```

**For DeepEval:**
```python
# src/deepeval_metrics.py
from deepeval.metrics import CustomMetric

self.available_metrics = {
    # ... existing metrics
    "custom_metric": lambda: CustomMetric(
        threshold=self.threshold,
        model=self.nesgen_model
    )
}
```

### Adding New Vector Stores

```python
# src/rag_pipeline.py
from langchain.vectorstores import Pinecone

class RAGPipeline:
    def __init__(self, vectorstore_type="chroma"):
        if vectorstore_type == "pinecone":
            self.vectorstore = Pinecone(...)
        # ... etc
```

### Adding New LLM Providers

```python
# config.py - Add new provider config
class AnthropicConfig(BaseModel):
    api_key: str
    model: str = "claude-3-opus-20240229"

# src/nesgen_llm.py or new provider file
from langchain_anthropic import ChatAnthropic

if config.provider == "anthropic":
    self.llm = ChatAnthropic(...)

# src/deepeval_metrics.py - Create wrapper
class AnthropicDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatAnthropic(...)
    # ... implement required methods
```

### Custom Evaluation Logic

```python
# src/evaluation_metrics.py
def custom_evaluate(self, question, answer, contexts):
    """Your custom evaluation logic"""
    score = your_custom_metric(question, answer)
    return score
```

## Performance Considerations

### Async Batch Processing
```python
# batch_evaluation.py - Production-grade async evaluation
from src.evaluation import ProductionEvaluator

evaluator = ProductionEvaluator()
result = await evaluator.evaluate_batch_async(
    questions, answers, contexts, ground_truths,
    max_concurrent=5  # Control concurrency
)
```

### Rate Limiting
```python
# src/utils/rate_limiter.py
# Built-in rate limiting: 100 API calls/minute
# Automatically applied in ProductionEvaluator
```

### Retry Logic
```python
# src/utils/retry_handler.py
# Automatic retry with exponential backoff
# 3 attempts with 1s, 2s, 4s delays
```

### Embedding Caching
```python
# Consider caching embeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
```

### Result Persistence
```python
# src/evaluation/storage_handler.py
# Automatic storage to local files
# Fallback when Langfuse unavailable
```

## Security Best Practices

1. **API Keys**: Never commit `.env` file
2. **Input Validation**: Sanitize user inputs
3. **Rate Limiting**: Implement rate limits for APIs
4. **Access Control**: Restrict Langfuse access
5. **Data Privacy**: Be careful with sensitive data in traces

## Testing Strategy

### Unit Tests
```python
# Note: Test files are placeholders - to be implemented
# Proposed structure:
# tests/test_rag_pipeline.py
- Test document ingestion
- Test retrieval
- Test generation
- Test query pipeline
```

### Integration Tests
```python
# Note: Test files are placeholders - to be implemented
# Proposed structure:
# tests/test_evaluation.py
- Test metric computation
- Test dataset preparation
- Test Langfuse logging
```

### End-to-End Tests
```python
# comprehensive_demo.py (serves as E2E test)
- Full pipeline execution with NESGEN
- Dual evaluation (RAGAS + DeepEval)
- Safety metrics evaluation
- Configuration comparison
- Framework comparison
- Langfuse integration
- Result persistence
```

## Monitoring in Production

### Key Metrics to Track (via Langfuse)

**Performance**: Latency, token usage, cost per query, API response times
**Quality**: 
- RAGAS metrics: Faithfulness, answer relevancy, context precision/recall trends
- DeepEval metrics: Hallucination rate, bias score, toxicity score
- User feedback scores
**System**: Error rates, success rates, throughput, retry counts
**User**: Session metrics, satisfaction, feature usage
**Safety**: Hallucination trends, bias patterns, toxicity incidents

For detailed monitoring setup, see **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)**.

## Scaling Considerations

### For Higher Load
- Migrate to managed vector store (Pinecone, Weaviate)
- Add caching layer (Redis)
- Use load balancing and async processing queues

### For More Data
- Optimize chunking strategy
- Implement hybrid search (dense + sparse)
- Add reranking and hierarchical retrieval

For detailed scaling strategies, see **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)**.

## Cost Optimization

- Cache embeddings and responses for common queries
- Use smaller models for non-critical use cases
- Batch operations when possible
- Run evaluations on samples, not all data
- Schedule batch evaluations during off-peak hours

For detailed cost optimization strategies, see **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)**.

## Future Enhancements

### Short Term
- [x] Support for DeepEval framework (COMPLETED)
- [x] Async batch evaluation (COMPLETED)
- [x] Result persistence and storage handler (COMPLETED)
- [x] Rate limiting and retry logic (COMPLETED)
- [ ] Support for multiple vector stores
- [ ] Embedding caching
- [ ] More example datasets

### Medium Term
- [ ] Hybrid search (dense + sparse)
- [ ] Automatic prompt optimization
- [ ] A/B testing framework
- [ ] Custom metric builder UI
- [ ] Real-time alerting for safety metrics
- [ ] Advanced hallucination detection

### Long Term
- [ ] Multi-modal RAG support
- [ ] Real-time evaluation dashboard
- [ ] Auto-tuning RAG parameters
- [ ] Production deployment templates
- [ ] Federated learning for evaluation models

## Resources

- **RAGAS**: https://docs.ragas.io/
- **DeepEval**: https://docs.confident-ai.com/
- **Langfuse**: https://langfuse.com/docs
- **LangChain**: https://python.langchain.com/
- **ChromaDB**: https://docs.trychroma.com/
- **NESGEN**: Internal Nestle GenAI documentation

---

For implementation details, see the source code and inline documentation.
For production deployment, see **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** and `batch_evaluation.py`.