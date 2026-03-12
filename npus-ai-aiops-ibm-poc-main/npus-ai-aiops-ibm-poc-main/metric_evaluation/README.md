# RAG Evaluation Framework with Langfuse Integration

A comprehensive, production-ready RAG (Retrieval-Augmented Generation) evaluation framework that supports **RAGAS**, **DeepEval**, and **Production-grade** evaluation with full Langfuse integration. Built specifically for **NESGEN (Nestle GenAI)** API.

## 🚀 Quick Start

See **[QUICKSTART.md](docs/QUICKSTART.md)** for detailed setup instructions.

```bash
# Install dependencies and create virtual environment
uv sync

# Run the main demo
uv run python comprehensive_demo.py
```

---

## ✨ Key Highlights

- 🎯 **21+ Evaluation Metrics**: 7 RAGAS + 14 DeepEval metrics
- 🏭 **Production-Ready**: Rate limiting, retry logic, and error handling
- 🛡️ **Safety First**: Bias, toxicity, and hallucination detection
- 🔍 **Full Observability**: Complete Langfuse integration
- ⚡ **High Performance**: Batch processing and parallel execution
- 🔧 **Flexible**: Use RAGAS, DeepEval, or both simultaneously
- 🎨 **NESGEN Native**: Built specifically for Nestle GenAI platform

## 📋 System Requirements

### Required
- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- NESGEN API credentials (client ID and secret)
- 2GB+ RAM for vector database operations

### Optional
- OpenAI API key (for embeddings fallback)
- Langfuse account (for observability, free tier available)
- GPU (for faster embedding generation with sentence-transformers)

### Supported Platforms
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 20.04+, Debian, etc.)

## 📖 Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[METRICS_GUIDE.md](docs/METRICS_GUIDE.md)** - All evaluation metrics explained
- **[PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)** - Production deployment guide
- **[FRAMEWORK_COMPARISON.md](docs/FRAMEWORK_COMPARISON.md)** - RAGAS vs DeepEval comparison
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture details

## 🎯 Features

### Complete RAG Pipeline
- **NESGEN Integration**: Built specifically for **NESGEN (Nestle GenAI)** API
- **Document Ingestion**: Chunk and embed documents into ChromaDB vector store
- **Semantic Retrieval**: Retrieve relevant contexts using embeddings
- **Answer Generation**: Generate answers using NESGEN gpt-4.1 model
- **Configurable**: Easily adjust chunk sizes, top-k, and other parameters

### Dual Evaluation Frameworks
- **RAGAS Framework**: 7 academically validated metrics
- **DeepEval Framework**: 14+ metrics including safety checks (bias, toxicity, hallucination)
- **Framework Comparison**: Built-in tools to compare RAGAS vs DeepEval results
- **Flexible Selection**: Use either framework or both simultaneously

### Production-Ready Architecture
- **Modular Evaluator**: Separated concerns for metrics, aggregation, and storage
- **Rate Limiting**: Built-in rate limiter for API calls
- **Retry Handler**: Automatic retry logic with exponential backoff
- **Structured Logging**: Comprehensive logging for debugging and monitoring
- **Result Storage**: Persistent storage of evaluation results
- **Batch Processing**: Efficient batch evaluation capabilities

### All Evaluation Metrics

#### RAGAS Framework (7 Metrics)

#### Quality Metrics
- **Faithfulness**: Measures how grounded the answer is in the retrieved contexts (no hallucinations)
- **Answer Correctness**: Compares generated answer against ground truth

#### Relevancy Metrics
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Context Relevancy**: Measures how relevant the retrieved contexts are to the question

#### Retrieval Metrics
- **Context Precision**: Measures the signal-to-noise ratio of retrieved contexts
- **Context Recall**: Measures if all relevant information was retrieved

#### Similarity Metrics
- **Answer Similarity**: Semantic similarity between generated and ground truth answers

#### DeepEval Framework (14+ Metrics)

##### Core RAG Metrics
- **Faithfulness**: Measures factual consistency with source contexts
- **Answer Relevancy**: Evaluates answer relevance to the question
- **Contextual Precision**: Measures ranking quality of retrieved contexts
- **Contextual Recall**: Measures completeness of context retrieval

##### Safety & Ethics Metrics
- **Hallucination**: Detects fabricated or unsupported information
- **Bias**: Identifies demographic, gender, political bias in responses
- **Toxicity**: Detects harmful, offensive, or inappropriate content

##### Advanced Metrics
- **G-Eval**: Custom criteria-based evaluation with LLM-as-judge
- **Summarization**: Evaluates quality of summarization tasks
- **RAG-AS**: RAGAS-style metrics with DeepEval implementation
- **Custom Metrics**: Build your own evaluation criteria

### Langfuse Integration
- **Automatic Tracing**: Every query and evaluation is automatically tracked
- **Score Logging**: All metrics are logged as scores in Langfuse
- **Session Management**: Batch evaluations can be grouped into sessions
- **Custom Metadata**: Add custom tags, metadata, and user information
- **Performance Analytics**: View latency, token usage, and costs in Langfuse dashboard

## 📁 Project Structure

```
aiopspoc/
├── config.py                           # Configuration management
├── pyproject.toml                      # Project metadata and dependencies (uv)
├── .python-version                     # Pinned Python version for uv
├── requirements.txt                    # Legacy dependency list (kept for reference)
├── .env                               # Environment variables (not committed)
├── README.md                          # This file
├── comprehensive_demo.py              # 🌟 MAIN DEMO - Showcases ALL features
│
├── docs/
│   ├── QUICKSTART.md                  # 5-minute quick start guide
│   ├── METRICS_GUIDE.md               # Detailed metrics explanation
│   ├── PRODUCTION_GUIDE.md            # Production deployment guide
│   ├── FRAMEWORK_COMPARISON.md        # RAGAS vs DeepEval comparison
│   └── ARCHITECTURE.md                # System architecture details
│
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py                # Complete RAG pipeline
│   ├── nesgen_llm.py                  # NESGEN API integration
│   ├── evaluation_metrics.py          # RAGAS evaluation framework
│   ├── deepeval_metrics.py            # DeepEval evaluation framework
│   ├── sample_data.py                 # Sample documents and test data
│   │
│   ├── evaluation/                    # Production evaluation module
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Main evaluator orchestrator
│   │   ├── metrics_calculator.py     # Metrics computation
│   │   ├── result_aggregator.py      # Results aggregation
│   │   └── storage_handler.py        # Persistent storage
│   │
│   └── utils/                         # Utility modules
│       ├── __init__.py
│       ├── logger.py                  # Structured logging
│       ├── rate_limiter.py            # API rate limiting
│       └── retry_handler.py           # Retry logic with backoff
│
└── tests/                             # Test directory (placeholder)
    └── __init__.py
```

## 💡 What the Demo Showcases

The `comprehensive_demo.py` demonstrates:
- RAG pipeline with NESGEN (ingest, retrieve, generate)
- RAGAS evaluation (7 metrics) + DeepEval (14+ metrics)
- Production features (async, rate limiting, retry logic)
- Configuration comparison and quality analysis
- Framework comparison and CSV export
- Full Langfuse integration

## 🤔 Which Framework to Use?

See **[FRAMEWORK_COMPARISON.md](docs/FRAMEWORK_COMPARISON.md)** for detailed comparison.

**Quick Answer:**
- **RAGAS**: Academic research, LangChain projects, proven metrics
- **DeepEval**: Production systems, safety metrics, faster performance
- **Both**: Maximum confidence and comprehensive coverage

## 📊 Usage Examples

### Basic RAG Query

```python
from src.rag_pipeline import RAGPipeline

# Initialize NESGEN RAG pipeline
rag = RAGPipeline(collection_name="my_docs")

# Ingest documents
documents = ["Document 1 text...", "Document 2 text..."]
rag.ingest_documents(documents)

# Query using NESGEN
result = rag.query("What is machine learning?")
print(result['answer'])
print(result['contexts'])
```

### Evaluate with RAGAS Metrics

```python
from src.evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()

# Evaluate with all RAGAS metrics
scores = evaluator.evaluate(
    questions=["What is AI?"],
    answers=["AI is artificial intelligence..."],
    contexts=[["AI is a field of computer science..."]],
    ground_truths=["AI stands for artificial intelligence..."],
    metrics=[
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_similarity",
        "answer_correctness",
        "context_relevancy"
    ]
)

print(evaluator.get_metrics_summary(scores))
```

### Evaluate with DeepEval Metrics

```python
from src.deepeval_metrics import DeepEvalEvaluator

evaluator = DeepEvalEvaluator()

# Evaluate with DeepEval including safety metrics
scores = evaluator.evaluate(
    questions=["What is AI?"],
    answers=["AI is artificial intelligence..."],
    contexts=[["AI is a field of computer science..."]],
    ground_truths=["AI stands for artificial intelligence..."],
    metrics=[
        "faithfulness",
        "answer_relevancy",
        "contextual_precision",
        "contextual_recall",
        "hallucination",
        "bias",
        "toxicity"
    ]
)

print(evaluator.get_metrics_summary(scores))
```

### Production Evaluator (Recommended)

```python
from src.evaluation.evaluator import ProductionEvaluator

evaluator = ProductionEvaluator(
    framework="both",  # Use both RAGAS and DeepEval
    enable_rate_limiting=True,
    enable_retry=True,
    persist_results=True
)

# Batch evaluation with automatic rate limiting and retry
results = evaluator.evaluate_batch(
    questions=["What is AI?", "What is ML?"],
    answers=["AI is...", "ML is..."],
    contexts=[[...], [...]],
    ground_truths=["...", "..."]
)

# Results are automatically saved and logged to Langfuse
print(evaluator.get_summary_report(results))
```

### Compare Different Configurations

```python
# Compare different retrieval configurations
from src.rag_pipeline import RAGPipeline
from src.evaluation.evaluator import ProductionEvaluator

results = {}
for top_k in [3, 5, 7, 10]:
    rag = RAGPipeline(top_k=top_k)
    evaluator = ProductionEvaluator()
    
    # Run evaluation
    scores = evaluator.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    results[f"top_k_{top_k}"] = scores

# Compare results
print(evaluator.compare_configurations(results))
```

### Compare RAGAS vs DeepEval

```python
# Use the comprehensive demo to see both frameworks in action
# python comprehensive_demo.py

# Or manually compare frameworks:
from src.evaluation_metrics import RAGEvaluator
from src.deepeval_metrics import DeepEvalEvaluator

# Evaluate with RAGAS
ragas_evaluator = RAGEvaluator()
ragas_scores = ragas_evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)

# Evaluate with DeepEval
deepeval_evaluator = DeepEvalEvaluator()
deepeval_scores = deepeval_evaluator.evaluate_batch(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)

# Compare results
print("RAGAS Results:", ragas_evaluator.get_metrics_summary(ragas_scores))
print("DeepEval Results:", deepeval_evaluator.get_metrics_summary(deepeval_scores))
```

## 🔍 Evaluation Metrics

See **[METRICS_GUIDE.md](docs/METRICS_GUIDE.md)** for comprehensive explanations.

### RAGAS Framework (7 Metrics)
- **Quality**: Faithfulness, Answer Correctness
- **Relevancy**: Answer Relevancy, Context Relevancy
- **Retrieval**: Context Precision, Context Recall
- **Similarity**: Answer Similarity

### DeepEval Framework (14+ Metrics)
- **Core RAG**: Faithfulness, Answer Relevancy, Contextual Precision/Recall
- **Safety**: Hallucination, Bias, Toxicity Detection
- **Advanced**: G-Eval, Summarization, and custom metrics

## 🧪 Testing

The framework includes a comprehensive test suite. To run tests:

```bash
# Install all dependencies including dev tools (pytest, pytest-asyncio, pytest-cov)
uv sync

# Run tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ -v --cov=src
```

The `comprehensive_demo.py` serves as a living integration test that validates all features work together correctly.

## 📈 Monitoring & Observability

See **[PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)** for production monitoring details.

**Langfuse Dashboard** provides:
- Full execution traces of RAG queries
- Score tracking and trend analysis
- Performance and cost metrics (latency, token usage)
- Session grouping for batch evaluations

## 🔧 Configuration Options

See **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** for detailed configuration.

Edit `config.py` to customize NESGEN settings, RAG parameters, evaluation options, and Langfuse configuration.

## 📝 Adding Custom Documents

Edit `src/sample_data.py` to add your own documents:

```python
SAMPLE_DOCUMENTS = [
    "Your document 1 text...",
    "Your document 2 text...",
    # ...
]

SAMPLE_EVAL_DATA = [
    {
        "question": "Your question?",
        "ground_truth": "Expected answer..."
    },
    # ...
]
```

## 🆕 What's New

### Latest Features
- ✨ **Dual Framework Support**: Both RAGAS and DeepEval in one framework
- 🏭 **Production Evaluator**: Enterprise-ready evaluation with retry and rate limiting
- 🛡️ **Safety Metrics**: Bias, toxicity, and hallucination detection
- 📊 **Framework Comparison**: Built-in tools to compare RAGAS vs DeepEval
- 🔄 **Batch Processing**: Efficient evaluation of large datasets
- 💾 **Result Persistence**: Automatic storage of evaluation results
- 🎯 **NESGEN Integration**: Full support for Nestle GenAI platform

### Coming Soon
- 🔮 Custom metric builder UI
- 📈 Advanced analytics and reporting
- 🌐 Multi-language support
- 🔌 Additional LLM provider integrations

## 🤝 Contributing

**How to Contribute:**
1. Fork the repository
2. Create a feature branch
3. Run `uv sync` to install all dependencies (including dev tools)
4. Make your changes with tests
5. Run `uv run pytest tests/ -v` to verify
6. Submit a pull request
7. Update documentation as needed

## 🔗 Resources

### Framework Documentation
- [RAGAS Documentation](https://docs.ragas.io/) - Official RAGAS guide
- [DeepEval Documentation](https://docs.confident-ai.com/) - Official DeepEval guide
- [Langfuse Documentation](https://langfuse.com/docs) - Observability platform
- [LangChain Documentation](https://python.langchain.com/) - LLM framework

### APIs
- [NESGEN API](https://eur-sdr-int-pub.nestle.com) - Nestle GenAI Platform
- [OpenAI API](https://platform.openai.com/docs) - Fallback for embeddings

## ⚡ Performance & Best Practices

See **[PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)** for comprehensive best practices.

**Quick Tips:**
- Use DeepEval for faster evaluation
- Enable rate limiting to prevent API throttling
- Use batch processing for better throughput
- Choose metrics wisely based on your use case
- Monitor usage in Langfuse dashboard

## 💡 Tips

1. **Start Small**: Test with a few documents and questions first
2. **Use Ground Truth**: Enables more comprehensive metrics
3. **Monitor Costs**: API calls can add up, especially with evaluations
4. **Iterate**: Use evaluation results to improve your RAG pipeline
5. **Safety First**: Always include bias and toxicity checks for production
6. **Check Documentation**: Each guide covers a specific aspect in detail

## 📦 Dependency Management (uv)

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# Set up environment and install all dependencies
uv sync

# Run any script inside the managed environment
uv run python comprehensive_demo.py
uv run python batch_evaluation.py

# Add a new package
uv add some-package

# Add a dev-only package
uv add --dev some-package

# Update all packages
uv lock --upgrade

# Show installed packages
uv pip list
```

The `uv.lock` file is committed to the repository to ensure reproducible installs across all environments.

## 🐛 Troubleshooting

### Common Issues

**Authentication Issues**
- Ensure `.env` contains valid NESGEN credentials
- Check `NESGEN_CLIENT_ID`, `NESGEN_CLIENT_SECRET`, and `NESGEN_API_BASE`

**Langfuse Issues**
- Verify Langfuse credentials in `.env`
- Call `langfuse.flush()` at end of script
- Check network connectivity to Langfuse host

**Performance Issues**
- Use DeepEval instead of RAGAS (faster)
- Enable rate limiting in ProductionEvaluator
- Reduce number of questions or metrics

**Database Issues**
- Delete `chroma_db/` directory and reinitialize if errors occur

For more troubleshooting details, see the specific guides in the documentation.

## ❓ FAQ

**Q: Do I need both RAGAS and DeepEval?**
A: No, use either one or both. See [FRAMEWORK_COMPARISON.md](docs/FRAMEWORK_COMPARISON.md) for guidance.

**Q: Is Langfuse required?**
A: No, it's optional. Set `config.langfuse.enabled = False` to disable.

**Q: How much does evaluation cost?**
A: Costs depend on API usage. Each evaluation requires multiple LLM calls. Monitor in Langfuse.

**Q: Can I run this in production?**
A: Yes! Use `ProductionEvaluator` with rate limiting and retry logic. See [PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md).

For more FAQs, see the individual guide documents.

## 📧 Support

For issues or questions:
1. Check the [FAQ](#-frequently-asked-questions-faq) above
2. Review the `comprehensive_demo.py` file for code examples
3. Read the comprehensive guides in the [documentation section](#-documentation)
4. Check Langfuse, RAGAS, and DeepEval official documentation

---

## 💬 Community & Support

- 📚 **Documentation**: Check out our comprehensive guides in the `docs/` directory
- 💡 **Examples**: See `comprehensive_demo.py` for complete use cases
- 🐛 **Issues**: Report bugs or request features
- 🤝 **Contribute**: We welcome contributions!

**For Nestle & Purina Teams:**
- Internal documentation and support available through Nestle GenAI team
- Contact the AI/ML CoE for questions specific to Nestle/Purina use cases
- Leverage NESGEN platform capabilities for enterprise deployment

---
