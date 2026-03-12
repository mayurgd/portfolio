# NPUS AI AIOps IBM POC

A comprehensive proof-of-concept (POC) repository for RAG (Retrieval-Augmented Generation) evaluation using multiple frameworks and observability tools. This project demonstrates enterprise-ready RAG evaluation patterns specifically built for **NESGEN (Nestle GenAI)** platform with full observability and production-grade features.

---

## 📁 Project Structure

This repository is organized into two main components:

```
npus-ai-aiops-ibm-poc/
├── metric_evaluation/              # Production RAG evaluation framework
│   ├── src/                        # Core source code
│   ├── docs/                       # Comprehensive documentation
│   ├── examples/                   # Usage examples
│   ├── tests/                      # Test suite
│   ├── evaluation_results/         # Stored evaluation results
│   ├── logs/                       # Application logs
│   ├── penv/                       # Python virtual environment
│   ├── comprehensive_demo.py       # Main demonstration script
│   ├── config.py                   # Configuration management
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Detailed module documentation
│
├── RAG-evaluation-cookbooks/       # Jupyter notebook tutorials
│   ├── cookbooks/                  # Interactive notebooks
│   ├── docs/                       # Setup guides
│   ├── pyproject.toml              # Project configuration (uv)
│   ├── uv.lock                     # Dependency lock file
│   └── README.md                   # Cookbook documentation
│
└── README.md                       # This file
```

---

## 📚 Folder Descriptions

### 1. `metric_evaluation/` - Production RAG Evaluation Framework

**Purpose:** A production-ready, comprehensive RAG evaluation framework that supports multiple evaluation frameworks (RAGAS and DeepEval) with full Langfuse integration for observability.

**Key Features:**
- ✅ **Dual Framework Support**: Both RAGAS (7 metrics) and DeepEval (14+ metrics)
- ✅ **NESGEN Integration**: Built specifically for Nestle GenAI platform
- ✅ **Production-Ready**: Rate limiting, retry logic, error handling, and structured logging
- ✅ **Safety Metrics**: Bias, toxicity, and hallucination detection
- ✅ **Full Observability**: Complete Langfuse integration for tracing and monitoring
- ✅ **Batch Processing**: Efficient evaluation of large datasets
- ✅ **Result Persistence**: Automatic storage and CSV export

**What's Inside:**

#### `src/` - Core Source Code
- **`rag_pipeline.py`**: Complete RAG pipeline with document ingestion, retrieval, and generation
- **`nesgen_llm.py`**: NESGEN API integration and authentication
- **`evaluation_metrics.py`**: RAGAS evaluation framework implementation
- **`deepeval_metrics.py`**: DeepEval evaluation framework implementation
- **`sample_data.py`**: Sample documents and test datasets
- **`evaluation/`**: Production evaluator module
  - `evaluator.py` - Main orchestrator for batch evaluations
  - `metrics_calculator.py` - Metrics computation engine
  - `result_aggregator.py` - Results aggregation and analysis
  - `storage_handler.py` - Persistent storage management
- **`utils/`**: Utility modules
  - `logger.py` - Structured logging system
  - `rate_limiter.py` - API rate limiting
  - `retry_handler.py` - Retry logic with exponential backoff

#### `docs/` - Comprehensive Documentation
- **`QUICKSTART.md`**: 5-minute quick start guide
- **`METRICS_GUIDE.md`**: Detailed explanation of all 21+ evaluation metrics
- **`PRODUCTION_GUIDE.md`**: Production deployment best practices
- **`FRAMEWORK_COMPARISON.md`**: RAGAS vs DeepEval comparison and guidance
- **`ARCHITECTURE.md`**: System architecture and design details

#### `examples/` - Usage Examples
Example scripts demonstrating various use cases and integration patterns

#### `tests/` - Test Suite
Unit and integration tests for the evaluation framework

#### `evaluation_results/` - Stored Results
Persistent storage of evaluation results in CSV format for analysis

#### `logs/` - Application Logs
Structured logs for debugging, monitoring, and audit trails

#### `penv/` - Python Virtual Environment
Isolated Python environment with all dependencies installed

**Main Files:**
- **`comprehensive_demo.py`**: Complete demonstration showcasing all features
- **`config.py`**: Centralized configuration management
- **`requirements.txt`**: Python package dependencies

**Best For:**
- Production RAG system evaluation
- Comparing different RAG configurations
- Safety and ethics compliance (bias, toxicity detection)
- Large-scale batch evaluations
- Monitoring and observability with Langfuse
- Framework comparison studies

**Quick Start:**
```bash
cd metric_evaluation
pip install -r requirements.txt
python comprehensive_demo.py
```

---

### 2. `RAG-evaluation-cookbooks/` - Jupyter Notebook Tutorials

**Purpose:** A collection of interactive Jupyter notebooks demonstrating RAG evaluation patterns using Langfuse and RAGAS. These cookbooks provide hands-on, educational examples for learning and experimentation.

**Key Features:**
- 📓 **Interactive Notebooks**: Learn by doing with executable examples
- 🔬 **Experiment Tracking**: Structured experiments using Langfuse
- 📊 **RAGAS Integration**: Practical examples of RAGAS metrics
- 🎯 **Real Datasets**: Uses FIQA dataset for realistic examples
- 🔄 **Trace Evaluation**: Post-hoc evaluation of production traces

**What's Inside:**

#### `cookbooks/` - Interactive Notebooks
- **`experiment_ragas_metrics.ipynb`**: 
  - Run structured experiments with RAGAS metrics
  - Uses Langfuse's `run_experiment()` API
  - Ideal for CI/CD quality gates and A/B testing
  - Works with FIQA financial Q&A dataset

- **`trace_evaluation_with_ragas.ipynb`**:
  - Offline evaluation of existing Langfuse traces
  - Retrieve and analyze production traces
  - Apply RAGAS metrics retroactively
  - Useful for continuous monitoring

#### `docs/` - Setup Guides
- **`setup-langfuse.md`**: Langfuse account and API key setup
- **`setup-openai.md`**: OpenAI API configuration

**Configuration Files:**
- **`pyproject.toml`**: Project metadata and dependencies (uv package manager)
- **`uv.lock`**: Locked dependency versions for reproducibility

**Best For:**
- Learning RAG evaluation concepts
- Prototyping evaluation workflows
- Running ad-hoc experiments
- Educational purposes and training
- Quick experimentation with RAGAS
- Understanding Langfuse integration

**Quick Start:**
```bash
cd RAG-evaluation-cookbooks
pyenv local 3.12
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --extra notebook
cp .env.example .env  # Fill in your API keys
jupyter notebook
```

---

## 🎯 Which Component Should I Use?

### Use `metric_evaluation/` if you need:
- ✅ Production-ready evaluation framework
- ✅ Multiple evaluation frameworks (RAGAS + DeepEval)
- ✅ Safety and ethics metrics (bias, toxicity, hallucination)
- ✅ Batch processing and automation
- ✅ Rate limiting and retry logic
- ✅ NESGEN API integration
- ✅ Result persistence and CSV export
- ✅ Comprehensive documentation and guides

### Use `RAG-evaluation-cookbooks/` if you need:
- ✅ Interactive learning and experimentation
- ✅ Jupyter notebook tutorials
- ✅ Quick prototyping
- ✅ Understanding RAGAS concepts
- ✅ Langfuse experiment tracking
- ✅ Post-hoc trace evaluation
- ✅ Educational examples

### Use Both if you want:
- ✅ Learn concepts in notebooks, deploy with production framework
- ✅ Prototype in cookbooks, scale with metric_evaluation
- ✅ Complete understanding from theory to production

---

## 🚀 Getting Started

### Prerequisites

**Common Requirements:**
- Python 3.9+ (3.11+ recommended for cookbooks)
- NESGEN API credentials (for `metric_evaluation/`)
- OpenAI API key (optional, for embeddings fallback)
- Langfuse account (optional, for observability)

**Platform Support:**
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu 20.04+, Debian, WSL)

### Environment Setup

#### For metric_evaluation (Production Framework):
```bash
cd metric_evaluation
python -m venv penv
source penv/bin/activate  # Windows: penv\Scripts\activate
pip install -r requirements.txt

# Create .env file with credentials
# NESGEN_CLIENT_ID=your_client_id
# NESGEN_CLIENT_SECRET=your_client_secret
# LANGFUSE_SECRET_KEY=your_key (optional)
# LANGFUSE_PUBLIC_KEY=your_key (optional)

# Run comprehensive demo
python comprehensive_demo.py
```

#### For RAG-evaluation-cookbooks (Notebooks):
```bash
cd RAG-evaluation-cookbooks

# Install pyenv and uv (if not already installed)
curl https://pyenv.run | bash
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup Python and environment
pyenv local 3.12
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync --extra notebook

# Configure environment
cp .env.example .env  # Fill in your API keys

# Launch Jupyter
jupyter notebook
```

---

## 📊 Evaluation Metrics Overview

### RAGAS Framework (7 Metrics)
Available in both components:
- **Faithfulness**: Measures factual grounding (no hallucinations)
- **Answer Relevancy**: Relevance of answer to question
- **Context Precision**: Signal-to-noise ratio of retrieved contexts
- **Context Recall**: Completeness of context retrieval
- **Answer Similarity**: Semantic similarity to ground truth
- **Answer Correctness**: Accuracy compared to ground truth
- **Context Relevancy**: Relevance of contexts to question

### DeepEval Framework (14+ Metrics)
Available in `metric_evaluation/` only:
- **Core RAG**: Faithfulness, Answer Relevancy, Contextual Precision/Recall
- **Safety**: Hallucination detection, Bias detection, Toxicity detection
- **Advanced**: G-Eval, Summarization, RAG-AS, and custom metrics

---

## 🔍 Key Technologies

### Evaluation Frameworks
- **RAGAS**: Academically validated RAG evaluation metrics
- **DeepEval**: Fast, comprehensive evaluation with safety checks

### LLM & APIs
- **NESGEN**: Nestle GenAI platform (gpt-4.1)
- **OpenAI**: Fallback for embeddings and evaluation

### Observability
- **Langfuse**: Full LLM observability, tracing, and experiment tracking

### Vector Database
- **ChromaDB**: Vector storage for semantic retrieval

### Infrastructure
- **LangChain**: LLM application framework
- **Jupyter**: Interactive notebook environment
- **uv**: Fast Python package manager (cookbooks)

---

## 📖 Documentation

### metric_evaluation Documentation:
- [Comprehensive README](metric_evaluation/README.md)
- [Quick Start Guide](metric_evaluation/docs/QUICKSTART.md)
- [Metrics Guide](metric_evaluation/docs/METRICS_GUIDE.md)
- [Production Guide](metric_evaluation/docs/PRODUCTION_GUIDE.md)
- [Framework Comparison](metric_evaluation/docs/FRAMEWORK_COMPARISON.md)
- [Architecture Details](metric_evaluation/docs/ARCHITECTURE.md)

### RAG-evaluation-cookbooks Documentation:
- [Cookbooks README](RAG-evaluation-cookbooks/README.md)
- [Langfuse Setup](RAG-evaluation-cookbooks/docs/setup-langfuse.md)
- [OpenAI Setup](RAG-evaluation-cookbooks/docs/setup-openai.md)

---

## 🎓 Learning Path

**Recommended learning sequence:**

1. **Start with Cookbooks** (`RAG-evaluation-cookbooks/`)
   - Run `experiment_ragas_metrics.ipynb` to understand RAGAS basics
   - Explore `trace_evaluation_with_ragas.ipynb` for Langfuse integration
   - Experiment with different questions and datasets

2. **Move to Production Framework** (`metric_evaluation/`)
   - Read the [Quick Start Guide](metric_evaluation/docs/QUICKSTART.md)
   - Run `comprehensive_demo.py` to see all features
   - Review [Metrics Guide](metric_evaluation/docs/METRICS_GUIDE.md) for deeper understanding
   - Explore [Production Guide](metric_evaluation/docs/PRODUCTION_GUIDE.md) for deployment

3. **Advanced Topics**
   - Compare RAGAS vs DeepEval using [Framework Comparison](metric_evaluation/docs/FRAMEWORK_COMPARISON.md)
   - Study [Architecture](metric_evaluation/docs/ARCHITECTURE.md) for system design
   - Implement safety metrics (bias, toxicity, hallucination)
   - Build custom evaluation workflows

---

## 🤝 Contributing

Contributions are welcome! 


## 🔗 Additional Resources

### Framework Documentation
- [RAGAS Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [LangChain Documentation](https://python.langchain.com/)

### Platform APIs
- [NESGEN API](https://eur-sdr-int-pub.nestle.com)
- [OpenAI API](https://platform.openai.com/docs)

---

## 💡 Best Practices

1. **Start Small**: Test with a few documents first
2. **Use Ground Truth**: Enables comprehensive metrics
3. **Monitor Costs**: API calls can add up
4. **Enable Safety Metrics**: Always check for bias and toxicity in production
5. **Iterate**: Use evaluation results to improve your RAG pipeline
6. **Check Logs**: Comprehensive logging helps debug issues
7. **Leverage Langfuse**: Use observability for continuous improvement

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` or `uv sync`

**Authentication Issues**
- Check `.env` file has valid credentials
- Verify NESGEN_CLIENT_ID and NESGEN_CLIENT_SECRET

**Langfuse Connection**
- Verify Langfuse keys in `.env`
- Call `langfuse.flush()` at end of scripts
- Check network connectivity

**Performance Issues**
- Use DeepEval for faster evaluation
- Enable rate limiting to prevent throttling
- Reduce batch sizes for memory-constrained systems

---

## 📧 Support

For issues or questions:
1. Check the component-specific README files
2. Review the documentation in `docs/` folders
3. Check the comprehensive demo scripts
4. Consult official framework documentation

---

<div align="center">

## 🌟 Built for the Purina Nestle Community


- **Python: Current File** — Debug the active Python file
- **Python Debugger: Jupyter** — Debug Jupyter notebooks
- **Attach to Kernel** — Attach to a running Jupyter kernel
