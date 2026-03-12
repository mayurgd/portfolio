# 🚀 Quick Start Guide

Get up and running with RAG evaluation in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Keys (1 minute)

1. Copy the example environment file:
The `.env` file is already included in the repo. Just edit it:

2. Edit `.env` and add your keys:

```env
# NESGEN Configuration (Primary)
NESGEN_CLIENT_ID=your-client-id
NESGEN_CLIENT_SECRET=your-client-secret
NESGEN_API_BASE=https://eur-sdr-int-pub.nestle.com/api/dv-exp-sandbox-openai-api/1/genai/Azure
NESGEN_MODEL=gpt-4.1
NESGEN_API_VERSION=2024-02-01

# OpenAI API key (for embeddings fallback)
OPENAI_API_KEY=sk-your-key-here

# Langfuse Keys (optional but recommended)
LANGFUSE_SECRET_KEY=sk-lf-your-key-here
LANGFUSE_PUBLIC_KEY=pk-lf-your-key-here
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Get Langfuse keys:**
1. Go to [cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign up (free)
3. Create a project
4. Copy API keys from project settings

## Step 3: Run the Comprehensive Demo (2 minutes)

```bash
python comprehensive_demo.py
```

**You'll see:**
- ✅ RAG pipeline initialization with NESGEN
- ✅ Documents being ingested and chunked
- ✅ RAG queries with retrieval & generation
- ✅ RAGAS evaluation (7 metrics)
- ✅ DeepEval evaluation (14+ metrics with safety)
- ✅ Production async evaluation with rate limiting
- ✅ Configuration comparison (different top_k)
- ✅ Quality analysis with recommendations
- ✅ Framework comparison (RAGAS vs DeepEval)
- ✅ Results exported to CSV
- ✅ Everything logged to Langfuse

## Step 4: View Results

1. **Check Console Output**: Detailed results printed in terminal
2. **View CSV Export**: `comprehensive_evaluation_results.csv`
3. **Check Logs**: `./logs/comprehensive_demo.log`
4. **Persisted Results**: `./evaluation_results/` directory
5. **Langfuse Dashboard**: [cloud.langfuse.com](https://cloud.langfuse.com)
   - View traces and spans
   - Analyze score distributions
   - Track performance trends

## 🎉 That's it!

You now have a complete RAG evaluation system with:
- ✅ 21+ evaluation metrics (RAGAS + DeepEval)
- ✅ Production-grade features (async, retry, rate limiting)
- ✅ Safety metrics (bias, toxicity, hallucination)
- ✅ Full Langfuse observability
- ✅ Configuration comparison tools
- ✅ Quality analysis & recommendations
- ✅ CSV export for further analysis

## What's Next?

### Add Your Own Documents
Edit `src/sample_data.py`:
```python
SAMPLE_DOCUMENTS = [
    "Your custom document text here...",
    "Another document...",
]

SAMPLE_EVAL_DATA = [
    {
        "question": "Your question?",
        "ground_truth": "Expected answer..."
    },
]
```

### Customize Configuration
Edit `config.py` to adjust:
- Model selection (NESGEN model)
- Chunk sizes and overlap
- Retrieval parameters (top_k)
- Evaluation metrics selection
- Rate limits and retry settings

### Integrate Into Your Application
```python
from src.rag_pipeline import RAGPipeline
from src.evaluation import ProductionEvaluator

# Initialize
rag = RAGPipeline(collection_name="my_app")
evaluator = ProductionEvaluator(framework="both")

# Use in your code
result = rag.query("Your question?")
scores = await evaluator.evaluate_async(
    questions=[result['question']],
    answers=[result['answer']],
    contexts=[result['contexts']]
)
```

### Run Specific Evaluations
```python
# RAGAS only
from src.evaluation_metrics import RAGEvaluator
evaluator = RAGEvaluator()
scores = evaluator.evaluate(...)

# DeepEval only (faster, with safety metrics)
from src.deepeval_metrics import DeepEvalEvaluator
evaluator = DeepEvalEvaluator()
scores = evaluator.evaluate(...)
```

## Need Help?

- 📚 Read the full [README.md](../README.md)
- 📊 Check [METRICS_GUIDE.md](METRICS_GUIDE.md) for metric details
- 🏭 See [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) for deployment
- ⚖️ Read [FRAMEWORK_COMPARISON.md](FRAMEWORK_COMPARISON.md) for RAGAS vs DeepEval

## Common Issues

**Issue**: Authentication errors
- **Fix**: Check your NESGEN credentials in `.env`

**Issue**: Langfuse not logging
- **Fix**: Verify LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY

**Issue**: Evaluation too slow
- **Fix**: Use DeepEval instead of RAGAS (faster)
- **Fix**: Reduce number of questions or metrics

**Issue**: Rate limit errors
- **Fix**: Adjust `rate_limit_per_minute` in ProductionEvaluator

---
