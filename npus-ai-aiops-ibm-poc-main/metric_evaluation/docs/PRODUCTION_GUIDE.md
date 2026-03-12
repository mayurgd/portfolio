

# 🏭 Production Deployment Guide

Complete guide to deploying RAG evaluation in production environments.

---

## 🎯 Overview

This guide covers:
- Production architecture and code structure
- Configuration for different environments
- Monitoring and observability
- Security and best practices
- Scaling strategies

For basic setup, see **[QUICKSTART.md](QUICKSTART.md)**. For metrics details, see **[METRICS_GUIDE.md](METRICS_GUIDE.md)**.

---

## 📁 Production Code Structure

```
src/
├── evaluation/                      # Production evaluation system
│   ├── __init__.py
│   ├── evaluator.py                # Main evaluator with async support
│   ├── metrics_calculator.py       # Metrics calculation
│   ├── result_aggregator.py        # Statistical aggregation
│   └── storage_handler.py          # Result persistence
│
└── utils/                          # Production utilities
    ├── __init__.py
    ├── rate_limiter.py             # API rate limiting
    ├── retry_handler.py            # Retry with exponential backoff
    └── logger.py                   # Structured logging

examples/
└── production_evaluation.py         # Production example

tests/
└── test_production_evaluator.py    # Comprehensive tests
```

---

## 🚀 Quick Start (Production)

### 1. Basic Usage

```python
from src.evaluation import ProductionEvaluator

# Initialize
evaluator = ProductionEvaluator(
    framework="deepeval",
    max_concurrent=5,
    rate_limit_per_minute=100,
    enable_retry=True,
    enable_persistence=True
)

# Run evaluation (async)
result = await evaluator.evaluate_async(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths
)

# Or synchronous
result = evaluator.evaluate_sync(questions, answers, contexts)
```

### 2. Run Production Example

```bash
cd examples
python production_evaluation.py
```

---

## 🏗️ Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Production Evaluator                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Async Evaluation Engine                    │   │
│  │  • Concurrency Control (Semaphore)                   │   │
│  │  • Rate Limiting (Token Bucket)                      │   │
│  │  • Retry Logic (Exponential Backoff)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│       ┌───────────────────┼───────────────────┐            │
│       │                   │                   │              │
│  ┌────▼────┐      ┌──────▼────────┐   ┌────▼─────┐       │
│  │ Metrics │      │  Aggregator   │   │ Storage  │       │
│  │  Calc   │      │  (Stats)      │   │ Handler  │       │
│  └─────────┘      └───────────────┘   └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. **Async/Await Support**
- Non-blocking evaluation
- Concurrent processing
- Better resource utilization

#### 2. **Rate Limiting**
- Token bucket algorithm
- Prevents API throttling
- Configurable rates

#### 3. **Retry Logic**
- Exponential backoff
- Jitter for distributed systems
- Automatic error classification

#### 4. **Result Persistence**
- JSON file storage
- Async I/O
- Query capabilities

#### 5. **Progress Tracking**
- Real-time progress callbacks
- Completion percentage
- ETA calculation

#### 6. **Error Handling**
- Comprehensive try-catch
- Detailed error logging
- Graceful degradation

---

## ⚙️ Configuration

### Production Configuration

```python
evaluator = ProductionEvaluator(
    # Framework Selection
    framework="deepeval",              # "ragas" or "deepeval"
    
    # Performance
    max_concurrent=10,                 # Concurrent evaluations
    rate_limit_per_minute=200,         # API rate limit
    
    # Reliability
    enable_retry=True,                 # Enable retries
    max_retries=5,                     # Retry attempts
    
    # Persistence
    enable_persistence=True,           # Save results
    storage_path="/data/evaluations",  # Storage location
    
    # Logging
    log_level="INFO"                   # Log level
)
```

### Environment-Specific Configs

```python
# config/production.py
PRODUCTION_CONFIG = {
    "max_concurrent": 20,
    "rate_limit_per_minute": 500,
    "enable_retry": True,
    "max_retries": 5,
    "storage_path": "/data/production/evaluations",
    "log_level": "WARNING"
}

# config/staging.py
STAGING_CONFIG = {
    "max_concurrent": 10,
    "rate_limit_per_minute": 200,
    "enable_retry": True,
    "max_retries": 3,
    "storage_path": "/data/staging/evaluations",
    "log_level": "INFO"
}

# config/development.py
DEVELOPMENT_CONFIG = {
    "max_concurrent": 3,
    "rate_limit_per_minute": 100,
    "enable_retry": False,
    "max_retries": 1,
    "storage_path": "./evaluation_results",
    "log_level": "DEBUG"
}
```

---

## 📊 Monitoring & Observability

### Key Metrics to Track

Track these via Langfuse dashboard:

**Performance**: Latency, token usage, cost per query
**Quality**: Faithfulness trends, answer relevancy, user feedback
**System**: Error rates, success rates, throughput
**User**: Session metrics, satisfaction, feature usage

### Health Checks

```python
def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    stats = evaluator.get_statistics()
    
    return {
        "status": "healthy" if stats['success_rate'] > 0.95 else "degraded",
        "success_rate": stats['success_rate'],
        "total_evaluations": stats['total_evaluations'],
        "last_evaluation": get_last_evaluation_time(),
        "storage_available": check_storage_space()
    }
```

---

## 🔒 Security Best Practices

### Key Security Considerations

1. **API Key Management**: Use environment variables and secrets management
2. **Input Validation**: Built-in validation in ProductionEvaluator
3. **Rate Limiting**: Protects against API throttling and cost overruns

For detailed security patterns, refer to your organization's security guidelines.

---

## 📈 Scaling

### Horizontal Scaling

```python
# Run multiple evaluator instances
# Each with subset of data

import multiprocessing

def evaluate_batch(data_subset):
    evaluator = ProductionEvaluator(...)
    return evaluator.evaluate_sync(**data_subset)

# Split data across processes
with multiprocessing.Pool(4) as pool:
    results = pool.map(evaluate_batch, data_subsets)
```

### Vertical Scaling

```python
# Increase concurrent evaluations
evaluator = ProductionEvaluator(
    max_concurrent=50,  # Higher concurrency
    rate_limit_per_minute=1000  # Higher rate limit
)
```

### Cloud Deployment

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-evaluator
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: evaluator
        image: rag-evaluator:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: MAX_CONCURRENT
          value: "20"
        - name: RATE_LIMIT
          value: "500"
```

---

## 🧪 Testing

### Run Tests

```bash
# Test files are placeholders - to be implemented
# Run comprehensive demo as integration test
python comprehensive_demo.py
```

### Integration Testing

```python
# Note: Test files are placeholders
# Use comprehensive_demo.py as integration test

# Run the comprehensive demo
python comprehensive_demo.py
```

---

## 📊 Performance Optimization

### 1. Batch Size Tuning

```python
# Experiment with different batch sizes
for batch_size in [5, 10, 20, 50]:
    evaluator = ProductionEvaluator(max_concurrent=batch_size)
    start = time.time()
    result = evaluator.evaluate_sync(...)
    duration = time.time() - start
    print(f"Batch size {batch_size}: {duration:.2f}s")
```

### 2. Metric Selection

```python
# Only evaluate necessary metrics
result = await evaluator.evaluate_async(
    ...,
    metrics=["faithfulness", "answer_relevancy"]  # Faster
)
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def evaluate_cached(question_hash, answer_hash):
    """Cache evaluation results"""
    return evaluator.evaluate_sync(...)
```

---

## 💰 Cost Optimization

### Estimate Costs

```python
def estimate_evaluation_cost(
    num_questions: int,
    num_metrics: int,
    model: str = "gpt-4"
) -> float:
    """Estimate API costs"""
    # Rough estimates
    calls_per_question = num_metrics * 2  # Average
    total_calls = num_questions * calls_per_question
    
    cost_per_call = {
        "gpt-4": 0.01,
        "gpt-3.5-turbo": 0.001
    }
    
    return total_calls * cost_per_call[model]

# Example
cost = estimate_evaluation_cost(1000, 5, "gpt-4")
print(f"Estimated cost: ${cost:.2f}")
```

### Cost Reduction Strategies

1. **Use GPT-3.5 for development**
2. **Sample-based evaluation** (not all data)
3. **Cached results** for repeated evaluations
4. **Metric selection** (only necessary metrics)
5. **Batch processing** (more efficient)

---

## 🚨 Troubleshooting

**Rate limit errors**: Reduce `rate_limit_per_minute` or `max_concurrent`
**Timeout errors**: Enable retry logic, increase timeout
**Out of memory**: Reduce `max_concurrent`, process in smaller batches
**Slow evaluation**: Increase `max_concurrent`, use faster model (DeepEval)

For more troubleshooting, see **[README.md](../README.md)** troubleshooting section.

---

## ✅ Production Checklist

Before deploying:

- [ ] Environment-specific configuration
- [ ] Secrets management configured
- [ ] Logging set up (structured logs)
- [ ] Monitoring dashboards created
- [ ] Alerting rules defined
- [ ] Rate limits tested
- [ ] Error handling verified
- [ ] Retry logic tested
- [ ] Storage configured
- [ ] Backups enabled
- [ ] Load testing completed
- [ ] Security review done
- [ ] Documentation updated
- [ ] Runbooks created

---

## 📚 Next Steps

1. **Run the comprehensive demo**: `python comprehensive_demo.py`
2. **Review the code**: Study `src/evaluation/` modules
3. **Test the system**: Run `python comprehensive_demo.py` to validate
4. **Configure for your environment**: Adjust settings
5. **Deploy**: Follow your deployment process
6. **Monitor**: Set up dashboards and alerts

---
