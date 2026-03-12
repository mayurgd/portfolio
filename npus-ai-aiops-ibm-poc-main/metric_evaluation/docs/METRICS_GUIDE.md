# 📊 RAG Evaluation Metrics - Complete Guide

This guide explains all 7 evaluation metrics implemented in this framework using the RAGAS library.

## Overview

RAG (Retrieval-Augmented Generation) evaluation requires measuring multiple aspects:
- **Generation Quality**: Is the answer good?
- **Retrieval Quality**: Did we retrieve the right information?
- **Relevance**: Is everything relevant to the question?
- **Correctness**: Is the answer factually correct?

## The 7 Metrics

### 1. 📝 Faithfulness (Groundedness)

**What it measures**: Whether the answer is supported by the retrieved contexts.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Detects hallucinations - when the model makes up information not in the contexts.

**Example**:
- **Context**: "Python was created in 1991."
- **Question**: "When was Python created?"
- **Good Answer** (High Faithfulness): "Python was created in 1991."
- **Bad Answer** (Low Faithfulness): "Python was created in 1995 by Microsoft."

**How it's calculated**:
- Extracts claims from the answer
- Checks if each claim can be inferred from contexts
- Score = (supported claims) / (total claims)

**When to use**: Always! This is crucial for preventing hallucinations.

---

### 2. 🎯 Answer Relevancy

**What it measures**: How well the answer addresses the question.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Ensures the answer is on-topic and directly addresses what was asked.

**Example**:
- **Question**: "What is the capital of France?"
- **Good Answer** (High Relevancy): "The capital of France is Paris."
- **Bad Answer** (Low Relevancy): "France is a country in Europe with many cities and a rich history..."

**How it's calculated**:
- Generates questions from the answer
- Measures similarity between original question and generated questions
- More similar = more relevant

**When to use**: Always! Helps ensure your RAG gives focused, relevant answers.

---

### 3. 🔍 Context Precision

**What it measures**: What proportion of retrieved contexts are actually relevant to the question.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: High precision means your retrieval system isn't bringing back noise.

**Example**:
- **Question**: "How do I install Python?"
- **Retrieved Contexts**:
  1. ✅ "To install Python, download from python.org..." (Relevant)
  2. ✅ "Python installation requires..." (Relevant)
  3. ❌ "Python was created by Guido..." (Not relevant to installation)
  4. ❌ "Python is used for web development..." (Not relevant to installation)
- **Score**: 2/4 = 0.5

**How it's calculated**:
- Checks which retrieved contexts are actually relevant to the question
- Score = (relevant contexts) / (total retrieved contexts)

**When to use**: To evaluate and improve your retrieval system.

---

### 4. 📚 Context Recall

**What it measures**: Whether all necessary information from the ground truth was retrieved.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Low recall means important information is missing from retrieval.

**Example**:
- **Question**: "What are the benefits of exercise?"
- **Ground Truth**: "Exercise improves cardiovascular health, mental well-being, and strength."
- **Retrieved Contexts**: Only mention cardiovascular health and strength (missing mental well-being)
- **Score**: 2/3 = 0.67

**How it's calculated**:
- Extracts facts from ground truth
- Checks how many can be attributed to retrieved contexts
- Score = (attributed facts) / (total facts in ground truth)

**When to use**: When you have ground truth answers. Essential for improving retrieval.

**Note**: Requires ground truth answers!

---

### 5. 🔄 Answer Similarity

**What it measures**: Semantic similarity between generated answer and ground truth.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Helps evaluate if your answer conveys the same meaning as the ground truth.

**Example**:
- **Ground Truth**: "Machine learning is AI that learns from data."
- **High Similarity**: "ML is a type of AI that learns patterns from data."
- **Low Similarity**: "Algorithms are used in computer science."

**How it's calculated**:
- Uses embeddings to measure semantic similarity
- Considers meaning, not just word overlap

**When to use**: When you have ground truth answers and want to measure semantic correctness.

**Note**: Requires ground truth answers!

---

### 6. ✅ Answer Correctness

**What it measures**: Combination of factual correctness and semantic similarity.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Most comprehensive answer quality metric.

**Example**:
- **Ground Truth**: "Python was created in 1991 by Guido van Rossum."
- **Good Answer** (High Correctness): "Python was created in 1991 by Guido van Rossum."
- **Partially Correct**: "Python was created by Guido van Rossum." (missing year)
- **Wrong Answer**: "Python was created in 1995."

**How it's calculated**:
- Combines semantic similarity with factual overlap
- Weighted average of similarity and F1 score

**When to use**: When you have ground truth and want the most comprehensive quality metric.

**Note**: Requires ground truth answers!

---

### 7. 🎓 Context Relevancy

**What it measures**: How relevant the retrieved contexts are to the question.

**Scale**: 0 to 1 (higher is better)

**Why it matters**: Similar to Context Precision but focuses more on semantic relevance.

**Example**:
- **Question**: "What are the symptoms of flu?"
- **High Relevancy Context**: "Flu symptoms include fever, cough, and fatigue."
- **Low Relevancy Context**: "Flu is caused by influenza virus." (Related but doesn't answer question)

**How it's calculated**:
- Extracts sentences from contexts
- Determines which sentences are relevant to question
- Score = (relevant sentences) / (total sentences)

**When to use**: To evaluate retrieval quality and relevance.

---

## Metric Dependencies

| Metric | Requires Ground Truth? | Requires Contexts? | Requires Answer? |
|--------|----------------------|-------------------|-----------------|
| Faithfulness | ❌ No | ✅ Yes | ✅ Yes |
| Answer Relevancy | ❌ No | ✅ Yes | ✅ Yes |
| Context Precision | ✅ Yes | ✅ Yes | ❌ No |
| Context Recall | ✅ Yes | ✅ Yes | ❌ No |
| Answer Similarity | ✅ Yes | ❌ No | ✅ Yes |
| Answer Correctness | ✅ Yes | ❌ No | ✅ Yes |
| Context Relevancy | ❌ No | ✅ Yes | ❌ No |

## Which Metrics Should I Use?

### Minimum Set (No Ground Truth)
If you don't have ground truth answers:
- ✅ Faithfulness
- ✅ Answer Relevancy
- ✅ Context Relevancy

### Complete Set (With Ground Truth)
If you have ground truth answers:
- ✅ All 7 metrics!

### For Retrieval Optimization
Focus on:
- ✅ Context Precision
- ✅ Context Recall
- ✅ Context Relevancy

### For Generation Optimization
Focus on:
- ✅ Faithfulness
- ✅ Answer Relevancy
- ✅ Answer Correctness

## Interpreting Scores

### Excellent (0.9 - 1.0)
- System is performing very well
- Minimal issues

### Good (0.7 - 0.9)
- System is working well
- Some room for improvement

### Fair (0.5 - 0.7)
- System needs improvement
- Review retrieval or generation settings

### Poor (< 0.5)
- System has significant issues
- Requires debugging and optimization

## Common Issues and Solutions

### Low Faithfulness
**Problem**: Model is hallucinating
**Solutions**:
- Lower temperature
- Better prompt engineering
- Stronger grounding in system prompt
- Use RAG instead of pure generation

### Low Answer Relevancy
**Problem**: Answers are off-topic
**Solutions**:
- Improve prompt to focus on question
- Better retrieval to get relevant context
- Add examples in prompt

### Low Context Precision
**Problem**: Too much irrelevant information retrieved
**Solutions**:
- Adjust retrieval threshold
- Better chunking strategy
- Improve embedding model
- Use reranking

### Low Context Recall
**Problem**: Missing important information
**Solutions**:
- Increase top-k retrieval
- Better chunking (smaller chunks)
- Improve document coverage
- Better query reformulation

## Example Usage in Code

### Evaluate All Metrics
```python
from src.evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()

scores = evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths,
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
```

### Evaluate Without Ground Truth
```python
scores = evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    metrics=[
        "faithfulness",
        "answer_relevancy",
        "context_relevancy"
    ]
)
```

### Focus on Retrieval
```python
scores = evaluator.evaluate(
    questions=questions, answers=answers,
    contexts=contexts, ground_truths=ground_truths,
    metrics=["context_precision", "context_recall", "context_relevancy"]
)
```

---

## Best Practices

1. **Always track Faithfulness**: Critical for preventing hallucinations
2. **Use ground truth when possible**: Enables comprehensive evaluation
3. **Monitor trends**: Track metrics over time in Langfuse
4. **Test different configurations**: Use metrics to optimize your RAG system
5. **Balance metrics**: Don't optimize for just one metric
6. **User feedback**: Metrics are proxies; real user feedback matters too

For implementation details, see the comprehensive demo and other guides.

---

