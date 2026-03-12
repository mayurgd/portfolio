# 🎯 Use-Case Based Evaluation Guide: RAGAS vs DeepEval

**Quick reference for choosing the right evaluation metrics based on your use case**

---

## 📋 Quick Decision Matrix

| Your Goal | Framework | Key Metrics | Why |
|-----------|-----------|-------------|-----|
| Prevent AI hallucinations | RAGAS | `faithfulness` | Checks if answers stick to retrieved facts |
| Detect made-up information | DeepEval | `hallucination` | More detailed than RAGAS faithfulness |
| Code generation from docs | DeepEval | `faithfulness`, `hallucination` | Catch fake APIs/libraries |
| Pet products/E-commerce | DeepEval | `faithfulness`, `hallucination`, `bias` | Safety + fair recommendations |
| Ensure relevant answers | Both | `answer_relevancy` | Confirms answers address the question |
| Optimize retrieval quality | RAGAS | `context_precision`, `context_recall` | Measures retrieval effectiveness |
| Check for bias/toxicity | DeepEval | `bias`, `toxicity` | Safety-critical applications |
| Academic/research work | RAGAS | All metrics | Published methodology, citations |
| Production deployment | DeepEval | All metrics | Faster, more comprehensive |

---

## 🎪 Real-World Use Cases

### Use Case 1: Customer Support Chatbot
**Scenario**: Building a chatbot for product documentation

**Primary Concerns**:
- ✅ No hallucinations (customers need accurate info)
- ✅ Relevant answers (not rambling)
- ✅ Professional tone (no toxicity)

**Recommended Setup**:

```python
# DeepEval (Production-ready)
metrics = [
    "faithfulness",      # Must be factually correct
    "answer_relevancy",  # Stay on topic
    "toxicity"          # Professional responses
]
threshold = 0.8  # High bar for customer-facing
```

**Why DeepEval**: Built-in safety metrics + faster evaluation

---

### Use Case 2: Medical/Legal RAG System
**Scenario**: AI assistant for medical records or legal documents

**Primary Concerns**:
- ✅ Absolute accuracy (no room for error)
- ✅ Complete information retrieval
- ✅ No bias in responses

**Recommended Setup**:

```python
# Use BOTH for validation
# RAGAS
ragas_metrics = [
    "faithfulness",
    "context_recall",     # Don't miss critical info
    "answer_correctness"
]

# DeepEval
deepeval_metrics = [
    "faithfulness",
    "hallucination",
    "bias"               # Critical for fairness
]
threshold = 0.95  # Extremely high bar
```

**Why Both**: Cross-validate critical applications

---

### Use Case 3: Educational Content Generator
**Scenario**: RAG system generating study materials

**Primary Concerns**:
- ✅ Accurate information
- ✅ Complete coverage of topics
- ✅ Age-appropriate content

**Recommended Setup**:

```python
# RAGAS (Educational rigor)
metrics = [
    "faithfulness",
    "context_recall",        # Cover all key points
    "answer_correctness"     # Validate against ground truth
]
# Need ground truth answers for this use case
```

**Why RAGAS**: Academic validation, works well with ground truth

---

### Use Case 4: Code Generation Assistant
**Scenario**: RAG system generating code from documentation (Copilot-style)

**Primary Concerns**:
- ✅ No hallucinated libraries/functions
- ✅ Code matches documentation context
- ✅ Relevant to coding question

**Recommended Setup**:

```python
# DeepEval (Strict hallucination check)
metrics = [
    "faithfulness",      # Code must match docs
    "hallucination",     # Catch fake APIs/libraries
    "answer_relevancy"   # Solve the actual problem
]
threshold = 0.85  # High accuracy for executable code
```

**Why DeepEval**: Separate hallucination metric critical for catching non-existent functions/libraries

---

### Use Case 5: Research Paper Assistant
**Scenario**: Helping researchers find and summarize papers

**Primary Concerns**:
- ✅ Precise context retrieval
- ✅ No information loss
- ✅ Semantic accuracy

**Recommended Setup**:

```python
# RAGAS (Research-focused)
metrics = [
    "context_precision",    # Relevant papers only
    "context_recall",       # Don't miss key papers
    "answer_similarity"     # Semantic matching
]
```

**Why RAGAS**: Research-backed methodology, good for academic use

---

### Use Case 6: Real-time FAQ Bot
**Scenario**: High-traffic FAQ system needing fast responses

**Primary Concerns**:
- ✅ Speed (low latency)
- ✅ Relevance
- ✅ Basic accuracy

**Recommended Setup**:

```python
# DeepEval (Performance-optimized)
metrics = [
    "answer_relevancy",     # Quick relevance check only
]
threshold = 0.7  # Balanced threshold
```

**Why DeepEval**: Faster evaluation, single metric for speed

---

### Use Case 7: Content Moderation System
**Scenario**: Reviewing AI-generated public-facing content

**Primary Concerns**:
- ✅ No toxic content
- ✅ No biased responses
- ✅ Factual accuracy

**Recommended Setup**:

```python
# DeepEval (Safety-first)
metrics = [
    "toxicity",         # Primary concern
    "bias",             # Fairness check
    "faithfulness"      # Accuracy check
]
threshold = 0.9  # Strict safety requirements
```

**Why DeepEval**: Only framework with built-in safety metrics

---

### Use Case 8: Pet Products E-commerce Assistant
**Scenario**: RAG system for pet food, supplies, and care recommendations

**Primary Concerns**:
- ✅ Accurate product information (ingredients, safety)
- ✅ No harmful recommendations (pet safety critical)
- ✅ Relevant product suggestions
- ✅ Unbiased recommendations (no brand bias)

**Recommended Setup**:

```python
# DeepEval (Safety + Accuracy)
metrics = [
    "faithfulness",      # Accurate product info
    "hallucination",     # No fake ingredients/claims
    "answer_relevancy",  # Match pet owner's needs
    "bias"              # Fair product recommendations
]
threshold = 0.85  # High bar for pet safety
```

**Why DeepEval**: Pet safety requires hallucination detection + bias checking for fair recommendations

**Specific Considerations**:
- Monitor for dietary/allergy misinformation
- Verify ingredient accuracy against product database
- Check for breed-specific recommendations
- Ensure age-appropriate product suggestions (puppy vs senior)

---

## 🔧 Metric Selection Guide

### When You DON'T Have Ground Truth Answers

**RAGAS Options**:
- `faithfulness` - Is answer supported by context?
- `answer_relevancy` - Does answer address the question?

**DeepEval Options**:
- `faithfulness` - Context-supported answers
- `answer_relevancy` - Question alignment
- `hallucination` - Dedicated hallucination detection
- `bias` - Bias detection
- `toxicity` - Safety check

### When You HAVE Ground Truth Answers

**RAGAS Additional Options**:
- `context_precision` - Quality of retrieved contexts
- `context_recall` - Completeness of retrieval
- `answer_correctness` - Overall answer quality
- `answer_similarity` - Semantic similarity to truth

**DeepEval Additional Options**:
- `contextual_precision` - Context relevance (with ground truth)
- `contextual_recall` - Retrieval completeness (with ground truth)

---

## ⚙️ Configuration Recommendations

### Development/Testing Phase
```python
# RAGAS - Thorough evaluation
metrics = ["faithfulness", "answer_relevancy", "context_precision"]
threshold = 0.6  # Lower bar while iterating
```

### Pre-Production Validation
```python
# DeepEval - Comprehensive check
metrics = [
    "faithfulness", "answer_relevancy", 
    "contextual_precision", "hallucination", "bias"
]
threshold = 0.8  # Higher bar before launch
```

### Production Monitoring
```python
# DeepEval - Fast, focused
metrics = ["faithfulness", "answer_relevancy"]
threshold = 0.75  # Balanced for production
```

---

## 📊 Threshold Guidelines

| Application Type | Recommended Threshold | Rationale |
|-----------------|----------------------|-----------|
| Experimental/POC | 0.5 - 0.6 | Learning phase |
| Internal tools | 0.6 - 0.7 | Good enough for internal use |
| Public-facing | 0.7 - 0.8 | User-facing quality |
| Safety-critical | 0.85 - 0.95 | Medical, legal, financial |

---

## 🚀 Quick Start Examples

### Scenario: "I just need to know if my RAG works"
```python
from src.deepeval_metrics import DeepEvalEvaluator

evaluator = DeepEvalEvaluator(threshold=0.7)
results = evaluator.evaluate_batch(
    questions=questions,
    answers=answers,
    contexts=contexts,
    metrics=["faithfulness", "answer_relevancy"]
)
```

### Scenario: "I need detailed analysis with ground truth"
```python
from src.evaluation_metrics import RAGEvaluator

evaluator = RAGEvaluator()
results = evaluator.evaluate(
    questions=questions,
    answers=answers,
    contexts=contexts,
    ground_truths=ground_truths,
    metrics=[
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
        "answer_correctness"
    ]
)
```

### Scenario: "I need to check for safety issues"
```python
from src.deepeval_metrics import DeepEvalEvaluator

evaluator = DeepEvalEvaluator(threshold=0.9)
results = evaluator.evaluate_batch(
    questions=questions,
    answers=answers,
    contexts=contexts,
    metrics=["toxicity", "bias", "hallucination"]
)
```

### Scenario: "Pet products - need accuracy + safety + fair recommendations"
```python
from src.deepeval_metrics import DeepEvalEvaluator

evaluator = DeepEvalEvaluator(threshold=0.85)
results = evaluator.evaluate_batch(
    questions=questions,  # e.g., "What's the best food for senior dogs?"
    answers=answers,
    contexts=contexts,    # Product descriptions, care guides
    metrics=[
        "faithfulness",     # Accurate product info
        "hallucination",    # No fake ingredients
        "answer_relevancy", # Match pet needs
        "bias"             # Fair brand recommendations
    ]
)
```

---

## ✅ Final Recommendation by Use Case

| If you're building... | Use | Key Metrics |
|----------------------|-----|-------------|
| MVP/Prototype | DeepEval | `faithfulness`, `answer_relevancy` |
| Customer-facing app | DeepEval | `faithfulness`, `answer_relevancy`, `toxicity` |
| Pet Products/E-commerce | DeepEval | `faithfulness`, `hallucination`, `answer_relevancy`, `bias` |
| Academic project | RAGAS | All 6 metrics with ground truth |
| Healthcare/Legal | Both | All metrics, cross-validate |
| Internal tool | RAGAS | `faithfulness`, `answer_relevancy`, `context_precision` |
| High-traffic system | DeepEval | `answer_relevancy` (fast) |

---

