"""
RAG Evaluation Metrics using DeepEval framework

DeepEval provides a more comprehensive set of metrics with better performance
and customization options compared to RAGAS.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import os
from datetime import datetime
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)
# Note: BiasMetric and ToxicityMetric are intentionally excluded.
# Their built-in evaluation templates contain politically sensitive example text
# ("Hitler hated jews...") that triggers NESGEN's content policy filter (HTTP 400).
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langfuse import Langfuse
from config import config
from src.nesgen_llm import NESGENChatModel
from langchain_core.messages import SystemMessage, HumanMessage


class NESGENDeepEvalModel(DeepEvalBaseLLM):
    """
    NESGEN LLM wrapper compatible with DeepEval
    
    This wrapper allows DeepEval to use NESGEN API instead of OpenAI
    """
    
    def __init__(self):
        """Initialize NESGEN model for DeepEval"""
        self.nesgen_model = NESGENChatModel()
        self.model_name = config.nesgen.model
    
    def load_model(self):
        """Load model (already initialized in __init__)"""
        return self.nesgen_model
    
    def generate(self, prompt: str) -> str:
        """
        Generate response using NESGEN
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text response
        """
        try:
            # Convert prompt to LangChain message format
            messages = [HumanMessage(content=prompt)]
            
            # Use invoke() instead of _generate() for proper request handling
            response = self.nesgen_model.invoke(messages)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            print(f"Error generating with NESGEN: {e}")
            raise
    
    async def a_generate(self, prompt: str) -> str:
        """
        Async generate response using NESGEN
        
        Args:
            prompt: The prompt text
            
        Returns:
            Generated text response
        """
        # For now, delegate to sync version
        # You can implement true async later if needed
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        """Return the model name"""
        return self.model_name


class DeepEvalEvaluator:
    """
    Comprehensive RAG Evaluation using DeepEval metrics
    
    DeepEval advantages:
    - More metrics (14+ vs 7 in RAGAS)
    - Faster evaluation
    - Better customization
    - Built-in testing framework
    - Synthetic data generation
    """
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize DeepEval Evaluator
        
        Args:
            threshold: Minimum score threshold for passing (0.0-1.0)
        """
        self.threshold = threshold
        
        # Initialize NESGEN model for DeepEval
        print("🔵 Initializing DeepEval with NESGEN model...")
        self.nesgen_model = NESGENDeepEvalModel()
        print(f"✓ Using NESGEN model: {config.nesgen.model}")
        
        # Initialize Langfuse with proper credential check
        if config.langfuse.is_available():
            try:
                self.langfuse = Langfuse(
                    secret_key=config.langfuse.secret_key,
                    public_key=config.langfuse.public_key,
                    host=config.langfuse.host
                )
                print("✓ Langfuse initialized for DeepEval")
                self.use_file_fallback = False
            except Exception as e:
                print(f"⚠ Langfuse initialization failed: {e}")
                print("  Results will be saved to local files instead")
                self.langfuse = None
                self.use_file_fallback = True
        else:
            print("⚠ Langfuse not available - credentials not configured")
            print("  Results will be saved to local files instead")
            self.langfuse = None
            self.use_file_fallback = True
        
        # Create output directory for file fallback
        if self.use_file_fallback:
            self.output_dir = "evaluation_results"
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Define available metrics with their initializers
        # Use the custom NESGEN model instead of string model name
        self.available_metrics = {
            "faithfulness": lambda: FaithfulnessMetric(
                threshold=self.threshold,
                model=self.nesgen_model
            ),
            "answer_relevancy": lambda: AnswerRelevancyMetric(
                threshold=self.threshold,
                model=self.nesgen_model
            ),
            "contextual_precision": lambda: ContextualPrecisionMetric(
                threshold=self.threshold,
                model=self.nesgen_model
            ),
            "contextual_recall": lambda: ContextualRecallMetric(
                threshold=self.threshold,
                model=self.nesgen_model
            ),
            "hallucination": lambda: HallucinationMetric(
                threshold=self.threshold,
                model=self.nesgen_model
            ),
        }
    
    def create_test_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> LLMTestCase:
        """
        Create a DeepEval test case
        
        Args:
            question: Input question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer (optional)
            
        Returns:
            LLMTestCase object
        """
        return LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
            expected_output=ground_truth if ground_truth else ""
        )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            metrics: List of metric names to evaluate
            
        Returns:
            Dictionary with metric scores
        """
        # Select metrics
        if metrics is None:
            metrics = config.evaluation.deepeval_metrics
        
        # Create test case
        test_case = self.create_test_case(question, answer, contexts, ground_truth)
        
        # Initialize metrics
        metric_objects = []
        for metric_name in metrics:
            if metric_name in self.available_metrics:
                metric_objects.append(self.available_metrics[metric_name]())
        
        # Evaluate
        results = {}
        for metric in metric_objects:
            try:
                metric.measure(test_case)
                results[metric.__class__.__name__.replace('Metric', '').lower()] = {
                    'score': metric.score,
                    'reason': metric.reason,
                    'success': metric.is_successful()
                }
            except Exception as e:
                print(f"Warning: Could not evaluate {metric.__class__.__name__}: {e}")
                results[metric.__class__.__name__.replace('Metric', '').lower()] = {
                    'score': 0.0,
                    'reason': f"Error: {str(e)}",
                    'success': False
                }
        
        return results
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts for each question
            ground_truths: Optional list of ground truth answers
            metrics: List of metric names to evaluate
            
        Returns:
            Dictionary with aggregated results
        """
        # Create test cases
        test_cases = []
        for i in range(len(questions)):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            test_case = self.create_test_case(
                question=questions[i],
                answer=answers[i],
                contexts=contexts[i],
                ground_truth=gt
            )
            test_cases.append(test_case)
        
        # Select metrics
        if metrics is None:
            metrics = config.evaluation.deepeval_metrics
        
        # Initialize metrics
        metric_objects = []
        for metric_name in metrics:
            if metric_name in self.available_metrics:
                metric_objects.append(self.available_metrics[metric_name]())
        
        # Evaluate all test cases
        all_results = []
        for test_case in test_cases:
            case_results = {}
            for metric in metric_objects:
                try:
                    metric.measure(test_case)
                    metric_name = metric.__class__.__name__.replace('Metric', '').lower()
                    case_results[metric_name] = metric.score
                except Exception as e:
                    metric_name = metric.__class__.__name__.replace('Metric', '').lower()
                    case_results[metric_name] = 0.0
            all_results.append(case_results)
        
        # Aggregate results (average across all test cases)
        aggregated = {}
        if all_results:
            for metric_name in all_results[0].keys():
                scores = [r[metric_name] for r in all_results if metric_name in r]
                aggregated[metric_name] = sum(scores) / len(scores) if scores else 0.0
        
        # Log to Langfuse or save to file
        if self.langfuse:
            self._log_to_langfuse(aggregated, questions, answers, contexts)
        elif self.use_file_fallback:
            self._save_to_file(aggregated, questions, answers, contexts)
        
        return aggregated
    
    def evaluate_with_dataframe(
        self,
        eval_data: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate batch and return detailed DataFrame
        
        Args:
            eval_data: List of dicts with 'question', 'answer', 'contexts', 'ground_truth'
            metrics: List of metric names to evaluate
            
        Returns:
            DataFrame with results for each question
        """
        results = []
        
        for item in eval_data:
            result = self.evaluate_single(
                question=item['question'],
                answer=item['answer'],
                contexts=item['contexts'],
                ground_truth=item.get('ground_truth'),
                metrics=metrics
            )
            
            # Flatten results
            row = {
                'question': item['question'],
                'answer': item['answer']
            }
            
            for metric_name, metric_data in result.items():
                row[f'{metric_name}_score'] = metric_data['score']
                row[f'{metric_name}_success'] = metric_data['success']
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _log_to_langfuse(
        self,
        results: Dict[str, float],
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ):
        """Log evaluation results to Langfuse"""
        try:
            # Create a trace for evaluation
            trace = self.langfuse.trace(
                name="deepeval_rag_evaluation",
                metadata={
                    "framework": "deepeval",
                    "num_questions": len(questions),
                    "metrics": list(results.keys())
                }
            )
            
            # Log overall scores
            for metric_name, score in results.items():
                if isinstance(score, (int, float)):
                    trace.score(
                        name=f"deepeval_{metric_name}",
                        value=score,
                        comment=f"DeepEval evaluation metric: {metric_name}"
                    )
            
            # Log individual examples (first 5)
            for i, (q, a, c) in enumerate(zip(questions[:5], answers[:5], contexts[:5])):
                trace.span(
                    name=f"example_{i}",
                    input={"question": q, "contexts": c},
                    output={"answer": a}
                )
            
            print("✓ DeepEval results logged to Langfuse")
        except Exception as e:
            print(f"Warning: Could not log to Langfuse: {e}")
            if self.use_file_fallback:
                self._save_to_file(results, questions, answers, contexts)
    
    def _save_to_file(
        self,
        results: Dict[str, float],
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]]
    ):
        """Save evaluation results to a JSON file as fallback"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"deepeval_results_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            output_data = {
                "framework": "deepeval",
                "timestamp": timestamp,
                "num_questions": len(questions),
                "metrics": results,
                "examples": [
                    {
                        "question": q,
                        "answer": a,
                        "contexts": c
                    }
                    for q, a, c in zip(questions[:5], answers[:5], contexts[:5])
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ DeepEval results saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
    
    def get_metrics_summary(self, results: Dict[str, Any]) -> str:
        """
        Get a formatted summary of evaluation metrics
        
        Args:
            results: Dictionary of metric scores
            
        Returns:
            Formatted string summary
        """
        summary = "\n" + "="*60 + "\n"
        summary += "DEEPEVAL RAG EVALUATION RESULTS\n"
        summary += "="*60 + "\n\n"
        
        # Group metrics by category
        quality_metrics = ["faithfulness", "hallucination"]
        relevancy_metrics = ["answerrelevancy", "answer_relevancy"]
        retrieval_metrics = ["contextualprecision", "contextualrecall", 
                            "contextual_precision", "contextual_recall"]
        safety_metrics = ["bias", "toxicity"]
        
        def print_metrics(metric_list, title):
            s = f"{title}:\n"
            found_any = False
            for metric in metric_list:
                if metric in results:
                    score = results[metric]
                    if isinstance(score, (int, float)):
                        s += f"  • {metric.replace('_', ' ').title()}: {score:.4f}\n"
                        found_any = True
            return s if found_any else ""
        
        summary += print_metrics(quality_metrics, "Quality Metrics")
        summary += "\n" + print_metrics(relevancy_metrics, "Relevancy Metrics")
        summary += "\n" + print_metrics(retrieval_metrics, "Retrieval Metrics")
        summary += "\n" + print_metrics(safety_metrics, "Safety Metrics")
        
        summary += "\n" + "="*60 + "\n"
        
        return summary
    
    def compare_with_threshold(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """
        Compare results with threshold
        
        Args:
            results: Dictionary of metric scores
            
        Returns:
            Dictionary indicating which metrics passed
        """
        passed = {}
        for metric_name, score in results.items():
            if isinstance(score, (int, float)):
                passed[metric_name] = score >= self.threshold
        return passed
