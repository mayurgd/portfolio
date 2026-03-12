"""
Metrics calculator with support for multiple frameworks

Handles calculation of evaluation metrics using RAGAS or DeepEval
with proper error handling and caching.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from functools import lru_cache

from ..deepeval_metrics import DeepEvalEvaluator
from ..evaluation_metrics import RAGEvaluator
from ..utils.logger import get_logger


class MetricsCalculator:
    """
    Calculate evaluation metrics using specified framework
    
    Supports:
    - RAGAS framework
    - DeepEval framework
    - Metric caching
    - Error handling per metric
    """
    
    def __init__(self, framework: str = "deepeval", threshold: float = 0.7):
        """
        Initialize metrics calculator
        
        Args:
            framework: "ragas" or "deepeval"
            threshold: Minimum passing threshold for DeepEval
        """
        self.framework = framework.lower()
        self.threshold = threshold
        self.logger = get_logger(__name__)
        
        # Initialize evaluators
        if self.framework == "ragas":
            self.evaluator = RAGEvaluator()
        elif self.framework == "deepeval":
            self.evaluator = DeepEvalEvaluator(threshold=threshold)
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        self.logger.info(f"Initialized MetricsCalculator with framework={framework}")
    
    async def calculate_async(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate metrics asynchronously
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth answer
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric scores
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                self._calculate_sync,
                question,
                answer,
                contexts,
                ground_truth,
                metrics
            )
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
        metrics: Optional[List[str]]
    ) -> Dict[str, float]:
        """
        Synchronous metrics calculation
        
        Args:
            Same as calculate_async
            
        Returns:
            Dictionary of metric scores
        """
        try:
            if self.framework == "ragas":
                result = self.evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    metrics=metrics
                )
            else:  # deepeval
                result = self.evaluator.evaluate_single(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                    metrics=metrics
                )
                
                # Extract scores from DeepEval format
                if isinstance(result, dict):
                    scores = {}
                    for metric_name, metric_data in result.items():
                        if isinstance(metric_data, dict) and 'score' in metric_data:
                            scores[metric_name] = metric_data['score']
                        else:
                            scores[metric_name] = metric_data
                    result = scores
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sync calculation error: {str(e)}")
            raise
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics for current framework
        
        Returns:
            List of metric names
        """
        if self.framework == "ragas":
            return [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "answer_similarity",
                "answer_correctness",
                "context_relevancy"
            ]
        else:  # deepeval
            return [
                "faithfulness",
                "answer_relevancy",
                "contextual_precision",
                "contextual_recall",
                "hallucination",
                "bias",
                "toxicity"
            ]
