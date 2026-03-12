"""
Production-grade RAG evaluator with enterprise features

Features:
- Async evaluation for better performance
- Retry logic with exponential backoff
- Rate limiting to avoid API throttling
- Comprehensive error handling
- Structured logging
- Metrics persistence
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import traceback

from .metrics_calculator import MetricsCalculator
from .result_aggregator import ResultAggregator
from .storage_handler import StorageHandler
from ..utils.rate_limiter import RateLimiter
from ..utils.retry_handler import RetryHandler
from ..utils.logger import get_logger


class EvaluationStatus(Enum):
    """Evaluation status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class EvaluationRequest:
    """Structured evaluation request"""
    request_id: str
    questions: List[str]
    answers: List[str]
    contexts: List[List[str]]
    ground_truths: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    request_id: str
    status: EvaluationStatus
    scores: Dict[str, float]
    item_scores: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ProductionEvaluator:
    """
    Production-grade RAG evaluation system
    
    Features:
    - Async evaluation with concurrency control
    - Rate limiting and retry logic
    - Comprehensive error handling
    - Structured logging
    - Result persistence
    - Progress tracking
    
    Example:
        ```python
        evaluator = ProductionEvaluator(
            framework="deepeval",
            max_concurrent=5,
            rate_limit_per_minute=100
        )
        
        result = await evaluator.evaluate_async(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths
        )
        ```
    """
    
    def __init__(
        self,
        framework: str = "deepeval",
        max_concurrent: int = 5,
        rate_limit_per_minute: int = 100,
        enable_retry: bool = True,
        max_retries: int = 3,
        enable_persistence: bool = True,
        storage_path: str = "./evaluation_results",
        log_level: str = "INFO"
    ):
        """
        Initialize production evaluator
        
        Args:
            framework: Evaluation framework ("ragas" or "deepeval")
            max_concurrent: Maximum concurrent evaluations
            rate_limit_per_minute: Rate limit for API calls
            enable_retry: Enable retry logic
            max_retries: Maximum retry attempts
            enable_persistence: Enable result persistence
            storage_path: Path for storing results
            log_level: Logging level
        """
        self.framework = framework
        self.max_concurrent = max_concurrent
        self.enable_retry = enable_retry
        self.enable_persistence = enable_persistence
        
        # Initialize components
        self.logger = get_logger(__name__, level=log_level)
        self.metrics_calculator = MetricsCalculator(framework=framework)
        self.result_aggregator = ResultAggregator()
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.retry_handler = RetryHandler(max_retries=max_retries) if enable_retry else None
        
        if enable_persistence:
            self.storage_handler = StorageHandler(storage_path)
        else:
            self.storage_handler = None
        
        # Tracking
        self._total_evaluations = 0
        self._successful_evaluations = 0
        self._failed_evaluations = 0
        
        self.logger.info(
            f"Initialized ProductionEvaluator with framework={framework}, "
            f"max_concurrent={max_concurrent}, rate_limit={rate_limit_per_minute}"
        )
    
    async def evaluate_async(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResult:
        """
        Async evaluation with full production features
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts for each question
            ground_truths: Optional ground truth answers
            metrics: List of metrics to evaluate
            metadata: Additional metadata
            progress_callback: Optional callback for progress updates
            
        Returns:
            EvaluationResult with scores and metadata
        """
        start_time = datetime.now(timezone.utc)
        request_id = self._generate_request_id()
        
        # Create request object
        request = EvaluationRequest(
            request_id=request_id,
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            metadata=metadata or {},
            metrics=metrics
        )
        
        self.logger.info(
            f"Starting evaluation {request_id} with {len(questions)} questions"
        )
        
        try:
            # Validate input
            self._validate_input(request)
            
            # Evaluate items concurrently
            item_scores = await self._evaluate_items_concurrent(
                request,
                progress_callback
            )
            
            # Aggregate scores
            aggregated_scores = self.result_aggregator.aggregate(item_scores)
            
            # Create result
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            result = EvaluationResult(
                request_id=request_id,
                status=EvaluationStatus.COMPLETED,
                scores=aggregated_scores,
                item_scores=item_scores,
                metadata={
                    **request.metadata,
                    "num_questions": len(questions),
                    "framework": self.framework,
                    "metrics_evaluated": metrics or "all"
                },
                timestamp=start_time,
                duration_seconds=duration
            )
            
            # Persist result
            if self.storage_handler:
                await self._persist_result(result)
            
            # Update stats
            self._total_evaluations += 1
            self._successful_evaluations += 1
            
            self.logger.info(
                f"Completed evaluation {request_id} in {duration:.2f}s "
                f"with average score {aggregated_scores.get('average', 0):.3f}"
            )
            
            return result
            
        except Exception as e:
            self._failed_evaluations += 1
            error_msg = f"Evaluation failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            return EvaluationResult(
                request_id=request_id,
                status=EvaluationStatus.FAILED,
                scores={},
                item_scores=[],
                metadata=metadata or {},
                timestamp=start_time,
                duration_seconds=duration,
                errors=[error_msg]
            )
    
    def evaluate_sync(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Synchronous wrapper for async evaluation
        
        Args:
            Same as evaluate_async
            
        Returns:
            EvaluationResult
        """
        return asyncio.run(
            self.evaluate_async(
                questions,
                answers,
                contexts,
                ground_truths,
                metrics,
                metadata
            )
        )
    
    async def _evaluate_items_concurrent(
        self,
        request: EvaluationRequest,
        progress_callback: Optional[Callable[[int, int], None]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate items with controlled concurrency
        
        Args:
            request: Evaluation request
            progress_callback: Optional progress callback
            
        Returns:
            List of item scores
        """
        total_items = len(request.questions)
        completed = 0
        results = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def evaluate_item_with_semaphore(index: int) -> Dict[str, Any]:
            """Evaluate single item with concurrency control"""
            async with semaphore:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                # Evaluate
                if self.retry_handler:
                    result = await self.retry_handler.execute_with_retry(
                        self._evaluate_single_item,
                        request,
                        index
                    )
                else:
                    result = await self._evaluate_single_item(request, index)
                
                # Update progress
                nonlocal completed
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_items)
                
                return result
        
        # Execute all evaluations concurrently
        tasks = [
            evaluate_item_with_semaphore(i)
            for i in range(total_items)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        item_scores = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(
                    f"Item {i} failed: {str(result)}"
                )
                item_scores.append({
                    "index": i,
                    "status": "failed",
                    "error": str(result),
                    "scores": {}
                })
            else:
                item_scores.append(result)
        
        return item_scores
    
    async def _evaluate_single_item(
        self,
        request: EvaluationRequest,
        index: int
    ) -> Dict[str, Any]:
        """
        Evaluate a single item
        
        Args:
            request: Evaluation request
            index: Item index
            
        Returns:
            Item scores
        """
        try:
            ground_truth = None
            if request.ground_truths and index < len(request.ground_truths):
                ground_truth = request.ground_truths[index]
            
            scores = await self.metrics_calculator.calculate_async(
                question=request.questions[index],
                answer=request.answers[index],
                contexts=request.contexts[index],
                ground_truth=ground_truth,
                metrics=request.metrics
            )
            
            return {
                "index": index,
                "question": request.questions[index],
                "status": "completed",
                "scores": scores
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating item {index}: {str(e)}")
            raise
    
    def _validate_input(self, request: EvaluationRequest) -> None:
        """
        Validate evaluation input
        
        Args:
            request: Evaluation request
            
        Raises:
            ValueError: If input is invalid
        """
        if len(request.questions) != len(request.answers):
            raise ValueError(
                f"Questions ({len(request.questions)}) and answers "
                f"({len(request.answers)}) length mismatch"
            )
        
        if len(request.questions) != len(request.contexts):
            raise ValueError(
                f"Questions ({len(request.questions)}) and contexts "
                f"({len(request.contexts)}) length mismatch"
            )
        
        if request.ground_truths is not None:
            if len(request.questions) != len(request.ground_truths):
                raise ValueError(
                    f"Questions ({len(request.questions)}) and ground_truths "
                    f"({len(request.ground_truths)}) length mismatch"
                )
        
        if len(request.questions) == 0:
            raise ValueError("No questions provided")
        
        self.logger.debug(f"Input validation passed for {len(request.questions)} items")
    
    async def _persist_result(self, result: EvaluationResult) -> None:
        """
        Persist evaluation result
        
        Args:
            result: Evaluation result
        """
        try:
            await self.storage_handler.save_async(result)
            self.logger.debug(f"Persisted result {result.request_id}")
        except Exception as e:
            self.logger.error(f"Failed to persist result: {str(e)}")
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        from uuid import uuid4
        return f"eval_{uuid4().hex[:12]}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_evaluations": self._total_evaluations,
            "successful_evaluations": self._successful_evaluations,
            "failed_evaluations": self._failed_evaluations,
            "success_rate": (
                self._successful_evaluations / self._total_evaluations
                if self._total_evaluations > 0 else 0
            ),
            "framework": self.framework
        }
