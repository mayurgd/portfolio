"""
Production-grade RAG evaluation module

This module provides enterprise-ready evaluation capabilities with:
- Proper error handling and logging
- Async/await support
- Rate limiting
- Retry logic
- Metrics aggregation
- Result persistence
"""

from .evaluator import ProductionEvaluator
from .metrics_calculator import MetricsCalculator
from .result_aggregator import ResultAggregator
from .storage_handler import StorageHandler

__all__ = [
    'ProductionEvaluator',
    'MetricsCalculator',
    'ResultAggregator',
    'StorageHandler',
]
