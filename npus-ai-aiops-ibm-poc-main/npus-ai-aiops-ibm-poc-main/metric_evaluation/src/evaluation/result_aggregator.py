"""
Result aggregation and statistical analysis

Aggregates individual item scores into overall statistics
with percentiles, trends, and quality thresholds.
"""

import logging
from typing import List, Dict, Any
import statistics

from ..utils.logger import get_logger


class ResultAggregator:
    """
    Aggregate and analyze evaluation results
    
    Provides:
    - Mean/median/std aggregation
    - Percentile calculation
    - Pass/fail analysis
    - Trend detection
    """
    
    def __init__(self):
        """Initialize result aggregator"""
        self.logger = get_logger(__name__)
    
    def aggregate(
        self,
        item_scores: List[Dict[str, Any]],
        include_percentiles: bool = True
    ) -> Dict[str, float]:
        """
        Aggregate individual item scores
        
        Args:
            item_scores: List of item score dictionaries
            include_percentiles: Include percentile statistics
            
        Returns:
            Aggregated scores dictionary
        """
        if not item_scores:
            self.logger.warning("No items to aggregate")
            return {}
        
        # Extract all metric names
        metric_names = set()
        for item in item_scores:
            if item.get("status") == "completed" and "scores" in item:
                metric_names.update(item["scores"].keys())
        
        aggregated = {}
        
        # Aggregate each metric
        for metric_name in metric_names:
            scores = []
            for item in item_scores:
                if item.get("status") == "completed" and "scores" in item:
                    score = item["scores"].get(metric_name)
                    if isinstance(score, (int, float)):
                        scores.append(score)
            
            if scores:
                aggregated[f"{metric_name}_mean"] = statistics.mean(scores)
                aggregated[f"{metric_name}_median"] = statistics.median(scores)
                
                if len(scores) > 1:
                    aggregated[f"{metric_name}_std"] = statistics.stdev(scores)
                else:
                    aggregated[f"{metric_name}_std"] = 0.0
                
                aggregated[f"{metric_name}_min"] = min(scores)
                aggregated[f"{metric_name}_max"] = max(scores)
                
                if include_percentiles:
                    aggregated[f"{metric_name}_p25"] = self._percentile(scores, 25)
                    aggregated[f"{metric_name}_p75"] = self._percentile(scores, 75)
                    aggregated[f"{metric_name}_p90"] = self._percentile(scores, 90)
        
        # Overall statistics
        all_scores = []
        for item in item_scores:
            if item.get("status") == "completed" and "scores" in item:
                for score in item["scores"].values():
                    if isinstance(score, (int, float)):
                        all_scores.append(score)
        
        if all_scores:
            aggregated["average"] = statistics.mean(all_scores)
            aggregated["overall_median"] = statistics.median(all_scores)
            if len(all_scores) > 1:
                aggregated["overall_std"] = statistics.stdev(all_scores)
        
        # Success rate
        total_items = len(item_scores)
        successful_items = sum(
            1 for item in item_scores if item.get("status") == "completed"
        )
        aggregated["success_rate"] = successful_items / total_items if total_items > 0 else 0
        
        return aggregated
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """
        Calculate percentile
        
        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value
        """
        sorted_data = sorted(data)
        index = (len(sorted_data) - 1) * percentile / 100
        floor_index = int(index)
        
        if floor_index == len(sorted_data) - 1:
            return sorted_data[floor_index]
        
        fraction = index - floor_index
        return sorted_data[floor_index] + fraction * (
            sorted_data[floor_index + 1] - sorted_data[floor_index]
        )
    
    def analyze_quality(
        self,
        aggregated_scores: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze quality against thresholds
        
        Args:
            aggregated_scores: Aggregated scores
            thresholds: Threshold for each metric
            
        Returns:
            Quality analysis
        """
        analysis = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
        
        for metric_name, threshold in thresholds.items():
            mean_key = f"{metric_name}_mean"
            if mean_key in aggregated_scores:
                score = aggregated_scores[mean_key]
                
                if score >= threshold:
                    analysis["passed"].append({
                        "metric": metric_name,
                        "score": score,
                        "threshold": threshold
                    })
                elif score >= threshold * 0.9:  # Within 10%
                    analysis["warnings"].append({
                        "metric": metric_name,
                        "score": score,
                        "threshold": threshold
                    })
                else:
                    analysis["failed"].append({
                        "metric": metric_name,
                        "score": score,
                        "threshold": threshold
                    })
        
        return analysis
