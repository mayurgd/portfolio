"""
Storage handler for persisting evaluation results

Supports multiple storage backends:
- Local JSON files
- SQLite database
- Cloud storage (future)
"""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path

from ..utils.logger import get_logger


class StorageHandler:
    """
    Handle storage and retrieval of evaluation results
    
    Features:
    - JSON file storage
    - Async I/O
    - Automatic directory creation
    - Result querying
    """
    
    def __init__(self, storage_path: str = "./evaluation_results"):
        """
        Initialize storage handler
        
        Args:
            storage_path: Base path for storing results
        """
        self.storage_path = Path(storage_path)
        self.logger = get_logger(__name__)
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized StorageHandler at {storage_path}")
    
    async def save_async(self, result: Any) -> str:
        """
        Save evaluation result asynchronously
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Path to saved file
        """
        try:
            # Convert to dict
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            # Generate filename (replace colons with hyphens for Windows compatibility)
            timestamp = result_dict.get('timestamp', datetime.now(timezone.utc).isoformat())
            # Replace colons with hyphens to make filename Windows-compatible
            safe_timestamp = timestamp.replace(':', '-')
            request_id = result_dict.get('request_id', 'unknown')
            filename = f"{request_id}_{safe_timestamp}.json"
            filepath = self.storage_path / filename
            
            # Write asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._write_json,
                filepath,
                result_dict
            )
            
            self.logger.info(f"Saved result to {filepath}")
            
            # Update consolidated file if this is a RAGAS evaluation
            metadata = result_dict.get('metadata', {})
            if metadata.get('framework') == 'ragas' or 'faithfulness' in result_dict.get('scores', {}):
                await loop.run_in_executor(
                    None,
                    self._update_consolidated_ragas,
                    result_dict
                )
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save result: {str(e)}")
            raise
    
    def save_sync(self, result: Any) -> str:
        """
        Save evaluation result synchronously
        
        Args:
            result: EvaluationResult object
            
        Returns:
            Path to saved file
        """
        return asyncio.run(self.save_async(result))
    
    def _write_json(self, filepath: Path, data: Dict[str, Any]) -> None:
        """
        Write JSON file
        
        Args:
            filepath: Path to write
            data: Data to write
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def load_async(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Load evaluation result by request ID
        
        Args:
            request_id: Request ID to load
            
        Returns:
            Result dictionary or None
        """
        try:
            # Find file with request_id
            files = list(self.storage_path.glob(f"{request_id}_*.json"))
            
            if not files:
                self.logger.warning(f"No result found for {request_id}")
                return None
            
            # Load most recent if multiple
            filepath = sorted(files)[-1]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._read_json,
                filepath
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load result: {str(e)}")
            return None
    
    def _read_json(self, filepath: Path) -> Dict[str, Any]:
        """
        Read JSON file
        
        Args:
            filepath: Path to read
            
        Returns:
            Loaded data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_results(
        self,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List stored results
        
        Args:
            limit: Maximum number of results
            status: Filter by status
            
        Returns:
            List of result summaries
        """
        results = []
        files = sorted(self.storage_path.glob("*.json"), reverse=True)
        
        for filepath in files[:limit]:
            try:
                data = self._read_json(filepath)
                
                if status is None or data.get('status') == status:
                    results.append({
                        "request_id": data.get('request_id'),
                        "timestamp": data.get('timestamp'),
                        "status": data.get('status'),
                        "average_score": data.get('scores', {}).get('average', 0),
                        "filepath": str(filepath)
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error reading {filepath}: {str(e)}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Statistics dictionary
        """
        files = list(self.storage_path.glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "total_results": len(files),
            "storage_path": str(self.storage_path),
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_result": min((f.stat().st_mtime for f in files), default=None),
            "newest_result": max((f.stat().st_mtime for f in files), default=None)
        }
    
    def _update_consolidated_ragas(self, result_dict: Dict[str, Any]) -> None:
        """
        Update consolidated RAGAS results file
        
        Args:
            result_dict: Result dictionary from evaluation
        """
        try:
            consolidated_path = self.storage_path / "ragas_consolidated_results.json"
            
            # Load existing consolidated file if it exists
            if consolidated_path.exists():
                consolidated = self._read_json(consolidated_path)
            else:
                # Create new consolidated structure
                consolidated = {
                    "metadata": {
                        "title": "Consolidated RAGAS Evaluation Results",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "last_updated": datetime.now(timezone.utc).isoformat()
                    },
                    "evaluations": []
                }
            
            # Add new result
            consolidated["evaluations"].append(result_dict)
            consolidated["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            consolidated["metadata"]["total_evaluations"] = len(consolidated["evaluations"])
            
            # Calculate aggregate statistics
            all_metrics = {}
            total_questions = 0
            
            for eval_result in consolidated["evaluations"]:
                metadata = eval_result.get("metadata", {})
                total_questions += metadata.get("num_questions", 0)
                scores = eval_result.get("scores", {})
                
                for metric_name, score in scores.items():
                    if isinstance(score, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(score)
            
            # Add statistics
            consolidated["statistics"] = {
                "total_evaluations": len(consolidated["evaluations"]),
                "total_questions": total_questions,
                "metrics": {}
            }
            
            for metric_name, scores in all_metrics.items():
                if scores:
                    consolidated["statistics"]["metrics"][metric_name] = {
                        "count": len(scores),
                        "mean": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "median": sorted(scores)[len(scores) // 2]
                    }
            
            # Save consolidated file
            self._write_json(consolidated_path, consolidated)
            self.logger.info(f"Updated consolidated RAGAS results: {consolidated_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not update consolidated RAGAS file: {e}")