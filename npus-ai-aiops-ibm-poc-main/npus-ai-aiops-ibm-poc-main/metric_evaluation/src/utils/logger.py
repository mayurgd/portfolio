"""
Logging configuration for production

Provides structured logging with proper formatting,
log levels, and output destinations.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for the application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string
    """
    # Default format
    if format_string is None:
        format_string = (
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (stdout for INFO and below, stderr for WARNING and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    )
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        )
        
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger
    
    Args:
        name: Logger name (usually __name__)
        level: Optional log level override
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


class StructuredLogger:
    """
    Structured logger for machine-readable logs
    
    Example:
        ```python
        logger = StructuredLogger(__name__)
        logger.log_event(
            "evaluation_completed",
            request_id="123",
            duration=5.2,
            score=0.85
        )
        ```
    """
    
    def __init__(self, name: str):
        """Initialize structured logger"""
        self.logger = get_logger(name)
    
    def log_event(self, event_name: str, **kwargs) -> None:
        """
        Log structured event
        
        Args:
            event_name: Event name
            **kwargs: Event attributes
        """
        import json
        
        event_data = {
            "event": event_name,
            **kwargs
        }
        
        self.logger.info(json.dumps(event_data))
    
    def log_metric(self, metric_name: str, value: float, **tags) -> None:
        """
        Log metric
        
        Args:
            metric_name: Metric name
            value: Metric value
            **tags: Additional tags
        """
        self.log_event(
            "metric",
            metric=metric_name,
            value=value,
            **tags
        )
