"""
Retry handler with exponential backoff

Handles transient failures with configurable retry logic,
exponential backoff, and error classification.
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Type, Tuple
import random

from .logger import get_logger


class RetryableError(Exception):
    """Exception that should trigger a retry"""
    pass


class RetryHandler:
    """
    Retry handler with exponential backoff
    
    Features:
    - Exponential backoff with jitter
    - Configurable retry attempts
    - Error classification (retryable vs permanent)
    - Detailed logging
    
    Example:
        ```python
        handler = RetryHandler(max_retries=3)
        
        result = await handler.execute_with_retry(
            my_function,
            arg1,
            arg2,
            kwarg1=value1
        )
        ```
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to delays
            retryable_exceptions: Tuple of exception types to retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        # Default retryable exceptions
        if retryable_exceptions is None:
            self.retryable_exceptions = (
                RetryableError,
                ConnectionError,
                TimeoutError,
            )
        else:
            self.retryable_exceptions = retryable_exceptions
        
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized RetryHandler: max_retries={max_retries}, "
            f"initial_delay={initial_delay}s"
        )
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute (can be sync or async)
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success
                if attempt > 0:
                    self.logger.info(
                        f"Retry successful on attempt {attempt + 1}"
                    )
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if retryable
                if not self._is_retryable(e):
                    self.logger.error(
                        f"Non-retryable error: {type(e).__name__}: {str(e)}"
                    )
                    raise
                
                # Check if more retries available
                if attempt >= self.max_retries:
                    self.logger.error(
                        f"Max retries ({self.max_retries}) exceeded. "
                        f"Last error: {type(e).__name__}: {str(e)}"
                    )
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """
        Check if exception is retryable
        
        Args:
            exception: Exception to check
            
        Returns:
            True if retryable, False otherwise
        """
        # Check if explicitly marked as retryable
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        # Check error message for common transient errors
        error_msg = str(exception).lower()
        transient_keywords = [
            'timeout',
            'connection',
            'rate limit',
            'too many requests',
            'service unavailable',
            'internal server error',
            '429',
            '500',
            '502',
            '503',
            '504'
        ]
        
        return any(keyword in error_msg for keyword in transient_keywords)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt
        
        Args:
            attempt: Attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
