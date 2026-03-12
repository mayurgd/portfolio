"""
Rate limiter for API calls

Implements token bucket algorithm for rate limiting to avoid
API throttling and respect rate limits.
"""

import asyncio
import time
import logging
from typing import Optional

from .logger import get_logger


class RateLimiter:
    """
    Token bucket rate limiter
    
    Features:
    - Async-aware
    - Configurable rate
    - Burst handling
    - Thread-safe
    
    Example:
        ```python
        limiter = RateLimiter(rate_limit_per_minute=100)
        
        await limiter.acquire()  # Wait if needed
        # Make API call
        ```
    """
    
    def __init__(
        self,
        rate_limit_per_minute: int = 100,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter
        
        Args:
            rate_limit_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (default: rate_limit)
        """
        self.rate_limit_per_minute = rate_limit_per_minute
        self.burst_size = burst_size or rate_limit_per_minute
        
        # Calculate tokens per second
        self.tokens_per_second = rate_limit_per_minute / 60.0
        
        # Token bucket
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Initialized RateLimiter: {rate_limit_per_minute} requests/min, "
            f"burst={self.burst_size}"
        )
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens (wait if necessary)
        
        Args:
            tokens: Number of tokens to acquire
        """
        async with self.lock:
            while True:
                # Update tokens
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(
                    self.burst_size,
                    self.tokens + elapsed * self.tokens_per_second
                )
                self.last_update = now
                
                # Check if enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.tokens_per_second
                
                self.logger.debug(
                    f"Rate limit reached, waiting {wait_time:.2f}s"
                )
                
                # Wait and retry
                await asyncio.sleep(wait_time)
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired, False otherwise
        """
        # Update tokens
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now
        
        # Check if enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_available_tokens(self) -> float:
        """
        Get number of available tokens
        
        Returns:
            Available tokens
        """
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(
            self.burst_size,
            self.tokens + elapsed * self.tokens_per_second
        )
