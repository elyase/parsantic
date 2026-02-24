"""Pluggable retry policy with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Configurable retry strategy.

    Attributes
    ----------
    max_retries
        Maximum number of retry attempts (0 = no retries).
    base_delay
        Initial delay in seconds before first retry.
    backoff_factor
        Multiplier applied to delay on each subsequent retry.
    max_delay
        Upper bound on delay in seconds.
    jitter
        If True, add random jitter (0 to 50% of delay) to prevent thundering herd.
    """

    max_retries: int = 2
    base_delay: float = 0.0
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    jitter: bool = False

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay < 0:
            raise ValueError(f"base_delay must be >= 0, got {self.base_delay}")
        if self.max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {self.max_delay}")

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute delay in seconds before the given retry attempt (0-indexed)."""
        if self.base_delay <= 0:
            return 0.0
        delay = self.base_delay * (self.backoff_factor**attempt)
        if self.jitter:
            delay += random.uniform(0, delay * 0.5)
        return min(delay, self.max_delay)

    def wait(self, attempt: int) -> None:
        """Sleep for the computed delay (sync)."""
        delay = self.delay_for_attempt(attempt)
        if delay > 0:
            time.sleep(delay)

    async def await_delay(self, attempt: int) -> None:
        """Sleep for the computed delay (async)."""
        import asyncio

        delay = self.delay_for_attempt(attempt)
        if delay > 0:
            await asyncio.sleep(delay)
