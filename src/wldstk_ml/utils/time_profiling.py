"""This module provides time profiling that reports through the shared logger.

Usage:

```python
from wldstk_ml.utils.time_profiling import log_time

with log_time("k-Means clustering"):
    labels = kmeans.predict(X)

@log_time("Training")
def train(): ...
```
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

from .logger import get_logger

_default_logger = get_logger(__name__)


@contextmanager
def log_time(
    label: str = "Block",
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Generator[None]:
    """Log how long a block or function takes. Works with `with` and as a context manager."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        yield
    finally:
        wall = time.perf_counter() - wall_start
        cpu = time.process_time() - cpu_start
        (logger or _default_logger).log(
            level, "%s took %s (CPU: %s)", label, _format_duration(wall), _format_duration(cpu)
        )


def _format_duration(seconds: float) -> str:
    """Format a duration in milliseconds below one second, otherwise in seconds."""
    return f"{seconds * 1000:.3f} ms" if seconds < 1 else f"{seconds:.3f} s"
