"""Retry logic for transient failures."""

import time
import random
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar("T")


class RetryError(Exception):
    """Raised when all retry attempts fail."""

    def __init__(self, message: str, last_error: Exception, attempts: int):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback called before each retry (exception, attempt, delay)

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get(url)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        raise RetryError(
                            f"Failed after {max_attempts} attempts: {e}",
                            last_error=e,
                            attempts=max_attempts
                        )

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )

                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    # Call retry callback
                    if on_retry:
                        on_retry(e, attempt, delay)

                    time.sleep(delay)

            # Should never reach here, but just in case
            raise RetryError(
                f"Failed after {max_attempts} attempts",
                last_error=last_exception,
                attempts=max_attempts
            )

        return wrapper
    return decorator


def retry_call(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: dict | None = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Callable[[Exception, int, float], None] | None = None
) -> T:
    """Call a function with retry logic.

    Args:
        func: Function to call
        args: Positional arguments for the function
        kwargs: Keyword arguments for the function
        max_attempts: Maximum number of attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback called before each retry

    Example:
        result = retry_call(
            requests.get,
            args=(url,),
            kwargs={"timeout": 10},
            max_attempts=3
        )
    """
    kwargs = kwargs or {}

    @retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        on_retry=on_retry
    )
    def wrapped():
        return func(*args, **kwargs)

    return wrapped()


# Common retryable exception patterns for API calls
API_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# For use with requests library
def is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code is retryable."""
    return status_code in (
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    )


class RetryableAPIError(Exception):
    """Exception for retryable API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code