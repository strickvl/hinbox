"""
Standardized error handling utilities for the Hinbox entity extraction system.

This module provides common error handling patterns, logging, and recovery strategies
that can be used consistently across the codebase.
"""

import logging
import time
import traceback
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from rich.console import Console

from src.exceptions import (
    ArticleProcessingError,
    EntityExtractionError,
    EntityMergeError,
    LLMError,
    ModelError,
    ProfileGenerationError,
    RetryableError,
    create_error_context,
    get_retry_delay,
    is_retryable_error,
)

logger = logging.getLogger(__name__)
console = Console()

T = TypeVar("T")


class ErrorHandler:
    """Centralized error handling utility class."""

    def __init__(self, operation: str, context: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.context = context or {}
        self.logger = logging.getLogger(f"hinbox.{operation}")

    def log_error(
        self,
        error: Exception,
        severity: str = "error",
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an error with standardized formatting and context."""
        full_context = {**self.context}
        if additional_context:
            full_context.update(additional_context)

        # Create error message with context
        error_msg = f"[{self.operation}] {str(error)}"
        if full_context:
            context_str = ", ".join(f"{k}={v}" for k, v in full_context.items())
            error_msg += f" (Context: {context_str})"

        # Log with appropriate severity
        log_func = getattr(self.logger, severity, self.logger.error)
        log_func(error_msg)

        # Log stack trace for debugging
        if severity in ["error", "critical"]:
            self.logger.debug(
                f"Stack trace for {self.operation}: {traceback.format_exc()}"
            )

        # Also log to console for immediate visibility
        color = {
            "debug": "blue",
            "info": "green",
            "warning": "yellow",
            "error": "red",
            "critical": "bold red",
        }.get(severity, "red")
        console.print(f"[{color}]{error_msg}[/{color}]")

    def create_fallback_result(
        self, fallback_value: Any = None, error: Exception = None
    ) -> Any:
        """Create a standardized fallback result when operations fail."""
        if fallback_value is not None:
            return fallback_value

        # Default fallbacks based on operation type
        if "extract" in self.operation.lower():
            return []
        elif "profile" in self.operation.lower():
            return {
                "text": f"Profile generation failed for {self.context.get('entity_name', 'unknown entity')}",
                "tags": [],
                "confidence": 0.0,
                "sources": [self.context.get("article_id", "unknown")],
            }
        elif "merge" in self.operation.lower():
            return {}
        else:
            return None

    def handle_exception(
        self,
        error: Exception,
        fallback_value: Any = None,
        re_raise: bool = False,
        severity: str = "error",
    ) -> Any:
        """Handle an exception with logging and optional fallback."""
        self.log_error(error, severity)

        if re_raise:
            raise error

        return self.create_fallback_result(fallback_value, error)


def with_error_handling(
    operation: str,
    fallback_value: Any = None,
    re_raise: bool = False,
    context_func: Optional[Callable] = None,
):
    """Decorator for standardized error handling."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Extract context from function arguments if context_func provided
            context = {}
            if context_func:
                try:
                    context = context_func(*args, **kwargs)
                except Exception:
                    pass  # Don't fail on context extraction

            handler = ErrorHandler(operation, context)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler.handle_exception(e, fallback_value, re_raise)

        return wrapper

    return decorator


def retry_on_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
):
    """Decorator for retrying operations on retryable errors."""
    if retryable_exceptions is None:
        retryable_exceptions = [RetryableError, ModelError, LLMError]

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Check if error is retryable
                    should_retry = attempt < max_retries and (
                        is_retryable_error(e)
                        or any(
                            isinstance(e, exc_type) for exc_type in retryable_exceptions
                        )
                    )

                    if not should_retry:
                        raise e

                    # Calculate delay (use error-specific delay if available)
                    error_delay = get_retry_delay(e)
                    actual_delay = error_delay if error_delay is not None else delay

                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {actual_delay}s: {str(e)}"
                    )
                    time.sleep(actual_delay)
                    delay *= backoff_factor

            # This should never be reached, but just in case
            raise last_error

        return wrapper

    return decorator


# Specific error handlers for common operations


def handle_extraction_error(
    entity_type: str, article_id: str, error: Exception, operation: str = "extraction"
) -> List[Dict]:
    """Standardized error handling for entity extraction operations."""
    context = create_error_context(entity_type=entity_type, article_id=article_id)
    handler = ErrorHandler(f"{entity_type}_{operation}", context)

    # Convert to specific exception type if needed
    if not isinstance(error, EntityExtractionError):
        error = EntityExtractionError(
            f"Failed to extract {entity_type} entities",
            entity_type,
            article_id,
            {"original_error": str(error)},
        )

    return handler.handle_exception(error, fallback_value=[])


def handle_profile_error(
    entity_name: str,
    entity_type: str,
    article_id: str,
    error: Exception,
    operation: str = "profile_generation",
) -> Dict[str, Any]:
    """Standardized error handling for profile operations."""
    context = create_error_context(
        entity_name=entity_name, entity_type=entity_type, article_id=article_id
    )
    handler = ErrorHandler(f"{entity_type}_{operation}", context)

    # Convert to specific exception type if needed
    if not isinstance(error, ProfileGenerationError):
        error = ProfileGenerationError(
            f"Failed to generate profile for {entity_name}",
            entity_name,
            entity_type,
            article_id,
            {"original_error": str(error)},
        )

    fallback_profile = {
        "text": f"Profile generation failed for {entity_name}^[{article_id}]",
        "tags": [],
        "confidence": 0.0,
        "sources": [article_id],
    }

    return handler.handle_exception(error, fallback_value=fallback_profile)


def handle_merge_error(
    entity_type: str, entity_key: str, error: Exception, operation: str = "merge"
) -> None:
    """Standardized error handling for merge operations."""
    context = create_error_context(entity_type=entity_type, entity_key=entity_key)
    handler = ErrorHandler(f"{entity_type}_{operation}", context)

    # Convert to specific exception type if needed
    if not isinstance(error, EntityMergeError):
        error = EntityMergeError(
            f"Failed to merge {entity_type} entity",
            entity_type,
            entity_key,
            {"original_error": str(error)},
        )

    # For merge operations, we typically just log and continue
    handler.handle_exception(error, severity="warning", re_raise=False)


def handle_article_processing_error(
    article_id: str, phase: str, error: Exception
) -> None:
    """Standardized error handling for article processing operations."""
    context = create_error_context(article_id=article_id, phase=phase)
    handler = ErrorHandler("article_processing", context)

    # Convert to specific exception type if needed
    if not isinstance(error, ArticleProcessingError):
        error = ArticleProcessingError(
            f"Failed to process article in {phase} phase",
            article_id,
            phase,
            {"original_error": str(error)},
        )

    # Log the error but don't re-raise to allow processing to continue
    handler.handle_exception(error, severity="error", re_raise=False)


# Context extraction functions for decorators


def extract_extraction_context(*args, **kwargs) -> Dict[str, Any]:
    """Extract context for entity extraction operations."""
    context = {}
    if len(args) > 0:
        context["text_length"] = len(str(args[0]))
    if "entity_type" in kwargs:
        context["entity_type"] = kwargs["entity_type"]
    if "article_id" in kwargs:
        context["article_id"] = kwargs["article_id"]
    return context


def extract_profile_context(*args, **kwargs) -> Dict[str, Any]:
    """Extract context for profile operations."""
    context = {}
    if len(args) >= 2:
        context["entity_type"] = args[0]
        context["entity_name"] = args[1]
    if len(args) >= 4:
        context["article_id"] = args[3]
    return context


def extract_merge_context(*args, **kwargs) -> Dict[str, Any]:
    """Extract context for merge operations."""
    context = {}
    if len(args) > 0:
        context["extracted_count"] = len(args[0]) if hasattr(args[0], "__len__") else 0
    if "entity_type" in kwargs:
        context["entity_type"] = kwargs["entity_type"]
    return context
