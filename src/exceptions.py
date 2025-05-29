"""
Custom exception classes for the Hinbox entity extraction system.

This module defines a hierarchy of exceptions to provide better error handling
and classification throughout the system.
"""

from typing import Any, Dict, Optional


class HinboxError(Exception):
    """Base exception class for all Hinbox-related errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} (Context: {context_str})"
        return super().__str__()


class ConfigurationError(HinboxError):
    """Raised when there are configuration-related issues."""

    pass


class DataError(HinboxError):
    """Base class for data-related errors."""

    pass


class ArticleLoadError(DataError):
    """Raised when article data cannot be loaded or parsed."""

    pass


class EntityDataError(DataError):
    """Raised when entity data is invalid or corrupted."""

    pass


class ExtractionError(HinboxError):
    """Base class for entity extraction errors."""

    def __init__(
        self,
        message: str,
        entity_type: str,
        article_id: str = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.entity_type = entity_type
        self.article_id = article_id


class ModelError(ExtractionError):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_name: str,
        entity_type: str = None,
        article_id: str = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, entity_type, article_id, context)
        self.model_name = model_name


class LLMError(ModelError):
    """Raised when LLM generation fails."""

    pass


class EmbeddingError(ModelError):
    """Raised when embedding generation fails."""

    pass


class RelevanceCheckError(ExtractionError):
    """Raised when relevance checking fails."""

    pass


class EntityExtractionError(ExtractionError):
    """Raised when entity extraction fails for a specific type."""

    pass


class ProfileGenerationError(ExtractionError):
    """Raised when profile generation fails."""

    def __init__(
        self,
        message: str,
        entity_name: str,
        entity_type: str = None,
        article_id: str = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, entity_type, article_id, context)
        self.entity_name = entity_name


class ProfileUpdateError(ProfileGenerationError):
    """Raised when profile update fails."""

    pass


class MergeError(HinboxError):
    """Base class for entity merging errors."""

    def __init__(
        self, message: str, entity_type: str, context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.entity_type = entity_type


class SimilarityCalculationError(MergeError):
    """Raised when similarity calculation fails during merging."""

    pass


class EntityMergeError(MergeError):
    """Raised when entity merging logic fails."""

    def __init__(
        self,
        message: str,
        entity_type: str,
        entity_key: str = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, entity_type, context)
        self.entity_key = entity_key


class ProcessingError(HinboxError):
    """Base class for processing pipeline errors."""

    pass


class ArticleProcessingError(ProcessingError):
    """Raised when article processing fails in the main pipeline."""

    def __init__(
        self,
        message: str,
        article_id: str,
        phase: str = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.article_id = article_id
        self.phase = phase


class ValidationError(HinboxError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        field: str = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.field = field
        self.value = value


class RetryableError(HinboxError):
    """Base class for errors that can be retried."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context)
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """Raised when rate limits are exceeded."""

    pass


class ServiceUnavailableError(RetryableError):
    """Raised when external services are temporarily unavailable."""

    pass


class NetworkError(RetryableError):
    """Raised when network operations fail."""

    pass


class FrontendError(HinboxError):
    """Base class for frontend-related errors."""

    pass


class RouteError(FrontendError):
    """Raised when route handling fails."""

    def __init__(
        self, message: str, route: str = None, context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, context)
        self.route = route


class TemplateError(FrontendError):
    """Raised when template rendering fails."""

    pass


# Utility functions for error handling


def create_error_context(
    article_id: str = None,
    entity_type: str = None,
    entity_name: str = None,
    model_name: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create a standardized error context dictionary."""
    context = {}
    if article_id:
        context["article_id"] = article_id
    if entity_type:
        context["entity_type"] = entity_type
    if entity_name:
        context["entity_name"] = entity_name
    if model_name:
        context["model_name"] = model_name
    context.update(kwargs)
    return context


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    return isinstance(error, RetryableError)


def get_retry_delay(error: Exception) -> Optional[int]:
    """Get the retry delay for a retryable error."""
    if isinstance(error, RetryableError):
        return error.retry_after
    return None
