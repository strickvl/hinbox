from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL

# Use the consolidated extractor rather than per-entity wrappers
from src.engine.extractors import EntityExtractor
from src.engine.relevance import gemini_check_relevance, ollama_check_relevance
from src.exceptions import RelevanceCheckError
from src.logging_config import get_logger
from src.utils.error_handler import handle_article_processing_error

logger = get_logger("article_processor")


class ArticleProcessor:
    """
    Orchestrates article-level operations: relevance checking, per-entity
    extraction, and progress metadata aggregation. The processor is domain-
    aware and supports both cloud and local (Ollama) model modes.

    This class intentionally keeps algorithmic behavior consistent with the
    rest of the system; only the entity extraction dispatch is centralized
    through EntityExtractor to avoid per-entity wrapper imports.
    """

    def __init__(self, domain: str, model_type: str = "gemini"):
        self.domain = domain
        self.model_type = model_type or "gemini"
        self.specific_model = (
            OLLAMA_MODEL if self.model_type == "ollama" else CLOUD_MODEL
        )
        self.logger = logger

    def check_relevance(
        self,
        article_content: str,
        article_id: str,
        langfuse_session_id: str,
        langfuse_trace_id: str,
    ) -> bool:
        """Check whether an article is relevant to the configured domain.

        Robustly interprets return shapes from relevance helpers while preserving
        existing import paths (src.relevance) which bridge to the engine module.
        """
        try:
            if self.model_type == "ollama":
                result = ollama_check_relevance(
                    text=article_content,
                    model=self.specific_model,
                    domain=self.domain,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            else:
                result = gemini_check_relevance(
                    text=article_content,
                    model=self.specific_model,
                    domain=self.domain,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )

            # Normalise result to boolean
            if isinstance(result, bool):
                return result
            if hasattr(result, "is_relevant"):
                return bool(getattr(result, "is_relevant"))
            if isinstance(result, dict) and "is_relevant" in result:
                return bool(result.get("is_relevant"))

            # Default fallback: treat as relevant if uncertain
            return True
        except RelevanceCheckError:
            # Known relevance failure path - treat as not relevant
            return False
        except Exception as e:
            # Standardized article processing error logging
            handle_article_processing_error(article_id, "relevance_check", e)
            return False

    def extract_single_entity_type(
        self,
        entity_type: str,
        article_content: str,
        langfuse_session_id: str = None,
        langfuse_trace_id: str = None,
    ) -> List[Dict]:
        """Extract a single entity type using the generic EntityExtractor.

        This central dispatcher replaces entity-specific wrapper calls. It
        defers to cloud (Gemini via LiteLLM) unless running in \"ollama\" mode.
        """
        try:
            extractor = EntityExtractor(entity_type, self.domain)
            if self.model_type == "ollama":
                return extractor.extract_local(
                    text=article_content,
                    model=self.specific_model,
                    temperature=0,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
            else:
                return extractor.extract_cloud(
                    text=article_content,
                    model=self.specific_model,
                    temperature=0,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
        except Exception:
            # Preserve fallback behavior: no entities on failure
            return []

    def track_reflection_attempts(
        self, extracted_entities: Any, entity_type: str, verbose: bool = False
    ) -> int:
        """
        Count reflection attempts if entities carry reflection history. This
        function remains tolerant of different shapes (dicts or pydantic).
        """
        try:
            if not extracted_entities:
                return 0

            attempts = 0
            for item in extracted_entities:
                # Support either direct dict or pydantic model with model_dump()
                data = (
                    item.model_dump()  # type: ignore[attr-defined]
                    if hasattr(item, "model_dump")
                    else item
                )
                if isinstance(data, dict):
                    hist = data.get("reflection_history") or []
                    if isinstance(hist, list):
                        attempts += len(hist)
            if verbose and attempts:
                self.logger.info(f"Reflection attempts for {entity_type}: {attempts}")
            return attempts
        except Exception:
            return 0

    def extract_all_entities(
        self,
        article_content: str,
        article_id: str,
        processing_metadata: Dict[str, Any],
        verbose: bool = False,
        langfuse_session_id: str = None,
        langfuse_trace_id: str = None,
    ) -> Dict[str, List[Dict]]:
        """Extract all supported entity types from an article."""
        entities: Dict[str, List[Dict]] = {
            "people": [],
            "organizations": [],
            "locations": [],
            "events": [],
        }

        try:
            for et in entities.keys():
                items = self.extract_single_entity_type(
                    et,
                    article_content,
                    langfuse_session_id=langfuse_session_id,
                    langfuse_trace_id=langfuse_trace_id,
                )
                entities[et] = items or []

            # Track reflection attempts across all types
            total_reflections = 0
            for et, items in entities.items():
                total_reflections += self.track_reflection_attempts(
                    items, et, verbose=verbose
                )

            processing_metadata["reflection_summary"] = {
                "total_attempts": total_reflections
            }
            processing_metadata["extracted_counts"] = {
                k: len(v) for k, v in entities.items()
            }

            return entities
        except Exception as e:
            handle_article_processing_error(article_id, "extraction", e)
            return entities

    def convert_pydantic_to_dict(self, items: List) -> List[Dict]:
        """Normalize items to plain dicts, preserving fields for parquet output."""
        results: List[Dict] = []
        for item in items or []:
            if item is None:
                continue
            if isinstance(item, dict):
                results.append(item)
            elif hasattr(item, "model_dump"):
                try:
                    results.append(item.model_dump())  # type: ignore[attr-defined]
                except Exception:
                    # Fallback to dict() if available
                    try:
                        results.append(item.dict())
                    except Exception:
                        pass
            elif hasattr(item, "dict"):
                try:
                    results.append(item.dict())
                except Exception:
                    pass
        return results

    def prepare_article_info(self, row: Dict, row_index: int) -> Dict[str, str]:
        """Extract standard article fields with robust fallbacks."""
        article_id = str(
            row.get("id")
            or row.get("article_id")
            or row.get("_id")
            or f"row_{row_index}"
        )
        title = (
            row.get("title")
            or row.get("article_title")
            or row.get("headline")
            or "(untitled)"
        )
        url = row.get("article_url") or row.get("url") or "#"
        published_date = row.get("published_date") or row.get("date") or ""
        content = (
            row.get("clean_text")
            or row.get("content")
            or row.get("text")
            or row.get("body")
            or ""
        )

        return {
            "id": article_id,
            "title": title,
            "url": url,
            "published_date": published_date,
            "content": content,
        }

    def initialize_processing_metadata(self, row: Dict) -> Dict[str, Any]:
        """Initialize processing metadata from existing row, preserving 'processed'."""
        meta = row.get("processing_metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        # Preserve any prior processed flag if present
        meta.setdefault("processed", False)
        return meta

    def finalize_processing_metadata(
        self,
        processing_metadata: Dict[str, Any],
        extracted_entities: Dict[str, List],
        extraction_timestamp: str,
        verbose: bool,
        row_index: int,
    ) -> float:
        """Finalize and stamp processing metadata for the article."""
        try:
            processing_metadata["processed"] = True
            processing_metadata["extraction_timestamp"] = extraction_timestamp
            counts = {k: len(v or []) for k, v in (extracted_entities or {}).items()}
            processing_metadata["extracted_counts"] = counts

            if "reflection_summary" not in processing_metadata:
                total_attempts = 0
                for items in (extracted_entities or {}).values():
                    total_attempts += self.track_reflection_attempts(
                        items, "aggregate", verbose=verbose
                    )
                processing_metadata["reflection_summary"] = {
                    "total_attempts": total_attempts
                }

            if verbose:
                self.logger.info(
                    f"[Row {row_index}] Extracted counts: {counts} | "
                    f"Reflections: {processing_metadata['reflection_summary'].get('total_attempts', 0)}"
                )
        except Exception as e:
            handle_article_processing_error(
                processing_metadata.get("article_id", "(unknown)"),
                "finalize_metadata",
                e,
            )

        # Historical interface returned a float duration; we return 0.0 by default
        return 0.0
