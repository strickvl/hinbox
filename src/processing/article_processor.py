"""Article processing module for handling individual article extraction and processing."""

from datetime import datetime
from typing import Any, Dict, List

from src.constants import CLOUD_MODEL, OLLAMA_MODEL
from src.events import gemini_extract_events, ollama_extract_events
from src.exceptions import EntityExtractionError, RelevanceCheckError
from src.locations import gemini_extract_locations, ollama_extract_locations
from src.logging_config import get_logger
from src.organizations import gemini_extract_organizations, ollama_extract_organizations
from src.people import gemini_extract_people, ollama_extract_people
from src.relevance import gemini_check_relevance, ollama_check_relevance
from src.utils.error_handler import handle_article_processing_error

logger = get_logger("article_processor")


class ArticleProcessor:
    """Handles processing of individual articles through the entity extraction pipeline."""

    def __init__(self, domain: str, model_type: str = "gemini"):
        """Initialize the ArticleProcessor.

        Args:
            domain: Domain configuration to use
            model_type: Either 'gemini' or 'ollama'
        """
        self.domain = domain
        self.model_type = model_type
        self.specific_model = OLLAMA_MODEL if model_type == "ollama" else CLOUD_MODEL

    def check_relevance(self, article_content: str, article_id: str) -> bool:
        """Check if article is relevant to the domain.

        Args:
            article_content: Full text of the article
            article_id: Unique identifier for the article

        Returns:
            True if article is relevant, False otherwise
        """
        logger.info("Performing relevance check...")
        try:
            if self.model_type == "ollama":
                relevance_result = ollama_check_relevance(
                    article_content, model="qwq", domain=self.domain
                )
            else:
                relevance_result = gemini_check_relevance(
                    article_content, domain=self.domain
                )

            if not relevance_result.is_relevant:
                logger.warning("Article is not relevant")
                logger.debug(f"Reason: {relevance_result.reason}")
                return False
            else:
                logger.info("Article is relevant")
                return True

        except Exception as e:
            error = RelevanceCheckError(
                "Relevance check failed",
                "unknown",
                article_id,
                {"original_error": str(e), "model_type": self.model_type},
            )
            handle_article_processing_error(article_id, "relevance_check", error)
            logger.warning("Proceeding with extraction despite relevance check error")
            return True  # Default to relevant if check fails

    def extract_single_entity_type(
        self, entity_type: str, article_content: str
    ) -> List[Dict]:
        """Extract a single entity type from article content.

        Args:
            entity_type: Type of entity to extract (people, organizations, locations, events)
            article_content: Full text of the article

        Returns:
            List of extracted entities as dictionaries
        """
        try:
            if self.model_type == "ollama":
                if entity_type == "people":
                    return ollama_extract_people(
                        article_content, model="qwq", domain=self.domain
                    )
                elif entity_type == "organizations":
                    return ollama_extract_organizations(
                        article_content, model="qwq", domain=self.domain
                    )
                elif entity_type == "locations":
                    return ollama_extract_locations(
                        article_content, model="qwq", domain=self.domain
                    )
                elif entity_type == "events":
                    return ollama_extract_events(
                        article_content, model="qwq", domain=self.domain
                    )
            else:
                if entity_type == "people":
                    return gemini_extract_people(article_content, domain=self.domain)
                elif entity_type == "organizations":
                    return gemini_extract_organizations(
                        article_content, domain=self.domain
                    )
                elif entity_type == "locations":
                    return gemini_extract_locations(article_content, domain=self.domain)
                elif entity_type == "events":
                    return gemini_extract_events(article_content, domain=self.domain)
        except Exception as e:
            logger.error(f"Error extracting {entity_type}: {e}")
            return []

        return []

    def track_reflection_attempts(
        self, extracted_entities: Any, entity_type: str, verbose: bool = False
    ) -> int:
        """Track reflection attempts for entity extraction.

        Args:
            extracted_entities: Result from entity extraction
            entity_type: Type of entity being extracted
            verbose: Whether to log detailed reflection information

        Returns:
            Number of reflection attempts made
        """
        reflection_history = []
        reflection_attempts = 1

        # Check if the response has reflection history attribute
        if hasattr(extracted_entities, "reflection_history"):
            reflection_history = extracted_entities.reflection_history
            reflection_attempts = len(reflection_history) if reflection_history else 1

            if verbose and reflection_history:
                logger.debug(f"Reflection history for {entity_type} extraction:")
                for i, reflection in enumerate(reflection_history):
                    passed = reflection.get("passed", False)
                    feedback = reflection.get("feedback", "No feedback")
                    logger.debug(
                        f"  Attempt {i + 1}: {'✓' if passed else '✗'} {feedback}"
                    )

        # For entity extraction modules that return merged results
        if (
            isinstance(extracted_entities, dict)
            and "reflection_history" in extracted_entities
        ):
            reflection_history = extracted_entities["reflection_history"]
            reflection_attempts = len(reflection_history) if reflection_history else 1

        return reflection_attempts

    def extract_all_entities(
        self,
        article_content: str,
        article_id: str,
        processing_metadata: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, List[Dict]]:
        """Extract all entity types from article content.

        Args:
            article_content: Full text of the article
            article_id: Unique identifier for the article
            processing_metadata: Dictionary to store processing metadata
            verbose: Whether to enable verbose logging

        Returns:
            Dictionary with entity types as keys and lists of extracted entities as values
        """
        extracted_entities = {}
        entity_types = ["people", "organizations", "locations", "events"]

        for entity_type in entity_types:
            logger.info(f"Extracting {entity_type}...")
            start_time = datetime.now()

            try:
                entities = self.extract_single_entity_type(entity_type, article_content)

                # Track reflection attempts
                reflection_attempts = self.track_reflection_attempts(
                    entities, entity_type, verbose
                )

                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Extracted {len(entities)} {entity_type} in {duration:.2f}s"
                )

                if reflection_attempts > 1:
                    logger.debug(
                        f"Required {reflection_attempts} reflection iterations"
                    )

                # Update reflection metadata
                processing_metadata["reflection_attempts"][entity_type] = {
                    "attempts": reflection_attempts,
                    "success": bool(entities),
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": duration,
                }

                # Update summary counts
                processing_metadata["reflection_summary"]["total_attempts"] += (
                    reflection_attempts
                )
                processing_metadata["reflection_summary"]["successful_attempts"] += 1

                extracted_entities[entity_type] = entities

            except Exception as e:
                error = EntityExtractionError(
                    f"{entity_type.title()} extraction failed",
                    entity_type,
                    article_id,
                    {"original_error": str(e), "model_type": self.model_type},
                )
                handle_article_processing_error(
                    article_id, f"{entity_type}_extraction", error
                )
                extracted_entities[entity_type] = []
                processing_metadata["reflection_attempts"][entity_type] = {
                    "error": str(error),
                    "timestamp": datetime.now().isoformat(),
                    "attempts": 1,
                    "success": False,
                }
                processing_metadata["reflection_summary"]["failed_attempts"] += 1

        return extracted_entities

    def convert_pydantic_to_dict(self, items: List) -> List[Dict]:
        """Convert Pydantic models to dictionaries.

        Args:
            items: List of items that may be Pydantic models

        Returns:
            List of dictionaries
        """
        dicts = []
        for obj in items:
            # pydantic models have model_dump() or dict()
            if hasattr(obj, "model_dump"):
                dicts.append(obj.model_dump())
            elif hasattr(obj, "dict"):
                dicts.append(obj.dict())
            else:
                dicts.append(obj)
        return dicts

    def prepare_article_info(self, row: Dict, row_index: int) -> Dict[str, str]:
        """Extract and prepare article information from row data.

        Args:
            row: Dictionary containing article data
            row_index: Index of the row being processed

        Returns:
            Dictionary with standardized article information
        """
        return {
            "id": row.get("id", f"article_{row_index}"),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
            "published_date": row.get("published_date", ""),
            "content": row.get("content", ""),
        }

    def initialize_processing_metadata(self, row: Dict) -> Dict[str, Any]:
        """Initialize or get processing metadata for an article.

        Args:
            row: Dictionary containing article data

        Returns:
            Processing metadata dictionary
        """
        if "processing_metadata" not in row:
            row["processing_metadata"] = {}

        processing_metadata = row["processing_metadata"]

        # Initialize enhanced reflection metadata tracking
        processing_metadata["reflection_attempts"] = {}
        processing_metadata["reflection_summary"] = {
            "total_attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
        }

        # Mark processing started
        processing_metadata["processing_started"] = datetime.now().isoformat()
        processing_metadata["model_type"] = self.model_type
        processing_metadata["specific_model"] = self.specific_model

        return processing_metadata

    def finalize_processing_metadata(
        self,
        processing_metadata: Dict[str, Any],
        extracted_entities: Dict[str, List],
        extraction_timestamp: str,
        verbose: bool,
        row_index: int,
    ) -> float:
        """Finalize processing metadata and calculate processing time.

        Args:
            processing_metadata: Processing metadata dictionary
            extracted_entities: Dictionary of extracted entities
            extraction_timestamp: ISO timestamp of extraction
            verbose: Whether verbose logging is enabled
            row_index: Index of the processed row

        Returns:
            Processing time in seconds
        """
        # Mark processing complete
        processing_metadata["reflection_used"] = True
        processing_metadata["processed"] = True
        processing_metadata["processing_completed"] = datetime.now().isoformat()

        # Calculate processing time
        start_time = datetime.fromisoformat(processing_metadata["processing_started"])
        end_time = datetime.fromisoformat(processing_metadata["processing_completed"])
        processing_time = (end_time - start_time).total_seconds()

        # Record total reflection stats
        total_reflection_attempts = processing_metadata["reflection_summary"][
            "total_attempts"
        ]

        # Store extraction counts and processing time
        processing_metadata["entities_extracted"] = {
            "people": len(extracted_entities.get("people", [])),
            "organizations": len(extracted_entities.get("organizations", [])),
            "locations": len(extracted_entities.get("locations", [])),
            "events": len(extracted_entities.get("events", [])),
            "total": sum(len(entities) for entities in extracted_entities.values()),
        }
        processing_metadata["processing_time_seconds"] = processing_time

        # Log reflection summary
        logger.info(f"Reflection summary for article #{row_index}:")
        logger.info(f"  Total reflection attempts: {total_reflection_attempts}")

        # Only log detailed reflection stats in verbose mode or if there were multiple attempts
        if (
            verbose or total_reflection_attempts > 4
        ):  # 4 = minimum if all extractions took just 1 attempt
            for entity_type, reflection_data in processing_metadata[
                "reflection_attempts"
            ].items():
                attempts = reflection_data.get("attempts", 1)
                duration = reflection_data.get("duration_seconds", 0)
                if attempts > 1:
                    logger.info(
                        f"  • {entity_type}: {attempts} attempts in {duration:.2f}s"
                    )

        logger.info(
            f"Successfully processed article #{row_index} in {processing_time:.2f}s"
        )

        return processing_time
