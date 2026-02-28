from typing import Any, Dict, List, Optional, Tuple

from src.config_loader import get_system_prompt
from src.constants import CLOUD_MODEL, OLLAMA_MODEL

# Use the consolidated extractor rather than per-entity wrappers
from src.engine.extractors import EntityExtractor
from src.engine.relevance import gemini_check_relevance, ollama_check_relevance
from src.exceptions import RelevanceCheckError
from src.logging_config import get_logger
from src.utils.cache_utils import sha256_text
from src.utils.error_handler import handle_article_processing_error
from src.utils.outcomes import PhaseOutcome
from src.utils.quality_controls import (
    filter_entities_by_article_relevance,
    run_extraction_qc,
)
from src.utils.trace_schema import TraceEvent, TraceStage
from src.utils.trace_writer import TraceWriter

logger = get_logger("article_processor")

# QC flags that trigger a single extraction retry
_RETRY_TRIGGER_FLAGS = {
    "zero_entities",
    "high_drop_rate",
    "many_duplicates",
    "many_low_quality_names",
}


def _should_retry_extraction(flags: List[str]) -> bool:
    """Return True if any QC flag warrants a retry attempt."""
    return bool(_RETRY_TRIGGER_FLAGS.intersection(flags))


def _build_repair_hint(entity_type: str, flags: List[str]) -> str:
    """Build a short prompt suffix describing what went wrong on the first attempt."""
    active = sorted(set(flags) & _RETRY_TRIGGER_FLAGS)
    flag_str = ", ".join(active)
    hint = (
        f"IMPORTANT — Previous extraction of {entity_type} had quality issues "
        f"({flag_str}). Please ensure all required fields are populated, "
        f"avoid duplicate entries, and return every relevant entity found in "
        f"the text as a complete JSON array."
    )
    if "many_low_quality_names" in active:
        hint += (
            " Use proper nouns for entity names — avoid generic plurals like "
            "'defense departments' or descriptive phrases like 'military base "
            "in Cuba'. Name each entity with its specific, official name."
        )
    return hint


class ArticleProcessor:
    """
    Orchestrates article-level operations: relevance checking, per-entity
    extraction, and progress metadata aggregation. The processor is domain-
    aware and supports both cloud and local (Ollama) model modes.

    Methods return PhaseOutcome objects that carry both a usable value and
    observability metadata (success/failure, QC flags, error details).
    """

    def __init__(
        self,
        domain: str,
        model_type: str = "gemini",
        trace_writer: Optional[TraceWriter] = None,
    ):
        self.domain = domain
        self.model_type = model_type or "gemini"
        self.specific_model = (
            OLLAMA_MODEL if self.model_type == "ollama" else CLOUD_MODEL
        )
        self.logger = logger
        self.trace_writer = trace_writer

    def _emit_trace(self, event: TraceEvent) -> None:
        """Safely emit a trace event when tracing is enabled."""
        if self.trace_writer is None:
            return
        try:
            self.trace_writer.emit(event)
        except Exception as exc:  # pragma: no cover - tracing must not break pipeline
            self.logger.debug(f"Failed to emit trace event: {exc}")

    def _model_trace_info(self) -> Dict[str, Any]:
        """Return a consistent model descriptor for trace events."""
        return {
            "backend": "local" if self.model_type == "ollama" else "cloud",
            "name": self.specific_model,
            "temperature": 0,
        }

    def _prompt_fingerprint(self, prompt_key: str) -> Optional[str]:
        """Return a stable prompt fingerprint for a domain prompt key."""
        try:
            prompt = get_system_prompt(prompt_key, self.domain)
            return sha256_text(prompt)
        except Exception:
            return None

    def _prompt_fingerprint_with_flags(
        self, prompt_key: str
    ) -> Tuple[Optional[str], List[str]]:
        """Return prompt fingerprint plus any attribution flags."""
        prompt_hash = self._prompt_fingerprint(prompt_key)
        flags: List[str] = []
        if not prompt_hash:
            flags.append("prompt_fingerprint_missing")
        return prompt_hash, flags

    def _extraction_prompt_fingerprint(
        self,
        entity_type: str,
        repair_hint: Optional[str] = None,
    ) -> Optional[str]:
        """Fingerprint the exact extraction prompt used for an attempt."""
        try:
            base_prompt = get_system_prompt(entity_type, self.domain)
            full_prompt = (
                f"{base_prompt}\n\n{repair_hint}" if repair_hint else base_prompt
            )
            return sha256_text(full_prompt)
        except Exception:
            return None

    @staticmethod
    def _entity_display_name(entity_type: str, entity: Dict[str, Any]) -> str:
        """Extract a user-friendly entity name from extraction output."""
        if entity_type == "events":
            return str(entity.get("title") or "")
        return str(entity.get("name") or "")

    def check_relevance(
        self,
        article_content: str,
        article_id: str,
    ) -> PhaseOutcome:
        """Check whether an article is relevant to the configured domain.

        Returns PhaseOutcome[bool] — value is True/False, with error details
        if the check itself failed.
        """
        try:
            if self.model_type == "ollama":
                result = ollama_check_relevance(
                    text=article_content,
                    model=self.specific_model,
                    domain=self.domain,
                )
            else:
                result = gemini_check_relevance(
                    text=article_content,
                    model=self.specific_model,
                    domain=self.domain,
                )

            # Normalise result to boolean
            is_relevant: bool
            reason = ""
            if isinstance(result, bool):
                is_relevant = result
            elif hasattr(result, "is_relevant"):
                is_relevant = bool(result.is_relevant)
                reason = getattr(result, "reason", "")
            elif isinstance(result, dict) and "is_relevant" in result:
                is_relevant = bool(result.get("is_relevant"))
                reason = result.get("reason", "")
            else:
                is_relevant = True
                reason = "uncertain_result_shape"

            prompt_hash, prompt_flags = self._prompt_fingerprint_with_flags("relevance")
            outcome = PhaseOutcome.ok(
                "relevance",
                value=is_relevant,
                meta={"reason": reason},
            )
            self._emit_trace(
                TraceEvent(
                    stage=TraceStage.RELEVANCE,
                    article_id=article_id,
                    decision="relevant" if is_relevant else "irrelevant",
                    reason=reason or None,
                    inputs={"content_hash": sha256_text(article_content)},
                    model=self._model_trace_info(),
                    prompt_fingerprint=prompt_hash,
                    qc_flags=prompt_flags,
                )
            )
            return outcome
        except RelevanceCheckError as e:
            prompt_hash, prompt_flags = self._prompt_fingerprint_with_flags("relevance")
            self._emit_trace(
                TraceEvent(
                    stage=TraceStage.RELEVANCE,
                    article_id=article_id,
                    decision="error",
                    reason=str(e),
                    inputs={"content_hash": sha256_text(article_content)},
                    model=self._model_trace_info(),
                    prompt_fingerprint=prompt_hash,
                    qc_flags=["relevance_exception", *prompt_flags],
                )
            )
            return PhaseOutcome.fail(
                "relevance",
                error=e,
                fallback=False,
                context={"article_id": article_id},
            )
        except Exception as e:
            handle_article_processing_error(article_id, "relevance_check", e)
            prompt_hash, prompt_flags = self._prompt_fingerprint_with_flags("relevance")
            self._emit_trace(
                TraceEvent(
                    stage=TraceStage.RELEVANCE,
                    article_id=article_id,
                    decision="error",
                    reason=str(e),
                    inputs={"content_hash": sha256_text(article_content)},
                    model=self._model_trace_info(),
                    prompt_fingerprint=prompt_hash,
                    qc_flags=["relevance_exception", *prompt_flags],
                )
            )
            return PhaseOutcome.fail(
                "relevance",
                error=e,
                fallback=False,
                context={"article_id": article_id},
            )

    def _run_extraction(
        self,
        extractor: EntityExtractor,
        article_content: str,
        repair_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a single extraction attempt (cloud or local) and return raw dicts."""
        if self.model_type == "ollama":
            raw = extractor.extract_local(
                text=article_content,
                model=self.specific_model,
                temperature=0,
                repair_hint=repair_hint,
            )
        else:
            raw = extractor.extract_cloud(
                text=article_content,
                model=self.specific_model,
                temperature=0,
                repair_hint=repair_hint,
            )
        return self.convert_pydantic_to_dict(raw or [])

    def extract_single_entity_type(
        self,
        entity_type: str,
        article_content: str,
        article_id: str = "",
    ) -> PhaseOutcome:
        """Extract a single entity type, run QC, optionally retry once, and return a PhaseOutcome.

        When the first extraction attempt triggers severe QC flags (zero_entities,
        high_drop_rate, many_duplicates), a single retry is attempted with a
        repair hint appended to the system prompt.  The better result (by
        output_count, then fewer severe flags) is kept.

        The outcome's value is always a List[Dict] (possibly empty on failure).
        QC flags and counts are surfaced in the outcome metadata.
        """
        phase = f"extract.{entity_type}"

        try:
            extractor = EntityExtractor(entity_type, self.domain)
            base_prompt_hash = self._extraction_prompt_fingerprint(entity_type)
            retry_prompt_hash: Optional[str] = None

            # --- Attempt 1 ---
            raw_dicts = self._run_extraction(extractor, article_content)
            cleaned, qc_report = run_extraction_qc(
                entity_type=entity_type,
                entities=raw_dicts,
                domain=self.domain,
            )

            # --- Conditional retry ---
            retry_meta: Dict[str, Any] = {"retry_attempted": False}

            if _should_retry_extraction(qc_report.flags):
                retry_meta["retry_attempted"] = True
                retry_meta["retry_trigger_flags"] = sorted(
                    set(qc_report.flags) & _RETRY_TRIGGER_FLAGS
                )
                logger.info(
                    f"Retrying {entity_type} extraction "
                    f"(triggers: {retry_meta['retry_trigger_flags']})"
                )

                hint = _build_repair_hint(entity_type, qc_report.flags)
                retry_prompt_hash = self._extraction_prompt_fingerprint(
                    entity_type,
                    repair_hint=hint,
                )
                raw_dicts_v2 = self._run_extraction(
                    extractor, article_content, repair_hint=hint
                )
                cleaned_v2, qc_report_v2 = run_extraction_qc(
                    entity_type=entity_type,
                    entities=raw_dicts_v2,
                    domain=self.domain,
                )

                retry_meta["retry_output_count"] = qc_report_v2.output_count

                # Pick the better result: higher output_count wins; on tie,
                # fewer severe flags wins.
                v2_severe = len(set(qc_report_v2.flags) & _RETRY_TRIGGER_FLAGS)
                v1_severe = len(set(qc_report.flags) & _RETRY_TRIGGER_FLAGS)

                use_v2 = qc_report_v2.output_count > qc_report.output_count or (
                    qc_report_v2.output_count == qc_report.output_count
                    and v2_severe < v1_severe
                )

                if use_v2:
                    cleaned, qc_report = cleaned_v2, qc_report_v2
                    retry_meta["retry_used"] = True
                else:
                    retry_meta["retry_used"] = False

            counts = {
                "raw_count": qc_report.input_count,
                "dropped_missing_required": qc_report.dropped_missing_required,
                "deduped": qc_report.deduped,
                "final_count": qc_report.output_count,
            }
            outcome = PhaseOutcome.ok(
                phase,
                value=cleaned,
                counts=counts,
                flags=qc_report.flags,
                meta={"qc_fixes": qc_report.fixes, **retry_meta},
            )

            final_prompt_hash = (
                retry_prompt_hash
                if retry_meta.get("retry_attempted") and retry_meta.get("retry_used")
                else base_prompt_hash
            )
            prompt_flags: List[str] = []
            if not final_prompt_hash:
                prompt_flags.append("prompt_fingerprint_missing")

            inputs = {
                "content_hash": sha256_text(article_content),
                "prompt_hash": final_prompt_hash or "",
            }
            extraction_flags = list(qc_report.flags) + prompt_flags

            for entity in cleaned:
                self._emit_trace(
                    TraceEvent(
                        stage=TraceStage.EXTRACTION,
                        article_id=article_id or None,
                        entity_type=entity_type,
                        entity_name=self._entity_display_name(entity_type, entity)
                        or None,
                        inputs=inputs,
                        model=self._model_trace_info(),
                        prompt_fingerprint=final_prompt_hash,
                        decision="extracted",
                        qc_flags=extraction_flags,
                    )
                )

            self._emit_trace(
                TraceEvent(
                    stage=TraceStage.EXTRACTION_QC,
                    article_id=article_id or None,
                    entity_type=entity_type,
                    inputs=inputs,
                    model=self._model_trace_info(),
                    prompt_fingerprint=final_prompt_hash,
                    decision="pass" if not qc_report.flags else "fail",
                    qc_flags=extraction_flags,
                    counts=counts,
                    meta={"qc_fixes": qc_report.fixes},
                )
            )

            if retry_meta.get("retry_attempted"):
                retry_counts: Dict[str, int] = {
                    "final_count": qc_report.output_count,
                    "retry_output_count": int(
                        retry_meta.get("retry_output_count", qc_report.output_count)
                    ),
                }
                retry_event_prompt = retry_prompt_hash or final_prompt_hash
                retry_flags = list(retry_meta.get("retry_trigger_flags", []))
                if not retry_event_prompt:
                    retry_flags.append("prompt_fingerprint_missing")

                self._emit_trace(
                    TraceEvent(
                        stage=TraceStage.EXTRACTION_RETRY,
                        article_id=article_id or None,
                        entity_type=entity_type,
                        inputs={
                            "content_hash": sha256_text(article_content),
                            "prompt_hash": retry_event_prompt or "",
                        },
                        model=self._model_trace_info(),
                        prompt_fingerprint=retry_event_prompt,
                        decision=(
                            "used_v2" if retry_meta.get("retry_used") else "kept_v1"
                        ),
                        qc_flags=retry_flags,
                        counts=retry_counts,
                        meta={
                            "retry_attempted": True,
                            "retry_used": bool(retry_meta.get("retry_used", False)),
                            "retry_trigger_flags": list(
                                retry_meta.get("retry_trigger_flags", [])
                            ),
                            "prompt_hash_v1": base_prompt_hash,
                            "prompt_hash_v2": retry_prompt_hash,
                            "final_prompt_hash": final_prompt_hash,
                        },
                    )
                )

            return outcome
        except Exception as e:
            self.logger.exception(
                f"Extraction failed for entity_type={entity_type}, "
                f"domain={self.domain}, model_type={self.model_type}: {e}"
            )
            prompt_hash = self._extraction_prompt_fingerprint(entity_type)
            prompt_flags = ["prompt_fingerprint_missing"] if not prompt_hash else []
            self._emit_trace(
                TraceEvent(
                    stage=TraceStage.EXTRACTION_QC,
                    article_id=article_id or None,
                    entity_type=entity_type,
                    decision="error",
                    reason=str(e),
                    inputs={
                        "content_hash": sha256_text(article_content),
                        "prompt_hash": prompt_hash or "",
                    },
                    model=self._model_trace_info(),
                    prompt_fingerprint=prompt_hash,
                    qc_flags=["extraction_exception", *prompt_flags],
                )
            )
            return PhaseOutcome.fail(
                phase,
                error=e,
                fallback=[],
                context={
                    "entity_type": entity_type,
                    "domain": self.domain,
                    "model_type": self.model_type,
                    "article_id": article_id,
                },
                flags=["extraction_exception"],
            )

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
        max_workers: int = 1,
    ) -> Dict[str, List[Dict]]:
        """Extract all supported entity types from an article.

        When *max_workers* > 1 the four entity-type extractions run
        concurrently via a thread pool (bounded by the global LLM
        semaphore).  Results are assembled in stable type order.

        Writes per-type PhaseOutcome metadata into
        processing_metadata["phase_outcomes"].
        """
        entity_types = ["people", "organizations", "locations", "events"]
        entities: Dict[str, List[Dict]] = {et: [] for et in entity_types}

        try:
            extraction_outcomes: Dict[str, Any] = {}

            if max_workers > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(entity_types))
                ) as pool:
                    futures = {
                        et: pool.submit(
                            self.extract_single_entity_type,
                            et,
                            article_content,
                            article_id,
                        )
                        for et in entity_types
                    }
                    # Iterate in stable order, not completion order
                    for et in entity_types:
                        outcome = futures[et].result()
                        entities[et] = outcome.value or []
                        extraction_outcomes[et] = outcome.to_metadata_dict()
            else:
                for et in entity_types:
                    outcome = self.extract_single_entity_type(
                        et, article_content, article_id
                    )
                    entities[et] = outcome.value or []
                    extraction_outcomes[et] = outcome.to_metadata_dict()

            # Store outcomes in processing_metadata
            processing_metadata.setdefault("phase_outcomes", {})
            processing_metadata["phase_outcomes"]["extraction"] = extraction_outcomes

            # Entity-level relevance filter (mention validation)
            # Conservative: apply to people/orgs/locations; events opt-in
            relevance_types = ["people", "organizations", "locations"]
            relevance_outcomes: Dict[str, Any] = {}
            for et in relevance_types:
                if entities[et]:
                    filtered, rel_report = filter_entities_by_article_relevance(
                        entity_type=et,
                        entities=entities[et],
                        article_text=article_content,
                        domain=self.domain,
                        require_mention=True,
                    )
                    if rel_report.dropped > 0:
                        logger.info(
                            f"Relevance filter: dropped {rel_report.dropped}/{rel_report.input_count} "
                            f"{et} entities (no mention in article)"
                        )
                    entities[et] = filtered
                    relevance_outcomes[et] = rel_report.model_dump()
            processing_metadata["phase_outcomes"]["entity_relevance"] = (
                relevance_outcomes
            )

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
