"""Generic entity merger classes to eliminate code duplication."""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.config_loader import DomainConfig
from src.constants import (
    ENABLE_PROFILE_VERSIONING,
    MERGE_GRAY_BAND_DELTA,
    MERGE_UNCERTAIN_CONFIDENCE_CUTOFF,
    SIMILARITY_THRESHOLD,
)
from src.engine.match_checker import cloud_model_check_match, local_model_check_match
from src.engine.merge_candidates import find_similar_entity, lexical_text
from src.engine.merge_dispute_agent import (
    REVIEW_QUEUE_FILENAME,
    MergeDisputeAction,
    run_merge_dispute_agent,
)
from src.engine.merge_evidence import batch_embed_texts, build_evidence_text
from src.engine.profiles import VersionedProfile, create_profile, update_profile
from src.logging_config import (
    DecisionKind,
    display_markdown,
    get_logger,
    log,
    log_decision,
)
from src.utils.embeddings.manager import EmbeddingManager
from src.utils.embeddings.similarity import get_embedding_manager
from src.utils.name_variants import (
    compute_acronym,
    is_acronym_form,
    is_low_quality_name,
    is_name_contained,
    score_canonical_name,
)
from src.utils.profiles import extract_profile_text
from src.utils.task_models import is_local_model, task_model

# Module-specific logger
logger = get_logger("mergers")


def _profile_or_fallback(fallback: Tuple, fn: Callable, *args, **kwargs) -> Tuple:
    """Run a profile create/update; on error, return ``fallback`` instead.

    One failed profile call must not abort the merge of every other entity
    of this type from the same article.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log(f"Profile call failed, keeping fallback: {e}", level="error")
        return fallback


@dataclass
class MergeStats:
    """Aggregate counts from a single merge_entities() run."""

    new: int = 0
    merged: int = 0
    skipped: int = 0
    disputed: int = 0
    errors: int = 0

    @property
    def total(self) -> int:
        return self.new + self.merged + self.skipped + self.disputed + self.errors


@dataclass
class _EvidenceWorkItem:
    """Pre-computed evidence data for one entity in the batch pipeline."""

    idx: int
    entity_dict: Dict[str, Any]
    entity_key: Union[str, Tuple[str, str]]
    evidence_text: str
    evidence_embedding: List[float] = field(default_factory=list)
    evidence_model: Optional[str] = None
    evidence_dim: Optional[int] = None
    evidence_fingerprint: Optional[str] = None


@dataclass
class _PendingProfileEmbedding:
    """Tracks a profile that needs batch embedding after all merge decisions."""

    entity_ref: Dict[str, Any]
    profile_text: str


class EntityMerger:
    """Generic entity merger that handles similarity matching and profile merging."""

    # Configuration for different entity types
    ENTITY_CONFIGS = {
        "people": {
            "key_field": "name",
            "key_type": str,
            "alternative_field": "alternative_names",
            "log_color": "green",
        },
        "organizations": {
            "key_field": ("name", "type"),
            "key_type": tuple,
            "alternative_field": "alternative_names",
            "log_color": "blue",
        },
        "locations": {
            "key_field": ("name", "type"),
            "key_type": tuple,
            "alternative_field": "alternative_names",
            "log_color": "blue",
        },
        "events": {
            "key_field": ("title", "start_date"),
            "key_type": tuple,
            "alternative_field": "alternative_titles",
            "log_color": "blue",
        },
    }

    def __init__(self, entity_type: str):
        """Initialize merger for a specific entity type."""
        if entity_type not in self.ENTITY_CONFIGS:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        self.entity_type = entity_type
        self.config = self.ENTITY_CONFIGS[entity_type]
        self.key_field = self.config["key_field"]
        self.key_type = self.config["key_type"]
        self.alternative_field = self.config["alternative_field"]
        self.log_color = self.config["log_color"]

    def _extract_key(self, entity_dict: Dict[str, Any]) -> Union[str, Tuple[str, str]]:
        """Extract the entity key from an entity dictionary."""
        if self.key_type is str:
            return entity_dict.get(self.key_field, "")
        # tuple key
        return tuple(entity_dict.get(field, "") for field in self.key_field)  # type: ignore[index]

    def _format_key_for_display(self, key: Union[str, Tuple]) -> str:
        """Format entity key for display in logs."""
        return key[0] if isinstance(key, tuple) else key

    def _score_canonical_name(self, name: str) -> float:
        """Score how 'canonical' a name is.  Higher = better.

        Delegates to the shared ``score_canonical_name()`` in name_variants.
        """
        return score_canonical_name(name)

    def _pick_canonical_key(
        self,
        existing_key: Union[str, Tuple],
        incoming_key: Union[str, Tuple],
    ) -> Tuple[Union[str, Tuple], Union[str, Tuple], bool]:
        """Decide which entity key should be canonical after a merge.

        Returns (canonical_key, demoted_key, swapped) where *swapped* is
        True when the incoming name is better and the entity should be
        re-keyed.
        """
        existing_name = lexical_text(existing_key)
        incoming_name = lexical_text(incoming_key)

        # Identical (case-insensitive) — no swap needed
        if existing_name.lower() == incoming_name.lower():
            return existing_key, incoming_key, False

        existing_score = self._score_canonical_name(existing_name)
        incoming_score = self._score_canonical_name(incoming_name)

        # Containment bonus: the longer, more complete name gets +1.0
        if is_name_contained(existing_name, incoming_name):
            # incoming contains existing → incoming is more complete
            incoming_score += 1.0
        elif is_name_contained(incoming_name, existing_name):
            # existing contains incoming → existing is more complete
            existing_score += 1.0

        # Acronym derivation bonus: full form gets +2.0 over its acronym
        if is_acronym_form(existing_name):
            derived = compute_acronym(incoming_name)
            if derived and derived.upper() == existing_name.replace(".", "").upper():
                incoming_score += 2.0
        if is_acronym_form(incoming_name):
            derived = compute_acronym(existing_name)
            if derived and derived.upper() == incoming_name.replace(".", "").upper():
                existing_score += 2.0

        # Swap only when the incoming name is meaningfully better
        if incoming_score > existing_score + 0.3:
            return incoming_key, existing_key, True

        return existing_key, incoming_key, False

    def find_similar_entity(
        self,
        entity_key: Union[str, Tuple],
        entity_embedding: List[float],
        entities: Dict[str, Dict],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        **kwargs: Any,
    ) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
        """Find the closest existing entity; see ``merge_candidates.find_similar_entity``."""
        return find_similar_entity(
            self.entity_type,
            self.alternative_field,
            entity_key,
            entity_embedding,
            entities,
            similarity_threshold,
            **kwargs,
        )

    def _add_alternative_name(
        self,
        existing_entity: Dict[str, Any],
        new_key: Union[str, Tuple],
        source_entity: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add alternative name/title to existing entity if different.
        Matches legacy shape:
          • people: List[str]
          • locations/orgs: List[{'name','type'}]
          • events: List[{'title','start_date','event_type'}]
        """
        if self.alternative_field not in existing_entity:
            existing_entity[self.alternative_field] = []

        if self.entity_type == "people":
            if new_key not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(new_key)
                log(f"Added alternative name: '{new_key}'", level="processing")
                return True

        elif self.entity_type == "events":
            alt_entry = {
                "title": new_key[0] if isinstance(new_key, tuple) else str(new_key),
                "start_date": new_key[1]
                if isinstance(new_key, tuple) and len(new_key) > 1
                else "",
                "event_type": (source_entity or {}).get("event_type", ""),
            }
            if alt_entry not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(alt_entry)
                log(
                    f"Added alternative event: '{alt_entry['title']}'",
                    level="processing",
                )
                return True

        else:  # organizations, locations
            alt_entry = {
                "name": new_key[0] if isinstance(new_key, tuple) else str(new_key),
                "type": new_key[1]
                if isinstance(new_key, tuple) and len(new_key) > 1
                else "",
            }
            if alt_entry not in existing_entity[self.alternative_field]:
                existing_entity[self.alternative_field].append(alt_entry)
                log(
                    f"Added alternative {self.entity_type[:-1]}: '{alt_entry['name']}'",
                    level="processing",
                )
                return True

        return False

    def merge_entities(
        self,
        extracted_entities: List[Dict[str, Any]],
        entities: Dict[str, Dict],
        article_id: str,
        article_title: str,
        article_url: str,
        article_published_date: Any,
        article_content: str,
        extraction_timestamp: str,
        model_type: str = "gemini",
        similarity_threshold: Optional[float] = None,
        domain: str = "guantanamo",
        domain_config: Optional[DomainConfig] = None,
    ) -> MergeStats:
        """Merge extracted entities with existing entities database."""
        stats = MergeStats()
        embedding_manager = get_embedding_manager(domain=domain)

        # Reuse caller-provided config or construct one
        domain_cfg = domain_config or DomainConfig(domain)
        resolved_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else domain_cfg.get_similarity_threshold(self.entity_type)
        )

        # Load lexical blocking config for this entity type
        lexical_blocking_cfg = domain_cfg.get_lexical_blocking_config(self.entity_type)

        # Load merge evidence config (window sizes, max chars)
        evidence_cfg = domain_cfg.get_merge_evidence_config()

        # Load name-variant config (equivalence groups for alias-aware blocking)
        name_variants_cfg = domain_cfg.get_name_variants_config(self.entity_type)
        equivalence_groups = name_variants_cfg.get("equivalence_groups", [])

        # Dispute agent appends DEFER decisions here for human review
        review_queue_path = os.path.join(
            domain_cfg.get_output_dir(), REVIEW_QUEUE_FILENAME
        )

        log(
            f"Starting merge_{self.entity_type} with {len(extracted_entities)} {self.entity_type} to process",
            level="processing",
        )
        log(
            f"Using model: {model_type}, embedding: {embedding_manager.mode.value} "
            f"({embedding_manager.get_active_model_name()}), "
            f"threshold: {resolved_threshold:.2f}, "
            f"lexical_blocking: {'on' if lexical_blocking_cfg.get('enabled') else 'off'}",
            level="info",
        )

        # Batching configuration for embedding calls
        batching_cfg = domain_cfg.get_batching_config()
        embed_batch_size = batching_cfg["embed_batch_size"]

        # ── Phase 1: Build evidence texts for all valid entities ──
        work_items: List[_EvidenceWorkItem] = []
        for idx, entity_dict in enumerate(extracted_entities):
            entity_key = self._extract_key(entity_dict)
            if (
                not entity_key
                or (isinstance(entity_key, str) and not entity_key.strip())
                or (
                    isinstance(entity_key, tuple)
                    and (not entity_key[0] or not str(entity_key[0]).strip())
                )
            ):
                log_decision(
                    DecisionKind.ERROR, self.entity_type, "(empty key)", "empty key"
                )
                stats.errors += 1
                continue
            evidence_text = build_evidence_text(
                self.entity_type,
                self.alternative_field,
                entity_key,
                entity_dict,
                article_content,
                max_chars=evidence_cfg["max_chars"],
                window_chars=evidence_cfg["window_chars"],
                max_windows=evidence_cfg["max_windows"],
            )
            work_items.append(
                _EvidenceWorkItem(
                    idx=idx,
                    entity_dict=entity_dict,
                    entity_key=entity_key,
                    evidence_text=evidence_text,
                )
            )

        # ── Phase 2: Batch-embed all evidence texts ──
        if work_items:
            evidence_texts = [wi.evidence_text for wi in work_items]
            all_evidence_vecs, ev_model, ev_dim, ev_fp = batch_embed_texts(
                embedding_manager, evidence_texts, embed_batch_size, "embed.evidence"
            )
            for j, wi in enumerate(work_items):
                wi.evidence_embedding = all_evidence_vecs[j]
                wi.evidence_model = ev_model
                wi.evidence_dim = ev_dim or (
                    len(all_evidence_vecs[j]) if all_evidence_vecs[j] else None
                )
                wi.evidence_fingerprint = ev_fp

        # ── Phase 3: Sequential merge decisions (using precomputed embeddings) ──
        pending_profiles: List[_PendingProfileEmbedding] = []

        for wi in work_items:
            entity_key = wi.entity_key
            entity_dict = wi.entity_dict
            evidence_text = wi.evidence_text
            evidence_embedding = wi.evidence_embedding
            emb_model_name = wi.evidence_model
            emb_dim = wi.evidence_dim
            entity_updated = False

            # --- Similarity search (variant-aware) ---
            similar_key, similarity_score = self.find_similar_entity(
                entity_key,
                evidence_embedding,
                entities,
                resolved_threshold,
                embedding_model=emb_model_name,
                embedding_dim=emb_dim,
                lexical_blocking_config=lexical_blocking_cfg,
                query_entity=entity_dict,
                equivalence_groups=equivalence_groups,
            )

            if similar_key:
                # ── Guard: reject merge into low-quality candidate ──
                # If the existing entity has a generic/descriptive name like
                # "Defense departments" or "military base in Cuba", skip the
                # merge so the new (presumably better-named) entity gets
                # created fresh instead.
                similar_display = self._format_key_for_display(similar_key)
                if is_low_quality_name(similar_display, entity_type=self.entity_type):
                    log_decision(
                        DecisionKind.SKIP,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        f"candidate '{similar_display}' has low-quality name, forcing creation",
                    )
                    similar_key = None  # fall through to creation path

            if similar_key:
                # --- MERGE PATH (no create_profile needed) ---
                candidate_entity = entities[self.entity_type][similar_key]
                existing_profile_text = candidate_entity.get("profile", {}).get(
                    "text", ""
                )
                existing_profile_missing = not bool(existing_profile_text)
                if existing_profile_missing:
                    # Synthesise a deterministic stand-in so match-check /
                    # dispute can still run; the real profile will be created
                    # via the merge-path fallback if the merge is accepted.
                    existing_profile_text = build_evidence_text(
                        self.entity_type,
                        self.alternative_field,
                        similar_key,
                        candidate_entity,
                        article_content,
                        max_chars=evidence_cfg["max_chars"],
                        window_chars=evidence_cfg["window_chars"],
                        max_windows=evidence_cfg["max_windows"],
                    )
                    if not existing_profile_text:
                        log_decision(
                            DecisionKind.ERROR,
                            self.entity_type,
                            self._format_key_for_display(entity_key),
                            f"no profile text on candidate '{self._format_key_for_display(similar_key)}'"
                            " and could not synthesise stand-in",
                        )
                        stats.errors += 1
                        continue
                    log(
                        f"Candidate '{self._format_key_for_display(similar_key)}' "
                        f"missing profile; using synthesised evidence for match-check",
                        level="debug",
                    )

                # Match check: pass evidence_text as new_profile_text
                match_model = task_model("match", model_type)
                check_match = (
                    local_model_check_match
                    if is_local_model(match_model)
                    else cloud_model_check_match
                )
                result = check_match(
                    self._format_key_for_display(entity_key),
                    self._format_key_for_display(similar_key),
                    evidence_text,
                    existing_profile_text,
                    entity_type=self.entity_type,
                    model=match_model,
                )

                log(
                    f"Match check result: {result.is_match} (confidence={result.confidence:.2f}) - {result.reason}",
                    level="debug",
                )

                # Gray-band routing: when similarity is near the threshold
                # AND the match checker is uncertain, ask the dispute agent
                # for a second opinion before making the merge/skip decision.
                should_merge = result.is_match
                is_gray_band = (
                    similarity_score is not None
                    and abs(similarity_score - resolved_threshold)
                    <= MERGE_GRAY_BAND_DELTA
                )
                is_uncertain = result.confidence < MERGE_UNCERTAIN_CONFIDENCE_CUTOFF

                if is_gray_band and is_uncertain:
                    log(
                        f"Gray-band detected for '{self._format_key_for_display(entity_key)}' vs "
                        f"'{self._format_key_for_display(similar_key)}' "
                        f"(similarity={similarity_score:.4f}, threshold={resolved_threshold:.4f}, "
                        f"confidence={result.confidence:.2f}). Routing to dispute agent.",
                        level="debug",
                    )
                    stats.disputed += 1
                    dispute_decision = run_merge_dispute_agent(
                        entity_type=self.entity_type,
                        new_name=self._format_key_for_display(entity_key),
                        existing_name=self._format_key_for_display(similar_key),
                        new_profile_text=evidence_text,
                        existing_profile_text=existing_profile_text,
                        similarity_score=similarity_score,
                        similarity_threshold=resolved_threshold,
                        match_is_match=result.is_match,
                        match_confidence=result.confidence,
                        match_reason=result.reason,
                        model_type=model_type,
                        domain=domain,
                        article_id=article_id,
                        review_queue_path=review_queue_path,
                    )
                    should_merge = dispute_decision.action == MergeDisputeAction.MERGE

                if not should_merge:
                    # The candidate is a different entity, so the incoming one
                    # is new: fall through to the create path instead of
                    # dropping it.
                    log_decision(
                        DecisionKind.SKIP,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        f"vs '{self._format_key_for_display(similar_key)}' sim={similarity_score:.4f}, "
                        "creating as new",
                    )
                    similar_key = None

            if similar_key:
                log_decision(
                    DecisionKind.MERGE,
                    self.entity_type,
                    self._format_key_for_display(entity_key),
                    f"→ '{self._format_key_for_display(similar_key)}' sim={similarity_score:.4f}",
                )
                stats.merged += 1

                # ── Canonical name selection ──
                # Pick the better name regardless of creation order
                canonical_key, demoted_key, swapped = self._pick_canonical_key(
                    similar_key, entity_key
                )
                if swapped:
                    existing_entity = entities[self.entity_type].pop(
                        similar_key
                    )  # same object as candidate_entity
                    # Update the entity's own name/title field
                    if self.entity_type == "people":
                        existing_entity["name"] = canonical_key
                    elif self.entity_type in ("organizations", "locations"):
                        existing_entity["name"] = (
                            canonical_key[0]
                            if isinstance(canonical_key, tuple)
                            else canonical_key
                        )
                        if isinstance(canonical_key, tuple) and len(canonical_key) > 1:
                            existing_entity["type"] = canonical_key[1]
                    elif self.entity_type == "events":
                        existing_entity["title"] = (
                            canonical_key[0]
                            if isinstance(canonical_key, tuple)
                            else canonical_key
                        )
                        if isinstance(canonical_key, tuple) and len(canonical_key) > 1:
                            existing_entity["start_date"] = canonical_key[1]
                    # Demoted name goes to alternative_names
                    self._add_alternative_name(existing_entity, demoted_key)
                    similar_key = canonical_key
                    entity_updated = True
                    log(
                        f"Canonical name: '{self._format_key_for_display(canonical_key)}' "
                        f"(demoted '{self._format_key_for_display(demoted_key)}' to alternative)",
                        level="info",
                    )
                else:
                    existing_entity = candidate_entity

                # Ensure article is linked
                existing_entity.setdefault("articles", [])
                article_exists = any(
                    a.get("article_id") == article_id
                    for a in existing_entity["articles"]
                )
                if not article_exists:
                    existing_entity["articles"].append(
                        {
                            "article_id": article_id,
                            "article_title": article_title,
                            "article_url": article_url,
                            "article_published_date": article_published_date,
                        }
                    )
                    entity_updated = True

                # Profile update or repair.
                # Normal update: new article triggers update_profile on existing profile.
                # Profile repair: matched entity with missing/empty profile gets
                # create_profile regardless of whether the article was already linked
                # (the entity is broken and needs a profile to function properly).
                needs_profile_update = (
                    not article_exists and not existing_profile_missing
                )
                needs_profile_repair = existing_profile_missing

                if needs_profile_update:
                    # --- Normal update path: existing entity has a usable profile ---
                    versioned_profile = (
                        VersionedProfile(**existing_entity["profile_versions"])
                        if ENABLE_PROFILE_VERSIONING
                        and "profile_versions" in existing_entity
                        else VersionedProfile()
                    )

                    updated_profile, versioned_profile, reflection_history = (
                        _profile_or_fallback(
                            (existing_entity["profile"], versioned_profile, []),
                            update_profile,
                            self.entity_type[:-1],
                            self._format_key_for_display(similar_key),
                            existing_entity["profile"],
                            versioned_profile,
                            article_content,
                            article_id,
                            model_type,
                            domain,
                            model=task_model("profile", model_type),
                        )
                    )
                    existing_entity["profile"] = updated_profile
                    existing_entity["profile_versions"] = (
                        versioned_profile.model_dump()
                        if ENABLE_PROFILE_VERSIONING
                        else None
                    )
                    # Defer profile embedding to batch phase 4
                    pending_profiles.append(
                        _PendingProfileEmbedding(
                            entity_ref=existing_entity,
                            profile_text=updated_profile["text"],
                        )
                    )

                    # Reflection history
                    existing_entity.setdefault("reflection_history", [])
                    existing_entity["reflection_history"].extend(reflection_history)

                    display_markdown(
                        updated_profile["text"],
                        title=f"Updated Profile: {self._format_key_for_display(similar_key)}",
                        style="yellow",
                    )

                    # Update search_embedding with latest evidence
                    existing_entity["search_embedding"] = evidence_embedding
                    existing_entity["search_embedding_model"] = emb_model_name
                    existing_entity["search_embedding_dim"] = emb_dim
                    existing_entity["search_embedding_fingerprint"] = (
                        wi.evidence_fingerprint
                    )
                    entity_updated = True

                elif needs_profile_repair:
                    # --- Merge-path fallback: create profile for matched entity
                    # that has a missing or empty profile (repair). ---
                    log(
                        f"Repairing missing profile for "
                        f"'{self._format_key_for_display(similar_key)}'",
                        level="info",
                    )
                    new_profile, new_versioned_profile, reflection_history = (
                        _profile_or_fallback(
                            ({}, VersionedProfile(), []),
                            create_profile,
                            self.entity_type[:-1],
                            self._format_key_for_display(similar_key),
                            article_content,
                            article_id,
                            model_type,
                            domain,
                            model=task_model("profile", model_type),
                        )
                    )
                    new_profile = extract_profile_text(new_profile)
                    if not new_profile or not new_profile.get("text"):
                        log_decision(
                            DecisionKind.ERROR,
                            self.entity_type,
                            self._format_key_for_display(similar_key),
                            "profile repair failed",
                        )
                        stats.errors += 1
                        # Still allow article link and alias absorption below
                    else:
                        existing_entity["profile"] = new_profile
                        existing_entity["profile_versions"] = (
                            new_versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        # Defer profile embedding to batch phase 4
                        pending_profiles.append(
                            _PendingProfileEmbedding(
                                entity_ref=existing_entity,
                                profile_text=new_profile["text"],
                            )
                        )

                        existing_entity.setdefault("reflection_history", [])
                        existing_entity["reflection_history"].extend(reflection_history)

                        # Update search_embedding with latest evidence
                        existing_entity["search_embedding"] = evidence_embedding
                        existing_entity["search_embedding_model"] = emb_model_name
                        existing_entity["search_embedding_dim"] = emb_dim
                        existing_entity["search_embedding_fingerprint"] = (
                            wi.evidence_fingerprint
                        )

                        display_markdown(
                            new_profile["text"],
                            title=f"Repaired Profile: {self._format_key_for_display(similar_key)}",
                            style="green",
                        )
                        entity_updated = True

                # Alternative names/titles if different
                if entity_key != similar_key:
                    if self._add_alternative_name(
                        existing_entity, entity_key, source_entity=entity_dict
                    ):
                        entity_updated = True

                # Absorb incoming aliases so future blocking can find them
                existing_entity.setdefault("aliases", [])
                for alias in entity_dict.get("aliases", []):
                    if alias and alias not in existing_entity["aliases"]:
                        existing_entity["aliases"].append(alias)
                        entity_updated = True

                # Keep earliest extraction timestamp
                existing_timestamp = existing_entity.get(
                    "extraction_timestamp", extraction_timestamp
                )
                earliest = min(existing_timestamp, extraction_timestamp)
                if existing_timestamp != earliest:
                    existing_entity["extraction_timestamp"] = earliest
                    entity_updated = True

                if entity_updated:
                    entities[self.entity_type][similar_key] = existing_entity

            else:
                # --- CREATE PATH (expensive profile generation happens here) ---
                # Never overwrite an existing entity that shares this key (e.g.
                # an exact-name entity whose similarity was below threshold).
                if entity_key in entities[self.entity_type]:
                    log_decision(
                        DecisionKind.SKIP,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        "key already exists, not overwriting",
                    )
                    stats.skipped += 1
                    continue

                proposed_profile, proposed_versioned_profile, reflection_history = (
                    _profile_or_fallback(
                        ({}, VersionedProfile(), []),
                        create_profile,
                        self.entity_type[:-1],
                        self._format_key_for_display(entity_key),
                        article_content,
                        article_id,
                        model_type,
                        domain,
                        model=task_model("profile", model_type),
                    )
                )

                proposed_profile = extract_profile_text(proposed_profile)
                if not proposed_profile or not proposed_profile.get("text"):
                    log_decision(
                        DecisionKind.ERROR,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        "profile generation failed",
                    )
                    stats.errors += 1
                    continue

                # Create entity with empty profile embedding (filled in phase 5)
                new_entity = self._create_new_entity(
                    entity_dict,
                    entity_key,
                    proposed_profile,
                    proposed_versioned_profile,
                    [],  # profile_embedding — deferred to batch phase 4
                    reflection_history,
                    article_id,
                    article_title,
                    article_url,
                    article_published_date,
                    extraction_timestamp,
                    embedding_model=None,
                    embedding_dim=None,
                    embedding_fingerprint=None,
                    search_embedding=evidence_embedding,
                    search_embedding_model=emb_model_name,
                    search_embedding_dim=emb_dim,
                )

                entities[self.entity_type][entity_key] = new_entity
                log_decision(
                    DecisionKind.NEW,
                    self.entity_type,
                    self._format_key_for_display(entity_key),
                )
                stats.new += 1
                display_markdown(
                    proposed_profile["text"],
                    title=f"New Profile: {self._format_key_for_display(entity_key)}",
                    style="green",
                )

                # Defer profile embedding to batch phase 4
                pending_profiles.append(
                    _PendingProfileEmbedding(
                        entity_ref=new_entity,
                        profile_text=proposed_profile["text"],
                    )
                )

        # ── Phase 4: Batch-embed all profile texts ──
        # ── Phase 5: Commit profile embedding mutations ──
        if pending_profiles:
            profile_texts = [p.profile_text for p in pending_profiles]
            prof_vecs, prof_model, prof_dim, prof_fp = batch_embed_texts(
                embedding_manager, profile_texts, embed_batch_size, "embed.profile"
            )
            for j, pending in enumerate(pending_profiles):
                vec = prof_vecs[j]
                pending.entity_ref["profile_embedding"] = vec
                pending.entity_ref["profile_embedding_model"] = prof_model
                pending.entity_ref["profile_embedding_dim"] = prof_dim or (
                    len(vec) if vec else None
                )
                pending.entity_ref["profile_embedding_fingerprint"] = prof_fp

        log(
            f"merge_{self.entity_type} done — "
            f"new={stats.new} merged={stats.merged} skipped={stats.skipped} "
            f"disputed={stats.disputed} errors={stats.errors}",
            level="success",
        )
        return stats

    def _create_new_entity(
        self,
        entity_dict: Dict[str, Any],
        entity_key: Union[str, Tuple],
        profile: Dict[str, Any],
        versioned_profile: VersionedProfile,
        profile_embedding: List[float],
        reflection_history: List[Any],
        article_id: str,
        article_title: str,
        article_url: str,
        article_published_date: Any,
        extraction_timestamp: str,
        *,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_fingerprint: Optional[str] = None,
        search_embedding: Optional[List[float]] = None,
        search_embedding_model: Optional[str] = None,
        search_embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new entity dictionary with appropriate structure for each entity type."""
        resolved_dim = embedding_dim or (
            len(profile_embedding) if profile_embedding else None
        )
        search_dim = search_embedding_dim or (
            len(search_embedding) if search_embedding else None
        )
        base_entity: Dict[str, Any] = {
            "profile": profile,
            "profile_versions": versioned_profile.model_dump()
            if ENABLE_PROFILE_VERSIONING
            else None,
            "articles": [
                {
                    "article_id": article_id,
                    "article_title": article_title,
                    "article_url": article_url,
                    "article_published_date": article_published_date,
                }
            ],
            "profile_embedding": profile_embedding,
            "profile_embedding_model": embedding_model,
            "profile_embedding_dim": resolved_dim,
            "profile_embedding_fingerprint": embedding_fingerprint
            or EmbeddingManager.make_fingerprint(embedding_model, resolved_dim),
            "search_embedding": search_embedding or [],
            "search_embedding_model": search_embedding_model,
            "search_embedding_dim": search_dim,
            "search_embedding_fingerprint": EmbeddingManager.make_fingerprint(
                search_embedding_model, search_dim
            )
            if search_embedding
            else None,
            "extraction_timestamp": extraction_timestamp,
            self.alternative_field: [],
            "aliases": entity_dict.get("aliases", []),
            "reflection_history": reflection_history or [],
        }

        if self.entity_type == "people":
            base_entity["name"] = entity_key
        elif self.entity_type in ("organizations", "locations"):
            base_entity["name"] = entity_key[0]
            base_entity["type"] = entity_key[1]
        elif self.entity_type == "events":
            base_entity["title"] = entity_key[0]
            base_entity["start_date"] = entity_key[1]
            base_entity["description"] = entity_dict.get("description", "")
            base_entity["event_type"] = entity_dict.get("event_type", "")
            base_entity["end_date"] = entity_dict.get("end_date", "")
            base_entity["is_fuzzy_date"] = entity_dict.get("is_fuzzy_date", False)
            base_entity["tags"] = entity_dict.get("tags", [])

        return base_entity


# Convenience creators (unchanged behavior)
def create_people_merger() -> EntityMerger:
    return EntityMerger("people")


def create_organizations_merger() -> EntityMerger:
    return EntityMerger("organizations")


def create_locations_merger() -> EntityMerger:
    return EntityMerger("locations")


def create_events_merger() -> EntityMerger:
    return EntityMerger("events")
