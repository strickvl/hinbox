"""Generic entity merger classes to eliminate code duplication."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from src.config_loader import DomainConfig
from src.constants import (
    ENABLE_PROFILE_VERSIONING,
    MERGE_GRAY_BAND_DELTA,
    MERGE_UNCERTAIN_CONFIDENCE_CUTOFF,
    SIMILARITY_THRESHOLD,
)
from src.engine.match_checker import cloud_model_check_match, local_model_check_match
from src.engine.merge_dispute_agent import MergeDisputeAction, run_merge_dispute_agent
from src.engine.profiles import VersionedProfile, create_profile, update_profile
from src.logging_config import (
    DecisionKind,
    display_markdown,
    get_logger,
    log,
    log_decision,
)
from src.utils.embeddings.manager import EmbeddingManager
from src.utils.embeddings.similarity import compute_similarity, get_embedding_manager
from src.utils.name_variants import (
    compute_acronym,
    is_acronym_form,
    is_low_quality_name,
    is_name_contained,
    names_likely_same,
    score_canonical_name,
)
from src.utils.profiles import extract_profile_text

# Module-specific logger
logger = get_logger("mergers")


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

    @staticmethod
    def _get_search_embedding(entity: Dict[str, Any]) -> List[float]:
        """Return the best embedding for similarity search.

        Prefers search_embedding (evidence-derived) over profile_embedding
        (LLM-narrative-derived) for apples-to-apples comparison with incoming
        evidence queries. Falls back to profile_embedding for backward
        compatibility with entities created before search_embedding was stored.
        """
        return entity.get("search_embedding") or entity.get("profile_embedding", [])

    @staticmethod
    def _get_search_embedding_meta(
        entity: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[int]]:
        """Return (model, dim) for the best available search embedding."""
        if entity.get("search_embedding"):
            emb = entity["search_embedding"]
            model = entity.get("search_embedding_model")
            dim = entity.get("search_embedding_dim") or (len(emb) if emb else None)
            return model, dim
        emb = entity.get("profile_embedding", [])
        model = entity.get("profile_embedding_model")
        dim = entity.get("profile_embedding_dim") or (len(emb) if emb else None)
        return model, dim

    @staticmethod
    def _embeddings_compatible(
        new_dim: int,
        existing_entity: Dict[str, Any],
        new_model: Optional[str] = None,
    ) -> bool:
        """Check whether two embeddings can be meaningfully compared.

        Prefers search_embedding metadata when available, falls back to
        profile_embedding. Incompatible when:
        - Dimensions differ (cosine similarity would be 0.0 anyway)
        - Both model names are known and they differ
        """
        existing_model, existing_dim = EntityMerger._get_search_embedding_meta(
            existing_entity
        )
        if existing_dim != new_dim:
            return False

        if new_model and existing_model and new_model != existing_model:
            return False

        return True

    def _lexical_text(self, key: Union[str, Tuple]) -> str:
        """Extract plain text from an entity key for lexical comparison.

        For events (title, date) tuples, returns title only — the date is
        part of the identity key but shouldn't drive fuzzy string matching.
        """
        if isinstance(key, tuple):
            return str(key[0])
        return str(key)

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
        existing_name = self._lexical_text(existing_key)
        incoming_name = self._lexical_text(incoming_key)

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

    def _collect_entity_variant_texts(
        self,
        key: Union[str, Tuple],
        entity: Dict[str, Any],
    ) -> List[str]:
        """Collect all name variant strings for an entity (canonical + aliases + alternatives).

        Used to build a richer lexical index for blocking.
        """
        texts = [self._lexical_text(key)]

        # Aliases (uniform string list added by QC consolidation)
        for alias in entity.get("aliases", []):
            if isinstance(alias, str) and alias.strip():
                texts.append(alias.strip())

        # Alternative names/titles (legacy format varies by entity type)
        alt_field = self.alternative_field
        for alt in entity.get(alt_field, []):
            if isinstance(alt, str) and alt.strip():
                texts.append(alt.strip())
            elif isinstance(alt, dict):
                name = alt.get("name", alt.get("title", ""))
                if name and isinstance(name, str):
                    texts.append(name.strip())

        # Derived acronym from canonical name
        canonical = self._lexical_text(key)
        acronym = compute_acronym(canonical)
        if acronym:
            texts.append(acronym)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for t in texts:
            lower = t.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(t)

        return unique

    def _lexical_block(
        self,
        query_key: Union[str, Tuple],
        candidate_keys: List[Union[str, Tuple]],
        threshold: int = 60,
        max_candidates: int = 50,
        *,
        entities_data: Optional[Dict[Union[str, Tuple], Dict[str, Any]]] = None,
        query_entity: Optional[Dict[str, Any]] = None,
        equivalence_groups: Optional[List[List[str]]] = None,
    ) -> List[Union[str, Tuple]]:
        """Pre-filter candidates using RapidFuzz fuzzy string matching + variant-aware bypass.

        When entities_data is provided, the blocking considers aliases,
        alternative names, and derived acronyms for each candidate entity,
        plus equivalence group membership from config.

        Returns a shortlist of candidate keys whose lexical similarity to
        the query (or any of its variants) is >= threshold, capped at max_candidates.
        """
        if not candidate_keys:
            return []

        matched_keys: set = set()

        # Build query variants (canonical + aliases + acronym)
        query_variants = [self._lexical_text(query_key)]
        if query_entity:
            query_variants = self._collect_entity_variant_texts(query_key, query_entity)

        # Build flattened candidate index: (text, key) pairs
        # Each candidate contributes multiple texts (canonical + variants)
        choice_texts: List[str] = []
        choice_to_key: List[Union[str, Tuple]] = []

        for ck in candidate_keys:
            if entities_data:
                entity_data = entities_data.get(ck, {})
                variants = self._collect_entity_variant_texts(ck, entity_data)
            else:
                variants = [self._lexical_text(ck)]

            for variant_text in variants:
                choice_texts.append(variant_text)
                choice_to_key.append(ck)

        # Run RapidFuzz against flattened index for each query variant
        for q_text in query_variants:
            results = rfprocess.extract(
                q_text,
                choice_texts,
                scorer=fuzz.WRatio,
                score_cutoff=threshold,
                limit=max_candidates,
            )
            for _, _, idx in results:
                matched_keys.add(choice_to_key[idx])

        # Equivalence group bypass: if query name is in an equivalence group,
        # add all candidates whose names are in the same group
        if equivalence_groups:
            query_text_lower = self._lexical_text(query_key).lower()
            for group in equivalence_groups:
                group_lower = {g.lower() for g in group}
                if query_text_lower in group_lower:
                    for ck in candidate_keys:
                        ck_text_lower = self._lexical_text(ck).lower()
                        if ck_text_lower in group_lower:
                            matched_keys.add(ck)
                        # Also check candidate aliases
                        if entities_data:
                            entity_data = entities_data.get(ck, {})
                            for variant in self._collect_entity_variant_texts(
                                ck, entity_data
                            ):
                                if variant.lower() in group_lower:
                                    matched_keys.add(ck)
                    break  # Only one group can match

        # names_likely_same bypass: if deterministic heuristics say they match,
        # include the candidate even if WRatio was too low
        query_display = self._lexical_text(query_key)
        for ck in candidate_keys:
            if ck in matched_keys:
                continue
            ck_display = self._lexical_text(ck)
            if names_likely_same(
                query_display,
                ck_display,
                entity_type=self.entity_type,
                equivalence_groups=equivalence_groups or [],
            ):
                matched_keys.add(ck)

        # Respect max_candidates
        result = [ck for ck in candidate_keys if ck in matched_keys]
        return result[:max_candidates]

    def find_similar_entity(
        self,
        entity_key: Union[str, Tuple],
        entity_embedding: List[float],
        entities: Dict[str, Dict],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        *,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        lexical_blocking_config: Optional[Dict[str, Any]] = None,
        query_entity: Optional[Dict[str, Any]] = None,
        equivalence_groups: Optional[List[List[str]]] = None,
    ) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
        """Find the most similar entity in the entities database using embedding similarity.

        When embedding_model and embedding_dim are provided, entities whose stored
        embeddings are from an incompatible model/dimension are skipped during the
        scan (their similarity would be meaningless).  For exact-key matches with
        incompatible embeddings, we return a forced similarity of 1.0 so the
        downstream LLM match-check can still confirm or deny the merge — this
        prevents accidental duplicates after a model change.

        When lexical_blocking_config is provided and enabled, a variant-aware
        RapidFuzz pre-filter reduces the candidate set before running cosine
        similarity (O(n) → O(k)). The blocking now considers aliases, alternative
        names, derived acronyms, and configured equivalence groups.
        """
        if not entity_embedding or not entities[self.entity_type]:
            return None, None

        new_dim = embedding_dim or len(entity_embedding)
        best_match: Optional[Union[str, Tuple]] = None
        best_score = 0.0

        # Exact key match first
        if entity_key in entities[self.entity_type]:
            existing_entity = entities[self.entity_type][entity_key]
            existing_emb = self._get_search_embedding(existing_entity)
            if existing_emb:
                if self._embeddings_compatible(
                    new_dim, existing_entity, embedding_model
                ):
                    similarity = compute_similarity(entity_embedding, existing_emb)
                    log(
                        f"Exact match for {self.entity_type[:-1]} '{self._format_key_for_display(entity_key)}' with similarity: {similarity:.4f}",
                        level="info",
                    )
                    if similarity >= similarity_threshold:
                        return entity_key, similarity
                else:
                    # Exact key but incompatible embeddings — let match-check decide
                    log(
                        f"Exact key match for '{self._format_key_for_display(entity_key)}' "
                        f"but embedding model/dim mismatch — deferring to match-check",
                        level="warning",
                    )
                    return entity_key, 1.0

        # Build candidate list (excluding exact key and entities without embeddings)
        all_candidates = [
            k
            for k, e in entities[self.entity_type].items()
            if k != entity_key and self._get_search_embedding(e)
        ]

        # Apply lexical blocking if configured
        lb = lexical_blocking_config or {}
        if lb.get("enabled", False) and all_candidates:
            pre_count = len(all_candidates)
            all_candidates = self._lexical_block(
                entity_key,
                all_candidates,
                threshold=lb.get("threshold", 60),
                max_candidates=lb.get("max_candidates", 50),
                entities_data=entities.get(self.entity_type),
                query_entity=query_entity,
                equivalence_groups=equivalence_groups,
            )
            log(
                f"Lexical blocking: {pre_count} → {len(all_candidates)} candidates "
                f"for '{self._format_key_for_display(entity_key)}'",
                level="info",
            )

        # Run cosine similarity on (shortlisted) candidates
        for existing_key in all_candidates:
            existing_entity = entities[self.entity_type][existing_key]
            if not self._embeddings_compatible(
                new_dim, existing_entity, embedding_model
            ):
                continue
            existing_emb = self._get_search_embedding(existing_entity)
            similarity = compute_similarity(entity_embedding, existing_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = existing_key

        if best_match and best_score >= similarity_threshold:
            log(
                f"Found similar {self.entity_type[:-1]}: '{self._format_key_for_display(best_match)}' for '{self._format_key_for_display(entity_key)}' with similarity: {best_score:.4f}",
                level="warning",
            )
            return best_match, best_score

        return None, None

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

    def _extract_context_windows(
        self,
        article_content: str,
        needles: List[str],
        window_chars: int = 240,
        max_windows: int = 3,
    ) -> str:
        """Extract text windows around entity mentions in the article.

        Finds case-insensitive occurrences of each needle in article_content
        and returns up to max_windows non-overlapping snippets of ~window_chars
        each. Falls back to the first N characters of the article if no
        mentions are found.
        """
        if not article_content:
            return ""

        # Normalise needles: strip, dedupe, drop empties
        seen: set = set()
        clean_needles: List[str] = []
        for n in needles:
            n = n.strip()
            if n and n.lower() not in seen:
                seen.add(n.lower())
                clean_needles.append(n)

        if not clean_needles:
            return article_content[: window_chars * max_windows]

        # Collect match positions across all needles
        positions: List[Tuple[int, int]] = []
        lower_content = article_content.lower()
        for needle in clean_needles:
            for m in re.finditer(re.escape(needle.lower()), lower_content):
                positions.append((m.start(), m.end()))

        positions.sort()

        if not positions:
            return article_content[: window_chars * max_windows]

        # Build non-overlapping windows
        half = window_chars // 2
        windows: List[str] = []
        last_end = -1

        for start, end in positions:
            if len(windows) >= max_windows:
                break
            win_start = max(0, start - half)
            win_end = min(len(article_content), end + half)
            if win_start < last_end:
                continue  # overlaps previous window
            windows.append(article_content[win_start:win_end].strip())
            last_end = win_end

        return "\n\n---\n\n".join(windows)

    def _build_evidence_text(
        self,
        entity_key: Union[str, Tuple],
        entity_dict: Dict[str, Any],
        article_content: str,
        max_chars: int = 1500,
        window_chars: int = 240,
        max_windows: int = 3,
    ) -> str:
        """Build a deterministic pseudo-profile from entity data + article context.

        This is the cheap alternative to create_profile() — no LLM calls,
        just string concatenation from extractor-provided structured fields
        plus context windows around entity mentions in the article.
        """
        # Assemble search needles from key + any alternative names
        needles: List[str] = []
        if isinstance(entity_key, tuple):
            needles.append(str(entity_key[0]))
        else:
            needles.append(str(entity_key))

        # Include extractor-provided alternates when present
        alt_field = self.alternative_field
        for alt in entity_dict.get(alt_field, []):
            if isinstance(alt, dict):
                needles.append(str(alt.get("name", alt.get("title", ""))))
            elif isinstance(alt, str):
                needles.append(alt)

        context = self._extract_context_windows(
            article_content, needles, window_chars=window_chars, max_windows=max_windows
        )

        # Build header by entity type
        parts: List[str] = []

        if self.entity_type == "people":
            parts.append(f"Name: {entity_key}")
        elif self.entity_type in ("organizations", "locations"):
            name = entity_key[0] if isinstance(entity_key, tuple) else str(entity_key)
            etype = (
                entity_key[1]
                if isinstance(entity_key, tuple) and len(entity_key) > 1
                else ""
            )
            parts.append(f"Name: {name}")
            if etype:
                parts.append(f"Type: {etype}")
        elif self.entity_type == "events":
            title = entity_key[0] if isinstance(entity_key, tuple) else str(entity_key)
            start_date = (
                entity_key[1]
                if isinstance(entity_key, tuple) and len(entity_key) > 1
                else ""
            )
            parts.append(f"Title: {title}")
            if start_date:
                parts.append(f"Start date: {start_date}")
            event_type = entity_dict.get("event_type", "")
            if event_type:
                parts.append(f"Event type: {event_type}")
            description = entity_dict.get("description", "")
            if description:
                parts.append(f"Description: {description}")

        if context:
            parts.append(f"\nCONTEXT:\n{context}")

        text = "\n".join(parts)
        return text[:max_chars]

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

        for entity_dict in extracted_entities:
            entity_key = self._extract_key(entity_dict)
            if not entity_key or (
                isinstance(entity_key, str) and not entity_key.strip()
            ):
                log_decision(
                    DecisionKind.ERROR, self.entity_type, "(empty key)", "empty key"
                )
                stats.errors += 1
                continue
            entity_updated = False

            # --- Cheap evidence text (no LLM) ---
            evidence_text = self._build_evidence_text(
                entity_key,
                entity_dict,
                article_content,
                max_chars=evidence_cfg["max_chars"],
                window_chars=evidence_cfg["window_chars"],
                max_windows=evidence_cfg["max_windows"],
            )

            # --- Embed evidence text ---
            emb_result = embedding_manager.embed_text_result_sync(evidence_text)
            evidence_embedding = (
                emb_result.embeddings[0] if emb_result.embeddings else []
            )
            emb_model_name = emb_result.model
            emb_dim = emb_result.dimension or (
                len(evidence_embedding) if evidence_embedding else None
            )

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
                existing_profile_text = (
                    entities[self.entity_type][similar_key]
                    .get("profile", {})
                    .get("text", "")
                )
                if not existing_profile_text:
                    log_decision(
                        DecisionKind.ERROR,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        f"no profile text on candidate '{self._format_key_for_display(similar_key)}'",
                    )
                    stats.errors += 1
                    continue

                # Match check: pass evidence_text as new_profile_text
                if model_type == "ollama":
                    result = local_model_check_match(
                        self._format_key_for_display(entity_key),
                        self._format_key_for_display(similar_key),
                        evidence_text,
                        existing_profile_text,
                        entity_type=self.entity_type,
                    )
                else:
                    result = cloud_model_check_match(
                        self._format_key_for_display(entity_key),
                        self._format_key_for_display(similar_key),
                        evidence_text,
                        existing_profile_text,
                        entity_type=self.entity_type,
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
                    )
                    should_merge = dispute_decision.action == MergeDisputeAction.MERGE

                if not should_merge:
                    log_decision(
                        DecisionKind.SKIP,
                        self.entity_type,
                        self._format_key_for_display(entity_key),
                        f"vs '{self._format_key_for_display(similar_key)}' sim={similarity_score:.4f}",
                    )
                    stats.skipped += 1
                    continue

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
                    existing_entity = entities[self.entity_type].pop(similar_key)
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
                    existing_entity = entities[self.entity_type][similar_key]

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

                    # Update profile
                    if "profile" in existing_entity:
                        # Load or create versioned profile
                        versioned_profile = (
                            VersionedProfile(**existing_entity["profile_versions"])
                            if ENABLE_PROFILE_VERSIONING
                            and "profile_versions" in existing_entity
                            else VersionedProfile()
                        )

                        updated_profile, versioned_profile, reflection_history = (
                            update_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                existing_entity["profile"],
                                versioned_profile,
                                article_content,
                                article_id,
                                model_type,
                                domain,
                            )
                        )
                        existing_entity["profile"] = updated_profile
                        existing_entity["profile_versions"] = (
                            versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        upd_result = embedding_manager.embed_text_result_sync(
                            updated_profile["text"]
                        )
                        upd_vec = (
                            upd_result.embeddings[0] if upd_result.embeddings else []
                        )
                        existing_entity["profile_embedding"] = upd_vec
                        existing_entity["profile_embedding_model"] = upd_result.model
                        existing_entity["profile_embedding_dim"] = (
                            upd_result.dimension or (len(upd_vec) if upd_vec else None)
                        )
                        existing_entity["profile_embedding_fingerprint"] = (
                            EmbeddingManager.fingerprint_from_result(upd_result)
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
                            EmbeddingManager.fingerprint_from_result(emb_result)
                        )
                    else:
                        new_profile, new_versioned_profile, reflection_history = (
                            create_profile(
                                self.entity_type[:-1],
                                self._format_key_for_display(similar_key),
                                article_content,
                                article_id,
                                model_type,
                                domain,
                            )
                        )
                        existing_entity["profile"] = new_profile
                        existing_entity["profile_versions"] = (
                            new_versioned_profile.model_dump()
                            if ENABLE_PROFILE_VERSIONING
                            else None
                        )
                        new_emb_result = embedding_manager.embed_text_result_sync(
                            new_profile["text"]
                        )
                        new_vec = (
                            new_emb_result.embeddings[0]
                            if new_emb_result.embeddings
                            else []
                        )
                        existing_entity["profile_embedding"] = new_vec
                        existing_entity["profile_embedding_model"] = (
                            new_emb_result.model
                        )
                        existing_entity["profile_embedding_dim"] = (
                            new_emb_result.dimension
                            or (len(new_vec) if new_vec else None)
                        )
                        existing_entity["profile_embedding_fingerprint"] = (
                            EmbeddingManager.fingerprint_from_result(new_emb_result)
                        )

                        existing_entity.setdefault("reflection_history", [])
                        existing_entity["reflection_history"].extend(reflection_history)

                        # Update search_embedding with latest evidence
                        existing_entity["search_embedding"] = evidence_embedding
                        existing_entity["search_embedding_model"] = emb_model_name
                        existing_entity["search_embedding_dim"] = emb_dim
                        existing_entity["search_embedding_fingerprint"] = (
                            EmbeddingManager.fingerprint_from_result(emb_result)
                        )

                        display_markdown(
                            new_profile["text"],
                            title=f"New Profile: {self._format_key_for_display(similar_key)}",
                            style="green",
                        )

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
                proposed_profile, proposed_versioned_profile, reflection_history = (
                    create_profile(
                        self.entity_type[:-1],
                        self._format_key_for_display(entity_key),
                        article_content,
                        article_id,
                        model_type,
                        domain,
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

                # Embed the FULL profile (this becomes the stored embedding)
                prof_emb_result = embedding_manager.embed_text_result_sync(
                    proposed_profile["text"]
                )
                profile_embedding = (
                    prof_emb_result.embeddings[0] if prof_emb_result.embeddings else []
                )
                prof_model = prof_emb_result.model
                prof_dim = prof_emb_result.dimension or (
                    len(profile_embedding) if profile_embedding else None
                )
                prof_fingerprint = EmbeddingManager.fingerprint_from_result(
                    prof_emb_result
                )

                new_entity = self._create_new_entity(
                    entity_dict,
                    entity_key,
                    proposed_profile,
                    proposed_versioned_profile,
                    profile_embedding,
                    reflection_history,
                    article_id,
                    article_title,
                    article_url,
                    article_published_date,
                    extraction_timestamp,
                    embedding_model=prof_model,
                    embedding_dim=prof_dim,
                    embedding_fingerprint=prof_fingerprint,
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
