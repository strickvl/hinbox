"""Candidate search for entity merging: lexical blocking + embedding similarity.

Given a new entity, these helpers find the existing entity it most likely
duplicates. RapidFuzz name matching shortlists candidates cheaply, then cosine
similarity on stored embeddings picks the best one.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from rapidfuzz import fuzz
from rapidfuzz import process as rfprocess

from src.constants import (
    ENTITY_SINGULAR,
    MERGE_MAX_SIMILAR_CHECKS,
    SIMILARITY_THRESHOLD,
)
from src.logging_config import log
from src.utils.embeddings.similarity import compute_similarity
from src.utils.name_variants import (
    compute_acronym,
    is_acronym_form,
    is_name_contained,
    is_shouting_name,
    lone_name_in_full_name,
    names_are_variants,
    names_likely_same,
    person_name_tokens,
    score_canonical_name,
)


def lexical_text(key: Union[str, Tuple]) -> str:
    """Extract plain text from an entity key for lexical comparison.

    For events (title, date) tuples, returns title only — the date is
    part of the identity key but shouldn't drive fuzzy string matching.
    """
    if isinstance(key, tuple):
        return str(key[0])
    return str(key)


def get_search_embedding(entity: Dict[str, Any]) -> List[float]:
    """Return the best embedding for similarity search.

    Prefers search_embedding (evidence-derived) over profile_embedding
    (LLM-narrative-derived) for apples-to-apples comparison with incoming
    evidence queries. Falls back to profile_embedding for backward
    compatibility with entities created before search_embedding was stored.
    """
    return entity.get("search_embedding") or entity.get("profile_embedding", [])


def get_search_embedding_meta(
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


def embeddings_compatible(
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
    existing_model, existing_dim = get_search_embedding_meta(existing_entity)
    if existing_dim != new_dim:
        return False

    if new_model and existing_model and new_model != existing_model:
        return False

    return True


def collect_entity_variant_texts(
    key: Union[str, Tuple],
    entity: Dict[str, Any],
    alternative_field: str,
) -> List[str]:
    """Collect all name variant strings for an entity (canonical + aliases + alternatives).

    Used to build a richer lexical index for blocking.
    """
    texts = [lexical_text(key)]

    # Aliases (uniform string list added by QC consolidation)
    for alias in entity.get("aliases", []):
        if isinstance(alias, str) and alias.strip():
            texts.append(alias.strip())

    # Alternative names/titles (legacy format varies by entity type)
    for alt in entity.get(alternative_field, []):
        if isinstance(alt, str) and alt.strip():
            texts.append(alt.strip())
        elif isinstance(alt, dict):
            name = alt.get("name", alt.get("title", ""))
            if name and isinstance(name, str):
                texts.append(name.strip())

    # Derived acronym from canonical name
    acronym = compute_acronym(lexical_text(key))
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


def entity_name_texts(
    key: Union[str, Tuple], entity: Dict[str, Any], alternative_field: str
) -> List[str]:
    """Canonical, alias and alternative names as written (no derived acronyms)."""
    texts = [lexical_text(key), *entity.get("aliases", [])]
    for alt in entity.get(alternative_field, []):
        texts.append(
            alt.get("name", alt.get("title", "")) if isinstance(alt, dict) else alt
        )
    return [t for t in texts if isinstance(t, str) and t.strip()]


def _shares_name_variant(
    entity_type: str, names_a: List[str], names_b: List[str], groups: List[List[str]]
) -> bool:
    return any(
        names_are_variants(a, b, entity_type=entity_type, equivalence_groups=groups)
        for a in names_a
        for b in names_b
    )


def _shares_lone_name(names_a: List[str], names_b: List[str]) -> bool:
    return any(lone_name_in_full_name(a, b) for a in names_a for b in names_b)


def lexical_block(
    entity_type: str,
    alternative_field: str,
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
    query_variants = [lexical_text(query_key)]
    if query_entity:
        query_variants = collect_entity_variant_texts(
            query_key, query_entity, alternative_field
        )

    # Build flattened candidate index: (text, key) pairs
    # Each candidate contributes multiple texts (canonical + variants)
    choice_texts: List[str] = []
    choice_to_key: List[Union[str, Tuple]] = []

    for ck in candidate_keys:
        if entities_data:
            entity_data = entities_data.get(ck, {})
            variants = collect_entity_variant_texts(ck, entity_data, alternative_field)
        else:
            variants = [lexical_text(ck)]

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
        query_text_lower = lexical_text(query_key).lower()
        for group in equivalence_groups:
            group_lower = {g.lower() for g in group}
            if query_text_lower in group_lower:
                for ck in candidate_keys:
                    ck_text_lower = lexical_text(ck).lower()
                    if ck_text_lower in group_lower:
                        matched_keys.add(ck)
                    # Also check candidate aliases
                    if entities_data:
                        entity_data = entities_data.get(ck, {})
                        for variant in collect_entity_variant_texts(
                            ck, entity_data, alternative_field
                        ):
                            if variant.lower() in group_lower:
                                matched_keys.add(ck)
                break  # Only one group can match

    # names_likely_same bypass: if deterministic heuristics say they match,
    # include the candidate even if WRatio was too low
    query_display = lexical_text(query_key)
    for ck in candidate_keys:
        if ck in matched_keys:
            continue
        if names_likely_same(
            query_display,
            lexical_text(ck),
            entity_type=entity_type,
            equivalence_groups=equivalence_groups or [],
        ):
            matched_keys.add(ck)

    # Respect max_candidates
    result = [ck for ck in candidate_keys if ck in matched_keys]
    return result[:max_candidates]


def find_similar_entity(
    entity_type: str,
    alternative_field: str,
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
    exclude: Tuple = (),
) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
    """Find the most similar entity in the entities database using embedding similarity.

    Keys in ``exclude`` (candidates the match check already rejected) are
    skipped, so calling again gives the next-closest candidate.

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
    if not entity_embedding or not entities[entity_type]:
        return None, None

    singular = entity_type[:-1]
    new_dim = embedding_dim or len(entity_embedding)
    best_match: Optional[Union[str, Tuple]] = None
    best_score = 0.0

    # Exact key match first
    if entity_key in entities[entity_type] and entity_key not in exclude:
        existing_entity = entities[entity_type][entity_key]
        existing_emb = get_search_embedding(existing_entity)
        if existing_emb:
            if embeddings_compatible(new_dim, existing_entity, embedding_model):
                similarity = compute_similarity(entity_embedding, existing_emb)
                log(
                    f"Exact match for {singular} '{lexical_text(entity_key)}' with similarity: {similarity:.4f}",
                    level="info",
                )
                if similarity >= similarity_threshold:
                    return entity_key, similarity
            else:
                # Exact key but incompatible embeddings — let match-check decide
                log(
                    f"Exact key match for '{lexical_text(entity_key)}' "
                    f"but embedding model/dim mismatch — deferring to match-check",
                    level="warning",
                )
                return entity_key, 1.0

    # Build candidate list (excluding exact key and entities without embeddings)
    all_candidates = [
        k
        for k, e in entities[entity_type].items()
        if k != entity_key and k not in exclude and get_search_embedding(e)
    ]

    # Apply lexical blocking if configured
    lb = lexical_blocking_config or {}
    if lb.get("enabled", False) and all_candidates:
        pre_count = len(all_candidates)
        all_candidates = lexical_block(
            entity_type,
            alternative_field,
            entity_key,
            all_candidates,
            threshold=lb.get("threshold", 60),
            max_candidates=lb.get("max_candidates", 50),
            entities_data=entities.get(entity_type),
            query_entity=query_entity,
            equivalence_groups=equivalence_groups,
        )
        log(
            f"Lexical blocking: {pre_count} → {len(all_candidates)} candidates "
            f"for '{lexical_text(entity_key)}'",
            level="info",
        )

    # Run cosine similarity on (shortlisted) candidates
    for existing_key in all_candidates:
        existing_entity = entities[entity_type][existing_key]
        if not embeddings_compatible(new_dim, existing_entity, embedding_model):
            continue
        existing_emb = get_search_embedding(existing_entity)
        similarity = compute_similarity(entity_embedding, existing_emb)
        if similarity > best_score:
            best_score = similarity
            best_match = existing_key

    if best_match and best_score >= similarity_threshold:
        log(
            f"Found similar {singular}: '{lexical_text(best_match)}' for '{lexical_text(entity_key)}' with similarity: {best_score:.4f}",
            level="warning",
        )
        return best_match, best_score

    # Same-name fallback. Profiles written from different articles about the
    # same place ("Miami-Dade" in a sports story vs a voting story) can sit far
    # below the threshold, so a candidate whose name is a spelling variant
    # ("Guantánamo Bay" / "Guantanamo Bay", "Trump" / "Donald Trump") still goes
    # to the match check, which decides whether it is really the same entity.
    # Searches every entity, not just the lexical shortlist, which is capped.
    return name_variant_candidate(
        entity_type,
        alternative_field,
        entity_key,
        entity_embedding,
        entities,
        embedding_model=embedding_model,
        embedding_dim=new_dim,
        query_entity=query_entity,
        equivalence_groups=equivalence_groups,
        exclude=exclude,
    )


def next_candidate(
    entity_type: str,
    alternative_field: str,
    entity_key: Union[str, Tuple],
    entity_embedding: List[float],
    entities: Dict[str, Dict],
    similarity_threshold: float,
    rejected: List[Union[str, Tuple]],
    *,
    max_similar: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
    """The candidate to check after the match check rejected every key in ``rejected``.

    While fewer than ``max_similar`` have been rejected, this is the next-closest
    candidate above the threshold (or, when none is left, the best name-variant
    candidate). After that, only the best name-variant candidate. Example: an
    incoming "Resignation of Nashiri's defense lawyers" is closest to a
    sibling event from an earlier article in the same story; the match check
    rejects the sibling, and the real event, second-closest, is checked next.
    """
    exclude = tuple(rejected)
    if max_similar is None:
        max_similar = MERGE_MAX_SIMILAR_CHECKS
    if len(rejected) < max_similar:
        return find_similar_entity(
            entity_type,
            alternative_field,
            entity_key,
            entity_embedding,
            entities,
            similarity_threshold,
            exclude=exclude,
            **kwargs,
        )
    kwargs.pop("lexical_blocking_config", None)
    return name_variant_candidate(
        entity_type,
        alternative_field,
        entity_key,
        entity_embedding,
        entities,
        exclude=exclude,
        **kwargs,
    )


def rejected_note(entity_type: str, rejected: List[Union[str, Tuple]]) -> str:
    """A note for the match check on a second or third candidate; "" on the first.

    Checking more candidates gives the match check more chances to say a
    wrong "yes". Replaying the 7 Nov run, all 7 wrong merges it added were a
    sibling event from the same story: "Judge rejects the lawyers'
    resignations" joined into "Resignation of Nashiri's lawyers". The note
    says a closer candidate was already judged different and asks for the
    same entity, not a related one.
    """
    if not rejected:
        return ""
    singular = ENTITY_SINGULAR.get(entity_type, "entity")
    names = ", ".join(f'"{lexical_text(k)}"' for k in rejected)
    return (
        f"\n\nNote: this {singular} was already judged a different {singular}"
        f" from {names}, a closer match from the same coverage. Related"
        f" {entity_type} from the same story (an action and the response to it,"
        " one step and the next step of a process, two similar incidents) are"
        f" different. Answer yes only if the evidence shows this is the same"
        f" {singular}, not just a related one."
    )


def name_variant_candidate(
    entity_type: str,
    alternative_field: str,
    entity_key: Union[str, Tuple],
    entity_embedding: List[float],
    entities: Dict[str, Dict],
    *,
    embedding_model: Optional[str] = None,
    embedding_dim: Optional[int] = None,
    query_entity: Optional[Dict[str, Any]] = None,
    equivalence_groups: Optional[List[List[str]]] = None,
    exclude: Tuple = (),
) -> Tuple[Optional[Union[str, Tuple]], Optional[float]]:
    """The candidate whose name is a spelling variant of the query's, highest cosine first.

    For people, when no candidate is a spelling variant, a lone first or middle
    name ("Lara" / "Lara Tur") still counts if exactly one person in the table
    fits: with two ("Daniel Harding", "Daniel Ellsberg") there's no telling
    which one a bare "Daniel" is, so neither goes to the match check.

    Keys in ``exclude`` (candidates the match check already rejected) are skipped.
    """
    if not entity_embedding:
        return None, None
    new_dim = embedding_dim or len(entity_embedding)
    groups = equivalence_groups or []
    query_names = entity_name_texts(entity_key, query_entity or {}, alternative_field)
    best_key: Optional[Union[str, Tuple]] = None
    best_score = -1.0
    lone_name_keys: List[Union[str, Tuple]] = []
    for key, existing in entities[entity_type].items():
        names = entity_name_texts(key, existing, alternative_field)
        if not _shares_name_variant(entity_type, query_names, names, groups):
            if entity_type == "people" and _shares_lone_name(query_names, names):
                lone_name_keys.append(key)
            continue
        if key in exclude or not get_search_embedding(existing):
            continue
        if not embeddings_compatible(new_dim, existing, embedding_model):
            continue
        score = compute_similarity(entity_embedding, get_search_embedding(existing))
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None and len(lone_name_keys) == 1:
        key = lone_name_keys[0]
        existing = entities[entity_type][key]
        emb = get_search_embedding(existing)
        if (
            key not in exclude
            and emb
            and embeddings_compatible(new_dim, existing, embedding_model)
        ):
            best_key, best_score = key, compute_similarity(entity_embedding, emb)
    if best_key is None:
        return None, None
    log(
        f"Name-variant {entity_type[:-1]} '{lexical_text(best_key)}' for "
        f"'{lexical_text(entity_key)}' (similarity {best_score:.4f}); "
        "sending to match check",
        level="info",
    )
    return best_key, best_score


def pick_canonical_key(
    existing_key: Union[str, Tuple], incoming_key: Union[str, Tuple]
) -> Tuple[Union[str, Tuple], Union[str, Tuple], bool]:
    """Decide which entity key should be canonical after a merge.

    Returns (canonical_key, demoted_key, swapped) where *swapped* is True when
    the incoming name is better and the entity should be re-keyed.
    """
    existing_name = lexical_text(existing_key)
    incoming_name = lexical_text(incoming_key)

    # Same name in another case: keep the existing one, unless it is in ALL
    # CAPS ("GUANTANAMO BAY NAVY BASE") and the incoming one isn't.
    if existing_name.lower() == incoming_name.lower():
        if is_shouting_name(existing_name) and not is_shouting_name(incoming_name):
            return incoming_key, existing_key, True
        return existing_key, incoming_key, False

    existing_score = score_canonical_name(existing_name)
    incoming_score = score_canonical_name(incoming_name)

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


def surname_note(
    entity_type: str, new_name: str, candidate_name: str, entities: Dict[str, Dict]
) -> str:
    """A note for the match check when a person is named by surname alone
    ("Rubio") and the candidate ("Marco Rubio") is the only person in the
    table with that surname; "" otherwise.

    News stories name a well-known person by surname alone. The match check
    can't see the rest of the table, so on the 7 Nov Luna run it rejected
    "Rubio" against "Marco Rubio": "the available evidence does not establish
    that the unnamed Rubio is Marco Rubio." With two people of that surname
    ("Jennifer Parrella", "Keith Parrella") no note is added.
    """
    word = person_name_tokens(new_name)
    if (
        entity_type != "people"
        or len(word) != 1
        or person_name_tokens(candidate_name)[-1:] != word
    ):
        return ""
    if len(person_name_tokens(candidate_name)) < 2:
        return ""
    others = [
        key
        for key in entities.get("people", {})
        if isinstance(key, str)
        and key != candidate_name
        and len(tokens := person_name_tokens(key)) >= 2
        and tokens[-1] == word[0]
    ]
    if others:
        return ""
    return (
        f'\n\nNOTE: The article names this person only by the surname "{new_name}". '
        f"{candidate_name} is the only person in the database with that surname."
    )
