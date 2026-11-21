"""The steps that keep a profile's sections in order after it is written.

``run_section_steps()`` is called by ``profiles.update_profile()`` after
every update, with the update's model:

1. File the update's new pieces under their starter headings when the
   profile was reorganised before (``section_place.py``, given ``old_text``),
   then under the right sub-heading (``section_subplace.py``).
2. Keep a short profile flat, and give one past
   ``HINBOX_SECTION_REORGANISE_CHARS`` its kind's starter headings once
   (``section_reorganise.py``), for the tables ``HINBOX_SECTION_REORGANISE``
   names.
3. Move an over-long opening's facts into their sections
   (``opening_sort.py``).
4. Split over-long sections into sub-sections when ``HINBOX_SECTION_SPLIT``
   is on for the table (``section_split.py``).
"""

from typing import Dict, Optional

from src.utils.tracing import annotate


def run_section_steps(
    profile: Dict,
    entity_type: str,
    entity_name: str,
    model: str,
    local: bool,
    kind: Optional[str] = None,
    domain: str = "guantanamo",
    old_text: Optional[str] = None,
) -> Dict:
    from src.engine.opening_sort import sort_opening
    from src.engine.section_reorganise import reorganise_enabled, reorganise_profile
    from src.engine.section_split import split_enabled, split_profile

    if not profile:
        return profile
    if reorganise_enabled(entity_type, local):
        if old_text is not None:
            from src.engine.section_place import place_profile

            profile, report = place_profile(
                profile, old_text, entity_type, entity_name, kind, domain,
                model, local,
            )  # fmt: skip
            annotate(section_place=report and report["status"])
            from src.engine.section_subplace import subplace_profile

            profile, report = subplace_profile(
                profile, old_text, entity_type, entity_name,
                model, local,
            )  # fmt: skip
            annotate(
                section_subplace=report and report["status"],
                section_subplace_moved=report and sum(report.get("moved", {}).values()),
                section_subplace_refused=report and report.get("refused"),
            )
        profile, report = reorganise_profile(
            profile, entity_type, entity_name, kind, domain,
            model, local,
        )  # fmt: skip
        annotate(section_reorganise=report and report["status"])
        profile, report = sort_opening(
            profile, entity_type, entity_name, kind, domain,
            model, local,
        )  # fmt: skip
        annotate(opening_sort=report and report["status"])
    if not split_enabled(entity_type, local):
        return profile
    split, reports = split_profile(profile, entity_type, entity_name, model, local)
    annotate(section_splits=[f"{r['section']}: {r['status']}" for r in reports] or None)
    return split
