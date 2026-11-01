"""Names that are never places: nationality words, news outlets, laws, dates.

Small local models file these as locations ("Yemeni", "Miami Herald",
"NDAA", "March 2008"). ``non_place_reason()`` says which kind a name is.
Names are compared after ``normalize_for_match()``. No LLM calls.
"""

import re
from typing import FrozenSet

from src.utils.name_variants import normalize_for_match

# Nationality and group words ("Yemeni", "Algerian"). Matched as the whole
# name, also with a plural "s", so "Indian Ocean" or "Dominican Republic" stay.
_NATIONALITIES: FrozenSet[str] = frozenset(
    """
    afghan albanian algerian american arab argentine argentinian australian
    austrian azerbaijani bahraini bangladeshi belgian bolivian bosnian brazilian
    british bulgarian burmese canadian chadian chechen chilean chinese colombian
    congolese croatian cuban czech danish dominican dutch ecuadorian egyptian
    emirati english eritrean estonian ethiopian filipino finnish french georgian
    german ghanaian greek guatemalan haitian honduran hungarian icelandic indian
    indonesian iranian iraqi irish israeli italian jamaican japanese jordanian
    kazakh kenyan korean kurd kurdish kuwaiti kyrgyz latvian lebanese libyan
    lithuanian malaysian maldivian malian mauritanian mexican mongolian moroccan
    nepalese nicaraguan nigerian norwegian omani pakistani palestinian panamanian
    paraguayan pashtun persian peruvian polish portuguese qatari rohingya
    rohingyan romanian russian salvadoran saudi scottish senegalese serbian
    slovak slovenian somali somalian spanish sudanese swedish swiss syrian
    taiwanese tajik tanzanian thai tunisian turkish turkmen ugandan uighur
    ukrainian uruguayan uyghur uzbek venezuelan vietnamese welsh yemeni
    """.split()
)

# News outlets. A name ending in one of these words ("Miami Herald",
# "New York Times") or equal to a listed outlet.
_MEDIA_LAST_WORDS: FrozenSet[str] = frozenset(
    {"times", "herald", "tribune", "gazette", "chronicle", "intercept"}
    | {"newspaper", "magazine", "journal"}
)
_MEDIA_NAMES: FrozenSet[str] = frozenset(
    {"washington post", "new york post", "jerusalem post", "huffington post"}
    | {"associated press", "voice of america", "reuters", "cnn", "bbc", "npr"}
    | {"al jazeera", "fox news", "nbc news", "cbs news", "abc news", "politico"}
)

# Laws and legal instruments ("NDAA", "Authorization for the Use of Military
# Force", "Patriot Act", "Geneva Conventions").
_LAW_WORDS: FrozenSet[str] = frozenset(
    {"act", "authorization", "amendment", "statute", "treaty", "conventions"}
    | {"ndaa", "aumf"}
)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# "March 2008", "Sept 11 2001", "2008"
_DATE_RE = re.compile(rf"^(?:(?:{_MONTHS})(?: \d{{1,2}})? )?\d{{4}}$")


def non_place_reason(name: str) -> str:
    """Which kind of non-place ``name`` is: nationality, media, law, date or ""."""
    match = normalize_for_match(name)
    if match.startswith("the "):
        match = match[4:]
    words = match.split()
    if not words:
        return ""
    if match in _NATIONALITIES or (
        len(words) == 1 and match.endswith("s") and match[:-1] in _NATIONALITIES
    ):
        return "nationality"
    if match in _MEDIA_NAMES or words[-1] in _MEDIA_LAST_WORDS:
        return "media"
    if _LAW_WORDS & set(words):
        return "law"
    if _DATE_RE.match(match):
        return "date"
    return ""
