"""Small fixes to a written page's order and opening, all in code.

``keep_identity()``: the opening's first sentence says what the entity is
("Karachi is a major city in Pakistan."). On 21 Nov the model check judged
it against the wrong note and dropped it, and the Karachi page then opened
with arrests. When a note says it, word for word or nearly, it stays.

``date_sort()``: sentences the fill stage adds go at the end of their
section, so on 21 Nov Zaki's long section ended in a tail out of date order
(June 2017 after February 2018) in three draws of three. Each filled
sentence is moved before the first sentence of its section that is dated
later than it.
"""

from typing import List, Optional, Sequence, Tuple

from src.engine.written_page import Note, Written


def keep_identity(
    written: List[Written], notes: Sequence[Note], name: str, identity: str = ""
) -> bool:
    """Restore a dropped first sentence that a note gives with all its
    details and most of its words, or, when it opens with the name, whose
    every detail is in the plan's identity or the notes; True when it did.
    Round 21: "Kandahar is a location in Afghanistan." was dropped as "not
    in its notes: Afghanistan", and three more pages lost theirs the same
    way; the plan's identity, which said it, came from those notes."""
    from src.engine.written_checks import _content, missing_details

    first = next((w for w in written if not w.section and not w.summary), None)
    if first is None or not first.dropped or first.dropped.startswith("repeats"):
        return False
    words = _content(first.text)
    for n in notes:
        if missing_details(first.text, [n.text], name):
            continue
        if words and len(words & _content(n.text)) >= 0.8 * len(words):
            first.notes, first.dropped, first.check = [n.n], "", "identity"
            return True
    opens = first.text.lower().startswith(name.lower()) if name else False
    if (
        identity
        and opens
        and len(first.text.split()) <= 30
        and not missing_details(first.text, [identity] + [n.text for n in notes], name)
    ):
        first.dropped, first.check = "", "identity"
        return True
    return False


def _when(w: Written, by_n: dict) -> Optional[Tuple[int, int, int]]:
    """(year, month, day) the sentence names first, else None. Round 19
    read "Sept. 11, 2002" as 2002 with no month here and moved Karachi's
    bin Attash dispute before its February 2002 arrest."""
    from src.engine.written_groups import text_when

    return text_when(w.text)


def date_sort(written: List[Written], notes: Sequence[Note]) -> int:
    """Move each kept filled sentence that names a date before the first
    kept, dated sentence of its section dated later; the number moved. A
    sentence with only a year sorts before that year's months."""
    by_n = {n.n: n for n in notes}
    moved = 0
    for w in [w for w in written if w.filled and not w.dropped and w.section]:
        when = _when(w, by_n)
        if when is None:
            continue
        mine = [
            o
            for o in written
            if o.section == w.section and not o.dropped and o is not w
        ]
        later = next(
            (o for o in mine if not o.filled and (_when(o, by_n) or (0, 0, 0)) > when),
            None,
        )
        if later is None:
            continue
        written.pop(next(i for i, o in enumerate(written) if o is w))
        written.insert(next(i for i, o in enumerate(written) if o is later), w)
        w.para = later.para
        moved += 1
    return moved


def ensure_identity(
    written: List[Written], notes: Sequence[Note], name: str, identity: str,
    opening: Sequence[int],
) -> bool:  # fmt: skip
    """Put "{name} is {identity}." first when no kept opening sentence names
    the subject, if the code check passes it against the plan's opening
    notes; True when it did. Round 19's Rishikof page lost its first
    sentence and opened "His career and appointment ... are described"."""
    surname = name.split()[-1].lower() if name.split() else ""
    opening_kept = [w for w in written if not w.section and not w.dropped]
    if (
        not surname
        or not identity.strip()
        or any(surname in w.text.lower() for w in opening_kept)
    ):
        return False
    known = {n.n for n in notes}
    cited = [n for n in opening if n in known][:3] or [n.n for n in notes[:1]]
    what = identity.strip().rstrip(".")
    if what.lower().startswith(name.lower()):
        what = what[len(name) :].strip().removeprefix("is ").strip()
    first_word, _, rest = what.partition(" ")
    if first_word.lower() in ("a", "an", "the"):  # "is A detention facility"
        what = f"{first_word.lower()} {rest}"
    # Taken as the plan wrote it: round 21's checks turned down "Kandahar is
    # a location in Afghanistan." and cut Clinton's down to "a politician and
    # former U.S. Secretary of."
    written.insert(0, Written(f"{name} is {what}.", list(cited), check="identity"))
    return True
