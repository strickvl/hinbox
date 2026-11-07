"""Split a letters-to-the-editor page into its separate letters.

A Miami Herald letters page is one "article" holding a dozen short letters
on unrelated subjects, each signed on its own line ("Ray Turner, Kendall").
Checked as a whole, a page with one Guantánamo letter passes the relevance
check, and extraction then files every signer and everyone named in the
other letters (a hurricane chief, a teacher who drowned raccoons) as
entities. ``split_letters()`` finds the signature lines so each letter can
be checked on its own and only the relevant ones extracted.

It is deterministic and works on the text alone, so it behaves the same for
local and cloud models.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# A page needs this many signed letters before it is treated as a letters
# page; two "Name, Place" lines also turn up as bylines in news stories
MIN_LETTERS = 3
# A letter's own text (signature aside) must be at least this long, so a
# list of names with places ("Sara Leviten, North Miami" x 3) isn't one
MIN_BODY_CHARS = 40
# Text after the last signature shorter than this ("This story was
# originally published ...") is dropped rather than checked as a letter
MIN_TRAILING_CHARS = 200
MAX_SIGNATURE_CHARS = 80

_PARTICLES = {"de", "del", "della", "da", "di", "dos", "du", "la", "le"}
_PARTICLES |= {"van", "von", "der", "den", "al", "el", "bin", "ibn", "y"}
_SUFFIXES = {"Jr.", "Sr.", "II", "III", "IV"}
_TITLES = {"Dr.", "Rev.", "Rabbi", "Father", "Sister"}
_INITIAL_RE = re.compile(r"^[A-Z]\.$")
_WORD_RE = re.compile(r"^[^\W\d_][^\W\d_'’.-]*(?:['’-][^\W\d_]+)*$")
# Separator lines between letters: blanks, bullets, dashes, asterisks
_SEPARATOR_RE = re.compile(r"^[\s▪•·*_—–-]*$")
_PLACE_RE = re.compile(r"^[^\W\d_][\w .,'’-]*$")


@dataclass(frozen=True)
class Letter:
    body: str  # the letter's paragraphs
    signature: str  # its signature lines, "" for unsigned text
    signer: str  # the name from the signature, "" for unsigned text

    @property
    def text(self) -> str:
        return f"{self.body}\n\n{self.signature}".strip()


def _is_name_word(word: str) -> bool:
    return bool(_WORD_RE.match(word)) and word[0].isupper()


def _is_person_name(name: str) -> bool:
    """2-5 capitalised words, initials and particles ("Steven P. Willis")."""
    words = name.split()
    if words and words[0] in _TITLES:
        words = words[1:]
    if words and words[-1] in _SUFFIXES:
        words = words[:-1]
    if not 2 <= len(words) <= 5:
        return False
    if not (_is_name_word(words[0]) and _is_name_word(words[-1])):
        return False
    return all(
        _is_name_word(w) or _INITIAL_RE.match(w) or w in _PARTICLES for w in words[1:-1]
    )


def signature_name(line: str) -> Optional[str]:
    """The signer's name if ``line`` opens a letter signature, else None.

    A signature is a short line of a person's name, a comma, then a place,
    or nothing when the place follows on the next lines ("Phyllis
    Rodriguez,"). Sentences, bylines with "and" and lines ending in . ! ?
    don't count.
    """
    line = line.strip()
    if len(line) > MAX_SIGNATURE_CHARS or "," not in line or " and " in line:
        return None
    name, rest = (part.strip() for part in line.split(",", 1))
    if not _is_person_name(name):
        return None
    if rest and not (_PLACE_RE.match(rest) and rest[0].isupper()):
        return None
    if rest and len(rest.split()) > 8:
        return None
    return name


def _continues_signature(line: str) -> bool:
    """A follow-on signature line (group or place) after a line ending ','."""
    line = line.strip()
    return (
        0 < len(line) <= MAX_SIGNATURE_CHARS
        and line[0].isupper()
        and not re.search(r"[.!?:;\"“”]$", line)
        and len(line.split()) <= 8
    )


def _chunks(lines: List[str]) -> Tuple[List[Letter], List[str]]:
    """Signed letters in order, plus the lines left after the last one."""
    letters: List[Letter] = []
    body: List[str] = []
    i = 0
    while i < len(lines):
        name = signature_name(lines[i])
        if name is None:
            body.append(lines[i])
            i += 1
            continue
        signature = [lines[i]]
        i += 1
        while signature[-1].rstrip().endswith(","):
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _continues_signature(lines[j]):
                signature.append(lines[j])
                i = j + 1
            else:
                break
        sig_text = "\n".join(line.strip() for line in signature)
        letters.append(Letter("\n\n".join(_paragraphs(body)), sig_text, name))
        body = []
    return letters, body


def _paragraphs(lines: List[str]) -> List[str]:
    return [line.strip() for line in lines if not _SEPARATOR_RE.match(line)]


def split_letters(text: str) -> List[Letter]:
    """The page's letters, or [] when ``text`` isn't a letters page.

    Text before the first signature belongs to the first letter (a page
    intro, if any, goes with it). Text after the last signature becomes an
    unsigned letter if it is long enough to be one.
    """
    letters, rest = _chunks((text or "").split("\n"))
    if len(letters) < MIN_LETTERS:
        return []
    if min(len(letter.body) for letter in letters) < MIN_BODY_CHARS:
        return []
    trailing = "\n\n".join(_paragraphs(rest))
    if len(trailing) >= MIN_TRAILING_CHARS:
        letters.append(Letter(trailing, "", ""))
    return letters


def join_letters(letters: List[Letter]) -> str:
    return "\n\n".join(letter.text for letter in letters)
