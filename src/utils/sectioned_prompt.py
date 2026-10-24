"""Prompts with labelled ``[[ ## field ## ]]`` sections and a plain-text reply.

Each input goes under its own header, and the model answers with written
reasoning first and then each output field under its own header. This is the
layout DSPy's chat adapter uses. On the hard merge eval set it cut false merges
for local models, where the same instructions asked for a tool call did not.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

# (name, description) of a field; the description may be empty
Field = Tuple[str, str]

_FORMAT = """Your input fields are:
{inputs}
Your output fields are:
{outputs}

Every input and output field appears under its own `[[ ## field ## ]]` header, \
and the output ends with `[[ ## completed ## ]]`.

In adhering to this structure, your objective is:
{instructions}"""


def _describe(fields: Sequence[Field]) -> str:
    return "\n".join(
        f"{i}. `{name}`" + (f": {desc}" if desc else "")
        for i, (name, desc) in enumerate(fields, 1)
    )


def build_sectioned_messages(
    instructions: str,
    inputs: Sequence[Field],
    values: Dict[str, str],
    outputs: Sequence[Field],
    hints: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """System message describing the fields, user message holding the inputs.

    ``hints`` adds a short note after an output field in the closing request,
    e.g. ``{"is_match": "True or False"}``.
    """
    system = _FORMAT.format(
        inputs=_describe(inputs), outputs=_describe(outputs), instructions=instructions
    )
    body = "\n\n".join(f"[[ ## {name} ## ]]\n{values[name]}" for name, _ in inputs)
    hints = hints or {}
    names = [
        f"`[[ ## {name} ## ]]`" + (f" ({hints[name]})" if name in hints else "")
        for name, _ in outputs
    ]
    ask = (
        f"Respond with the corresponding output fields, starting with the field "
        f"{names[0]}, then {', then '.join(names[1:])}, and then ending with the "
        "marker for `[[ ## completed ## ]]`."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{body}\n\n{ask}"},
    ]


# Models sometimes drop the closing "##" ("[[ ## confidence ]]"), so it's optional
_HEADER = r"\[\[ ## (\w+)(?: ##)? \]\]"
_SECTION = re.compile(
    _HEADER + r"\s*(.*?)(?=" + _HEADER.replace("(\\w+)", "\\w+") + r"|$)", re.S
)


def parse_sections(text: str) -> Dict[str, str]:
    """Map each ``[[ ## name ## ]]`` header in a reply to the text under it."""
    return {name: value.strip() for name, value in _SECTION.findall(text)}


def read_bool(fields: Dict[str, str], name: str) -> bool:
    """First word of a field as a yes/no; ValueError if it isn't one."""
    word = (fields.get(name, "").split() or [""])[0].strip("`*\"' .").lower()
    if word not in ("true", "false", "yes", "no"):
        raise ValueError(f"no yes/no answer for {name!r}")
    return word in ("true", "yes")


def read_unit_float(fields: Dict[str, str], name: str, default: float) -> float:
    """A field as a number clamped to [0, 1], or ``default`` if unreadable."""
    try:
        return min(max(float(fields.get(name, "")), 0.0), 1.0)
    except ValueError:
        return default
