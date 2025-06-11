"""Frontend utility functions for text processing, URL handling, and UI formatting.

This module provides utility functions for processing entity profiles, handling citations,
generating stable colors, encoding/decoding URLs, and formatting article lists for display
in the FastHTML frontend.
"""

import hashlib
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import quote, unquote

from fasthtml.common import A, Div, Li, Ul


def build_citation_map(
    text: str, articles: List[Dict[str, Any]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build mappings for article citations found in profile text.

    Processes profile text to find citation markers (^[article_id]) and creates
    two mappings: article IDs to sequential citation numbers, and article IDs
    to their URLs for link generation.

    Args:
        text: Profile text containing citation markers in ^[id] format
        articles: List of article dictionaries with 'article_id' and 'article_url' fields

    Returns:
        Tuple of (marker_map, article_map) where:
        - marker_map: Maps article IDs to sequential citation numbers
        - article_map: Maps article IDs to article URLs

    Note:
        Citation numbers are assigned in order of first appearance in the text.
        Supports comma-separated multiple IDs in single citation markers.
    """
    article_map = {}
    for a in articles:
        aid = a.get("article_id")
        article_map[aid] = a.get("article_url", "#")

    pattern = r"\^\[([0-9a-fA-F-,\s]+)\]"
    marker_map = {}
    marker_counter = [1]

    # Find all citations in text to build the mapping
    for match in re.finditer(pattern, text):
        refs_str = match.group(1)
        refs = [r.strip() for r in refs_str.split(",")]

        for ref in refs:
            if ref not in marker_map:
                marker_map[ref] = str(marker_counter[0])
                marker_counter[0] += 1

    return marker_map, article_map


def transform_profile_text(text: str, articles: List[Dict[str, Any]]) -> str:
    """Transform profile text by replacing citation markers with clickable links.

    Processes profile text to convert citation markers (^[article_id]) into
    clickable superscript links that open articles in new tabs. Handles
    multiple comma-separated IDs in single citations.

    Args:
        text: Profile text containing citation markers in ^[id] format
        articles: List of article dictionaries for building citation mappings

    Returns:
        Transformed text with citations replaced by HTML superscript links

    Note:
        Links open in new tabs using target="_blank".
        Missing article IDs are replaced with "?" marker.
    """
    marker_map, article_map = build_citation_map(text, articles)
    pattern = r"\^\[([0-9a-fA-F-,\s]+)\]"

    def replacer(match):
        refs_str = match.group(1)
        # Handle multiple comma-separated references
        refs = [r.strip() for r in refs_str.split(",")]

        markers = []
        for ref in refs:
            marker = marker_map.get(ref, "?")
            url = article_map.get(ref, "#")
            markers.append(f'<a href="{url}" target="_blank">{marker}</a>')

        return f"<sup>{','.join(markers)}</sup>"

    return re.sub(pattern, replacer, text)


def random_pastel_color(label: str) -> str:
    """Generate a stable pastel RGB color from a text label.

    Uses MD5 hash of the label to generate consistent pastel colors for
    UI elements like filter chips. Colors are in the lighter RGB range
    (128-230) for better readability on various backgrounds.

    Args:
        label: Text label to generate color for

    Returns:
        RGB color string in format "rgb(r,g,b)"

    Note:
        Same label always produces the same color for UI consistency.
        Colors are designed to be readable on both light and dark backgrounds.
    """
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    seed = int(digest[:6], 16)
    r = 128 + (seed % 103)
    g = 128 + ((seed // 103) % 103)
    b = 128 + ((seed // (103 * 103)) % 103)
    return f"rgb({r},{g},{b})"


def encode_key(k: str) -> str:
    """URL-encode an entity key for safe use in URLs.

    Encodes entity keys for use in URL paths, handling special characters
    and spaces that might be present in entity names or composite keys.

    Args:
        k: Entity key to encode

    Returns:
        URL-encoded key safe for use in URL paths
    """
    return quote(k, safe="")


def decode_key(k: str) -> str:
    """URL-decode an entity key from a URL parameter.

    Decodes entity keys extracted from URL paths back to their original
    form for database lookups and display.

    Args:
        k: URL-encoded entity key

    Returns:
        Decoded entity key in original format
    """
    return unquote(k)


def format_article_list(
    articles: List[Dict[str, Any]], profile_text: str = ""
) -> Union[Div, Ul]:
    """Create a formatted list of articles with optional citation numbers.

    Generates a styled article list showing titles and "View Source" links.
    If profile text is provided, includes citation numbers based on the
    text's citation markers.

    Args:
        articles: List of article dictionaries with title, URL, and ID fields
        profile_text: Optional profile text for citation number mapping

    Returns:
        Ul component with formatted article list, or Div with empty state message

    Note:
        Citation numbers are only shown if profile_text contains citations.
        Each article includes a "View Source" link that opens in a new tab.
    """
    if not articles:
        return Div("No articles associated with this entity.", cls="empty-state")

    # Build citation mapping if profile text is provided
    marker_map = {}
    if profile_text:
        marker_map, _ = build_citation_map(profile_text, articles)

    art_list = []
    for art in articles:
        aid = art.get("article_id")
        title = art.get("article_title", "Untitled")
        url = art.get("article_url", "#")

        # Add citation number if we have a mapping
        if marker_map and aid in marker_map:
            citation_num = marker_map[aid]
            title_with_num = f"{citation_num}. {title}"
        else:
            title_with_num = title

        art_list.append(
            Li(
                title_with_num + " ",
                A("(View Source)", href=url, target="_blank"),
                style="display:flex; justify-content:space-between; align-items:center;",
            )
        )

    return Ul(*art_list, cls="article-list")
