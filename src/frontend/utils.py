import hashlib
import re
from urllib.parse import quote, unquote


def build_citation_map(text, articles):
    """Build a mapping of article IDs to citation numbers based on text citations."""
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


def transform_profile_text(text, articles):
    """Replace footnote references with links to articles."""
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
    """Generate a stable pastel-ish RGB color from a label."""
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    seed = int(digest[:6], 16)
    r = 128 + (seed % 103)
    g = 128 + ((seed // 103) % 103)
    b = 128 + ((seed // (103 * 103)) % 103)
    return f"rgb({r},{g},{b})"


def encode_key(k: str) -> str:
    """Encode the entity key for a URL."""
    return quote(k, safe="")


def decode_key(k: str) -> str:
    """Decode the entity key from a URL."""
    return unquote(k)


def format_article_list(articles, profile_text=""):
    """Create a consistent formatting for article lists with citation numbers."""
    from fasthtml.common import A, Div, Li, Ul

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
