import hashlib
import re
from urllib.parse import quote, unquote


def transform_profile_text(text, articles):
    """Replace footnote references with links to articles."""
    article_map = {}
    for a in articles:
        aid = a.get("article_id")
        article_map[aid] = a.get("article_url", "#")

    pattern = r"\^\[([0-9a-fA-F-,\s]+)\]"

    marker_map = {}
    marker_counter = [1]  # Using a list so it can be updated in replacer

    def replacer(match):
        refs_str = match.group(1)
        # Handle multiple comma-separated references
        refs = [r.strip() for r in refs_str.split(",")]

        markers = []
        for ref in refs:
            if ref not in marker_map:
                marker_map[ref] = str(marker_counter[0])
                marker_counter[0] += 1
            marker = marker_map[ref]
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


def format_article_list(articles):
    """Create a consistent formatting for article lists."""
    from fasthtml.common import A, Div, Li, Ul

    if not articles:
        return Div("No articles associated with this entity.", cls="empty-state")

    art_list = []
    for art in articles:
        art_list.append(
            Li(
                f"{art.get('article_title', 'Untitled')} ",
                A("(View Source)", href=art.get("article_url", "#"), target="_blank"),
                style="display:flex; justify-content:space-between; align-items:center;",
            )
        )

    return Ul(*art_list, cls="article-list")
