"""Fetch Carol Rosenberg's Guantánamo articles from the NYT Article Search API.

Usage:
    export NYT_API_KEY="your-api-key-here"
    uv run python scripts/get_nyt_articles.py

    # Dry run (metadata only, no content scraping):
    uv run python scripts/get_nyt_articles.py --dry-run

    # Full run with content scraping:
    uv run python scripts/get_nyt_articles.py

Get a free API key at: https://developer.nytimes.com/
"""

import argparse
import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
NYT_API_BASE = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
RATE_LIMIT_DELAY = 6.5  # seconds between API requests (10 req/min limit)
PAGE_TIMEOUT = 15000  # 15 seconds for Playwright
ARTICLE_SCRAPE_DELAY = 5.0  # seconds between content scrapes
SAVE_FREQUENCY = 20  # Save progress every N articles

# Paths
PARQUET_PATH = Path("data/guantanamo/raw_sources/nyt_articles.parquet")


class NYTArticleSearchAPI:
    """Client for the NYT Article Search API v2."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._last_request_time: float = 0

    def _wait_for_rate_limit(self) -> None:
        now = datetime.now(UTC).timestamp()
        elapsed = now - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = datetime.now(UTC).timestamp()

    def search(
        self,
        query: str,
        fq: Optional[str] = None,
        begin_date: Optional[str] = None,
        end_date: Optional[str] = None,
        page: int = 0,
        sort: str = "newest",
    ) -> Dict[str, Any]:
        """Execute a single search request.

        Args:
            query: Search query string
            fq: Filter query (Lucene syntax)
            begin_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            page: Page number (0-based, max 100)
            sort: Sort order ("newest" or "oldest")

        Returns:
            API response dict with 'response' containing 'docs' and 'meta'
        """
        self._wait_for_rate_limit()

        params: Dict[str, Any] = {
            "q": query,
            "api-key": self.api_key,
            "page": page,
            "sort": sort,
        }
        if fq:
            params["fq"] = fq
        if begin_date:
            params["begin_date"] = begin_date
        if end_date:
            params["end_date"] = end_date

        response = requests.get(NYT_API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_all_articles(
        self,
        query: str = "guantanamo",
        author: str = "Carol Rosenberg",
        begin_date: Optional[str] = None,
        known_urls: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all articles matching query + author filter.

        Paginates through all results (up to 1000). If known_urls is
        provided, stops early when hitting consecutive known articles.

        Args:
            query: Search terms
            author: Author name for byline filter
            begin_date: Optional start date (YYYYMMDD)
            known_urls: Set of URLs already in our dataset

        Returns:
            List of article dicts with normalized fields
        """
        fq = f'byline:("{author}")'
        known_urls = known_urls or set()
        OVERLAP_THRESHOLD = 5

        articles: List[Dict[str, Any]] = []
        page = 0
        total_hits = None

        while page <= 100:  # API max is 100 pages
            try:
                result = self.search(
                    query=query,
                    fq=fq,
                    begin_date=begin_date,
                    page=page,
                    sort="newest",
                )
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning("Rate limited — waiting 60s before retry")
                    sleep(60)
                    continue
                raise

            meta = result.get("response", {}).get("meta", {})
            docs = result.get("response", {}).get("docs", [])

            if total_hits is None:
                total_hits = meta.get("hits", 0)
                logger.info(f"API reports {total_hits} total hits")

            if not docs:
                break

            consecutive_known = 0
            for doc in docs:
                url = doc.get("web_url", "")
                if url in known_urls:
                    consecutive_known += 1
                    if consecutive_known >= OVERLAP_THRESHOLD:
                        logger.info(
                            f"Hit {OVERLAP_THRESHOLD} consecutive known URLs — stopping"
                        )
                        return articles
                else:
                    consecutive_known = 0
                    articles.append(self._normalize_doc(doc))

            page += 1
            logger.info(
                f"Page {page}: fetched {len(docs)} docs, {len(articles)} new so far"
            )

            # Stop if we've fetched everything
            if page * 10 >= total_hits:
                break

        logger.info(f"Fetched {len(articles)} new articles total")
        return articles

    def _normalize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an API doc to our standard article format."""
        # Parse pub_date to Unix timestamp to match Miami Herald format
        pub_date_str = doc.get("pub_date", "")
        try:
            pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            published_date = pub_dt.timestamp()
        except (ValueError, AttributeError):
            published_date = None

        # Extract byline
        byline = doc.get("byline", {})
        byline_str = (
            byline.get("original", "") if isinstance(byline, dict) else str(byline)
        )

        return {
            "title": doc.get("headline", {}).get("main", ""),
            "url": doc.get("web_url", ""),
            "published_date": published_date,
            "scrape_timestamp": datetime.now(UTC).isoformat(),
            "content": doc.get("lead_paragraph", None),  # Only snippet for now
            "content_scrape_timestamp": None,
            "id": str(uuid.uuid4()),
            "processing_metadata": json.dumps(
                {
                    "source": "nyt",
                    "snippet": doc.get("snippet", ""),
                    "byline": byline_str,
                    "section": doc.get("section_name", ""),
                    "news_desk": doc.get("news_desk", ""),
                    "word_count": doc.get("word_count", 0),
                    "keywords": [kw.get("value", "") for kw in doc.get("keywords", [])],
                    "document_type": doc.get("document_type", ""),
                    "type_of_material": doc.get("type_of_material", ""),
                }
            ),
        }


# -- Content scraping (reused pattern from Miami Herald script) --


async def get_article_text(url: str) -> Optional[Tuple[str, str]]:
    """Scrape article title and content from a URL using Playwright."""
    try:
        async with async_playwright() as p:
            browser = await p.firefox.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            try:
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; "
                        "rv:123.0) Gecko/20100101 Firefox/123.0"
                    ),
                    locale="en-US",
                    timezone_id="America/New_York",
                    bypass_csp=True,
                )
                page = await context.new_page()
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type
                        in ["image", "stylesheet", "font"]
                        else route.continue_()
                    ),
                )

                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT
                )

                if response and response.ok:
                    content = await page.content()
                    soup = BeautifulSoup(content, "html.parser")

                    # NYT article structure
                    article = (
                        soup.find("article")
                        or soup.find("section", attrs={"name": "articleBody"})
                        or soup.find("main")
                    )

                    if article:
                        title = soup.find("h1")
                        title_text = title.text.strip() if title else ""

                        paragraphs = article.find_all("p")
                        if paragraphs:
                            content_text = "\n\n".join(
                                p.text.strip()
                                for p in paragraphs
                                if p.text.strip()
                                and not p.text.strip().startswith("Advertisement")
                            )
                        else:
                            content_text = article.get_text(
                                separator="\n\n", strip=True
                            )

                        return title_text, content_text

            except Exception as e:
                logger.error(f"Error during scraping: {e}")
            finally:
                await browser.close()

    except Exception as e:
        logger.error(f"Error launching browser: {e}")

    return None


async def scrape_content_for_articles(
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Scrape full content for articles that only have snippets."""
    needs_scraping = [a for a in articles if not a.get("content_scrape_timestamp")]

    if not needs_scraping:
        logger.info("All articles already have scraped content")
        return articles

    logger.info(f"Scraping content for {len(needs_scraping)} articles")
    semaphore = asyncio.Semaphore(2)  # Conservative for NYT
    success = 0
    failed = 0

    async def scrape_one(article: Dict[str, Any]) -> None:
        nonlocal success, failed
        async with semaphore:
            try:
                result = await get_article_text(article["url"])
                if result:
                    _, content = result
                    article["content"] = content
                    article["content_scrape_timestamp"] = datetime.now(UTC).isoformat()
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error scraping {article['url']}: {e}")
                failed += 1
            await asyncio.sleep(ARTICLE_SCRAPE_DELAY)

    # Process in small batches
    batch_size = 3
    with tqdm(total=len(needs_scraping), desc="Scraping NYT articles") as pbar:
        for i in range(0, len(needs_scraping), batch_size):
            batch = needs_scraping[i : i + batch_size]
            await asyncio.gather(*(scrape_one(a) for a in batch))
            pbar.update(len(batch))

    logger.info(f"Scraping done: {success} success, {failed} failed")
    return articles


# -- Parquet I/O --


def load_existing_urls(parquet_path: Path) -> Set[str]:
    if not parquet_path.exists():
        return set()
    table = pq.read_table(parquet_path, columns=["url"])
    return set(table.column("url").to_pylist())


def save_to_parquet(parquet_path: Path, articles: List[Dict[str, Any]]) -> int:
    """Save articles to Parquet, appending to existing file if present."""
    if not articles:
        return 0

    new_table = pa.table(
        {
            "title": [a["title"] for a in articles],
            "url": [a["url"] for a in articles],
            "published_date": [a["published_date"] for a in articles],
            "scrape_timestamp": [a["scrape_timestamp"] for a in articles],
            "content": [a["content"] for a in articles],
            "content_scrape_timestamp": [
                a["content_scrape_timestamp"] for a in articles
            ],
            "id": [a["id"] for a in articles],
            "processing_metadata": [a["processing_metadata"] for a in articles],
        }
    )

    if parquet_path.exists():
        existing = pq.read_table(parquet_path)
        new_table = new_table.cast(existing.schema)
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp_path)
    tmp_path.replace(parquet_path)

    return len(articles)


# -- Main --


async def async_main(dry_run: bool = False) -> None:
    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        logger.error(
            "NYT_API_KEY environment variable not set.\n"
            "Get a free key at: https://developer.nytimes.com/"
        )
        return

    parquet_path = PARQUET_PATH
    api = NYTArticleSearchAPI(api_key)

    # Load existing URLs for incremental updates
    known_urls = load_existing_urls(parquet_path)
    if known_urls:
        logger.info(f"Found {len(known_urls)} existing NYT articles in Parquet")

    # Fetch article metadata from API
    new_articles = api.fetch_all_articles(
        query="guantanamo",
        author="Carol Rosenberg",
        known_urls=known_urls if known_urls else None,
    )

    if not new_articles:
        logger.info("No new articles found — everything is up to date")
        return

    # Show what we found
    logger.info(f"\nFound {len(new_articles)} new articles:")
    for i, a in enumerate(new_articles[:10]):
        pub_ts = a["published_date"]
        if pub_ts:
            pub_str = datetime.fromtimestamp(pub_ts, tz=UTC).strftime("%Y-%m-%d")
        else:
            pub_str = "unknown"
        logger.info(f"  {i + 1}. [{pub_str}] {a['title'][:80]}")
    if len(new_articles) > 10:
        logger.info(f"  ... and {len(new_articles) - 10} more")

    if dry_run:
        logger.info("\n--- DRY RUN: not saving or scraping ---")
        return

    # Scrape full content
    new_articles = await scrape_content_for_articles(new_articles)

    # Save to Parquet
    count = save_to_parquet(parquet_path, new_articles)
    logger.info(f"Saved {count} new articles to {parquet_path}")

    if parquet_path.exists():
        table = pq.read_table(parquet_path)
        logger.info(f"Parquet now contains {table.num_rows} total NYT articles")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Carol Rosenberg's NYT Guantánamo articles"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch metadata from API, don't scrape content or save",
    )
    args = parser.parse_args()

    asyncio.run(async_main(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
