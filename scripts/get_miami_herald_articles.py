import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass
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
PAGE_TIMEOUT = 15000  # 15 seconds
ARTICLE_SCRAPE_DELAY = 5.0  # seconds between article scrapes
SAVE_FREQUENCY = 20  # Save progress every N articles
TEST_MODE = False  # Set to True to process only a small subset of articles
TEST_SAMPLE_SIZE = 10  # Number of articles to process in test mode

# Path to the canonical Parquet file
PARQUET_PATH = Path("data/guantanamo/raw_sources/miami_herald_articles.parquet")


@dataclass
class SearchConfig:
    query: str
    start: int = 0
    rows: int = 25  # Reduced batch size to be more conservative
    sort: str = "newest"


@dataclass
class Article:
    title: str
    url: str
    published_date: str
    scrape_timestamp: str
    content: Optional[str] = None
    content_scrape_timestamp: Optional[str] = None

    @classmethod
    def from_api_response(cls, article_data: Dict[str, Any]) -> "Article":
        return cls(
            title=article_data.get("title", ""),
            url=article_data.get("url", ""),
            published_date=article_data.get("published_date", ""),
            scrape_timestamp=datetime.now(UTC).isoformat(),
            content=article_data.get("content"),
            content_scrape_timestamp=article_data.get("content_scrape_timestamp"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Article":
        return cls(**data)


def load_existing_urls(parquet_path: Path) -> Set[str]:
    """Load the set of URLs already present in the Parquet file."""
    if not parquet_path.exists():
        return set()
    table = pq.read_table(parquet_path, columns=["url"])
    return set(table.column("url").to_pylist())


def load_existing_table(parquet_path: Path) -> Optional[pa.Table]:
    """Load the existing Parquet table, or None if it doesn't exist."""
    if not parquet_path.exists():
        return None
    return pq.read_table(parquet_path)


def append_to_parquet(parquet_path: Path, new_articles: List[Dict[str, Any]]) -> int:
    """Append new articles to the Parquet file, preserving all existing data.

    Returns the number of new rows added.
    """
    if not new_articles:
        return 0

    # Build a table from the new articles, matching the existing schema
    existing_table = load_existing_table(parquet_path)

    # Generate unique IDs for new articles
    for article in new_articles:
        if "id" not in article or article["id"] is None:
            article["id"] = str(uuid.uuid4())
        if "processing_metadata" not in article:
            article["processing_metadata"] = None

    new_table = pa.table(
        {
            "title": [a["title"] for a in new_articles],
            "url": [a["url"] for a in new_articles],
            "published_date": [a["published_date"] for a in new_articles],
            "scrape_timestamp": [a["scrape_timestamp"] for a in new_articles],
            "content": [a["content"] for a in new_articles],
            "content_scrape_timestamp": [
                a["content_scrape_timestamp"] for a in new_articles
            ],
            "id": [a["id"] for a in new_articles],
            "processing_metadata": [a["processing_metadata"] for a in new_articles],
        }
    )

    if existing_table is not None:
        # Cast new_table columns to match existing schema types
        new_table = new_table.cast(existing_table.schema)
        combined = pa.concat_tables([existing_table, new_table])
    else:
        combined = new_table

    # Write atomically via temp file
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(combined, tmp_path)
    tmp_path.replace(parquet_path)

    return len(new_articles)


class MiamiHeraldAPI:
    BASE_URL = "https://publicapi.misitemgr.com/webapi-public/v2/publications/miamiherald/search"
    RATE_LIMIT_DELAY = 2.0  # seconds between requests

    def __init__(self) -> None:
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "text/plain;charset=UTF-8",
            "dnt": "1",
            "origin": "https://www.miamiherald.com",
            "referer": "https://www.miamiherald.com/",
            "sec-ch-ua": '"Chromium";v="133", "Not(A:Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "cross-site",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "x-forwarded-host": "www.miamiherald.com",
        }
        self._last_request_time: float = 0

    def _wait_for_rate_limit(self) -> None:
        """Ensure we wait an appropriate amount of time between requests."""
        now = datetime.now(UTC).timestamp()
        time_since_last_request = now - self._last_request_time
        if time_since_last_request < self.RATE_LIMIT_DELAY:
            sleep(self.RATE_LIMIT_DELAY - time_since_last_request)
        self._last_request_time = datetime.now(UTC).timestamp()

    def search_articles(self, config: SearchConfig) -> Dict[str, Any]:
        """Search for articles using the provided configuration."""
        self._wait_for_rate_limit()

        payload = {
            "q": config.query,
            "start": config.start,
            "rows": config.rows,
            "sort": config.sort,
        }

        try:
            response = requests.post(
                self.BASE_URL, headers=self.headers, data=json.dumps(payload)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            raise

    def get_new_articles(self, query: str, known_urls: Set[str]) -> List[Article]:
        """Fetch articles from the API, stopping when we hit a known URL.

        Since the API returns newest-first, we paginate until we encounter
        an article URL that's already in our dataset. We also stop if we
        see OVERLAP_THRESHOLD consecutive known URLs in a single batch
        (handles edge cases where articles might have been missed).

        Returns a list of new Article objects (newest first).
        """
        OVERLAP_THRESHOLD = 5  # stop after this many consecutive known URLs

        start = 0
        total_available = None
        new_articles: List[Article] = []
        stop = False

        while not stop:
            config = SearchConfig(query=query, start=start)
            results = self.search_articles(config)

            if total_available is None:
                total_available = results.get("total", 0)
                logger.info(
                    f"API reports {total_available} total articles for query '{query}'"
                )

            if not results.get("results"):
                break

            consecutive_known = 0
            for article_data in results["results"]:
                url = article_data.get("url", "")
                if url in known_urls:
                    consecutive_known += 1
                    if consecutive_known >= OVERLAP_THRESHOLD:
                        logger.info(
                            f"Hit {OVERLAP_THRESHOLD} consecutive known URLs — stopping fetch"
                        )
                        stop = True
                        break
                else:
                    consecutive_known = 0
                    new_articles.append(Article.from_api_response(article_data))

            if not stop:
                start += config.rows
                # Safety: don't paginate past what the API says exists
                if start >= total_available:
                    break

        logger.info(f"Found {len(new_articles)} new articles from API")
        return new_articles

    def get_all_articles(self, query: str) -> List[Article]:
        """Get all articles for a query (full fetch, no dedup). Used for
        initial population when no Parquet exists yet."""
        start = 0
        total_fetched = 0
        total_available = None
        all_articles: List[Article] = []

        while True:
            config = SearchConfig(query=query, start=start)
            results = self.search_articles(config)

            if total_available is None:
                total_available = results.get("total", 0)
                print(
                    f"\nFetching {total_available} articles "
                    f"(with {config.rows} per batch)..."
                )
                progress_bar = tqdm(total=total_available, desc="Articles fetched")

            if not results.get("results"):
                break

            batch_count = 0
            for article_data in results["results"]:
                all_articles.append(Article.from_api_response(article_data))
                total_fetched += 1
                batch_count += 1

            progress_bar.update(batch_count)

            if total_fetched >= total_available:
                progress_bar.close()
                break

            start += config.rows

        return all_articles


async def get_article_text(url: str) -> Optional[Tuple[str, str]]:
    """Get article title and content from URL.

    Returns:
        Optional tuple of (title, content) if successful, None if failed
    """
    try:
        async with async_playwright() as p:
            # Launch Firefox with minimal configuration
            browser = await p.firefox.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )

            try:
                # Create a new context with basic configuration
                context = await browser.new_context(
                    viewport={"width": 1920, "height": 1080},
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0",
                    locale="en-US",
                    timezone_id="America/New_York",
                    bypass_csp=True,
                )

                # Create page and block unnecessary resources
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

                # Navigate to URL
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT
                )

                if response and response.ok:
                    # Get page content
                    content = await page.content()

                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")

                    # Find article container
                    article = (
                        soup.find("article")
                        or soup.find("div", class_="article-body")
                        or soup.find("main")
                        or soup.find("div", class_="container")
                    )

                    if article:
                        # Get title
                        title = soup.find("h1") or soup.find("header")
                        title_text = title.text.strip() if title else ""

                        # Get content
                        paragraphs = article.find_all(["p", "div.paragraph"])
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


@dataclass
class ProcessingStats:
    """Statistics for article processing."""

    total_articles: int = 0
    processed_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    processing_times: List[float] = None
    start_time: float = 0.0
    last_save_count: int = 0

    def __post_init__(self):
        self.processing_times = []
        self.start_time = datetime.now(UTC).timestamp()

    def should_save_progress(self) -> bool:
        return (self.processed_count - self.last_save_count) >= SAVE_FREQUENCY

    def update_last_save(self) -> None:
        self.last_save_count = self.processed_count

    def add_processing_time(self, time_taken: float) -> None:
        self.processing_times.append(time_taken)
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)

    def get_avg_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_estimated_remaining_time(self) -> float:
        if not self.processing_times:
            return 0.0
        remaining_articles = self.total_articles - self.processed_count
        return remaining_articles * self.get_avg_processing_time()

    def format_time(self, seconds: float) -> str:
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def get_progress_string(self) -> str:
        elapsed = datetime.now(UTC).timestamp() - self.start_time
        remaining = self.get_estimated_remaining_time()

        return (
            f"Progress: {self.processed_count}/{self.total_articles} "
            f"({(self.processed_count / self.total_articles * 100):.1f}%) | "
            f"Success: {self.success_count} | "
            f"Failed: {self.failure_count} | "
            f"Elapsed: {self.format_time(elapsed)} | "
            f"Remaining: {self.format_time(remaining)}"
        )


async def process_article_batch(
    articles: List[Article],
    semaphore: asyncio.Semaphore,
    stats: ProcessingStats,
) -> List[Article]:
    """Process a batch of articles to fetch their content."""

    async def process_single_article(article: Article) -> Article:
        async with semaphore:
            if not article.content:
                start_time = datetime.now(UTC).timestamp()
                try:
                    result = await get_article_text(article.url)
                    if result:
                        _, content = result
                        article.content = content
                        article.content_scrape_timestamp = datetime.now(UTC).isoformat()
                        stats.success_count += 1
                    else:
                        stats.failure_count += 1

                    end_time = datetime.now(UTC).timestamp()
                    stats.add_processing_time(end_time - start_time)

                except Exception as e:
                    logger.error(f"Error processing article {article.url}: {e}")
                    stats.failure_count += 1

                stats.processed_count += 1
                await asyncio.sleep(ARTICLE_SCRAPE_DELAY)
        return article

    return await asyncio.gather(
        *(process_single_article(article) for article in articles)
    )


async def scrape_content_for_articles(
    articles: List[Article],
) -> List[Article]:
    """Scrape full content for a list of articles using Playwright.

    Returns the articles with content populated where successful.
    """
    if not articles:
        return articles

    # Filter to only articles needing content
    needs_content = [a for a in articles if not a.content]
    if not needs_content:
        logger.info("All articles already have content")
        return articles

    if TEST_MODE:
        needs_content = needs_content[:TEST_SAMPLE_SIZE]
        logger.info(f"TEST MODE: Processing {len(needs_content)} articles")

    logger.info(f"Scraping content for {len(needs_content)} articles")

    stats = ProcessingStats(total_articles=len(needs_content))
    semaphore = asyncio.Semaphore(3)
    batch_size = 5
    processed: List[Article] = []

    with tqdm(
        total=len(needs_content),
        desc="Scraping articles",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ",
    ) as pbar:
        for i in range(0, len(needs_content), batch_size):
            batch = needs_content[i : i + batch_size]
            processed_batch = await process_article_batch(batch, semaphore, stats)
            processed.extend(processed_batch)
            pbar.update(len(batch))
            pbar.set_description(stats.get_progress_string())

    logger.info("Scraping completed:")
    logger.info(stats.get_progress_string())

    return articles


async def async_main() -> None:
    """Main entry point: incremental fetch + scrape + merge into Parquet."""
    parquet_path = PARQUET_PATH

    try:
        api = MiamiHeraldAPI()

        if not parquet_path.exists():
            # First run: full fetch
            logger.info("No existing Parquet found — doing full initial fetch")
            all_articles = api.get_all_articles("Carol Rosenberg")
            logger.info(f"Fetched {len(all_articles)} articles from API")

            # Scrape content
            all_articles = await scrape_content_for_articles(all_articles)

            # Write to Parquet
            article_dicts = [a.to_dict() for a in all_articles]
            count = append_to_parquet(parquet_path, article_dicts)
            logger.info(f"Wrote {count} articles to {parquet_path}")

        else:
            # Incremental update: fetch only new articles
            known_urls = load_existing_urls(parquet_path)
            logger.info(f"Found {len(known_urls)} existing articles in Parquet")

            new_articles = api.get_new_articles("Carol Rosenberg", known_urls)

            if not new_articles:
                logger.info("No new articles found — everything is up to date")
                return

            logger.info(f"Found {len(new_articles)} new articles to process")
            for a in new_articles[:5]:
                logger.info(f"  NEW: {a.title[:80]}")
            if len(new_articles) > 5:
                logger.info(f"  ... and {len(new_articles) - 5} more")

            # Scrape content for new articles
            new_articles = await scrape_content_for_articles(new_articles)

            # Append to Parquet
            article_dicts = [a.to_dict() for a in new_articles]
            count = append_to_parquet(parquet_path, article_dicts)
            logger.info(f"Appended {count} new articles to {parquet_path}")

            # Summary
            table = pq.read_table(parquet_path)
            logger.info(f"Parquet now contains {table.num_rows} total articles")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def main() -> None:
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
