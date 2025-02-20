import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterator, Optional, Tuple

import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import requests
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
        """
        Search for articles using the provided configuration.

        Args:
            config: SearchConfig object containing search parameters

        Returns:
            Dict containing the API response

        Raises:
            requests.exceptions.RequestException: If the API request fails
        """
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

    def get_all_articles(self, query: str) -> Iterator[Article]:
        """
        Get all articles for a given query using pagination.

        Args:
            query: Search query string

        Yields:
            Article objects for each article found
        """
        start = 0
        total_fetched = 0
        total_available = None

        while True:
            config = SearchConfig(query=query, start=start)
            results = self.search_articles(config)

            if total_available is None:
                total_available = results.get("total", 0)
                print(
                    f"\nFetching {total_available} articles (with {config.rows} per batch)..."
                )
                progress_bar = tqdm(total=total_available, desc="Articles fetched")

            if not results.get("results"):
                break

            batch_count = 0
            for article_data in results["results"]:
                yield Article.from_api_response(article_data)
                total_fetched += 1
                batch_count += 1

            progress_bar.update(batch_count)

            if total_fetched >= total_available:
                progress_bar.close()
                break

            start += config.rows


async def get_article_text(url: str) -> Optional[Tuple[str, str]]:
    """Get article title and content from URL.

    Args:
        url: The URL of the article to scrape

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
                    lambda route: route.abort()
                    if route.request.resource_type in ["image", "stylesheet", "font"]
                    else route.continue_(),
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
    processing_times: list[float] = None
    start_time: float = 0.0
    last_save_count: int = 0  # Track when we last saved

    def __post_init__(self):
        self.processing_times = []
        self.start_time = datetime.now(UTC).timestamp()

    def should_save_progress(self) -> bool:
        """Check if we should save progress based on processed count."""
        return (self.processed_count - self.last_save_count) >= SAVE_FREQUENCY

    def update_last_save(self) -> None:
        """Update the last save count."""
        self.last_save_count = self.processed_count

    def add_processing_time(self, time_taken: float) -> None:
        """Add a processing time to the moving average calculation."""
        self.processing_times.append(time_taken)
        # Keep only the last 10 times for moving average
        if len(self.processing_times) > 10:
            self.processing_times.pop(0)

    def get_avg_processing_time(self) -> float:
        """Get moving average of processing times."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_estimated_remaining_time(self) -> float:
        """Get estimated remaining time in seconds."""
        if not self.processing_times:
            return 0.0
        remaining_articles = self.total_articles - self.processed_count
        return remaining_articles * self.get_avg_processing_time()

    def format_time(self, seconds: float) -> str:
        """Format time in seconds to a human-readable string."""
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def get_progress_string(self) -> str:
        """Get a formatted progress string."""
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
    articles: list[Article], semaphore: asyncio.Semaphore, stats: ProcessingStats
) -> list[Article]:
    """Process a batch of articles to fetch their content.

    Args:
        articles: List of Article objects to process
        semaphore: Semaphore to control concurrent requests
        stats: ProcessingStats object for tracking progress

    Returns:
        List of processed Article objects
    """

    async def process_single_article(article: Article) -> Article:
        async with semaphore:
            if not article.content:  # Only process if content not already fetched
                start_time = datetime.now(UTC).timestamp()
                try:
                    result = await get_article_text(article.url)
                    if result:
                        _, content = result  # We already have the title
                        article.content = content
                        article.content_scrape_timestamp = datetime.now(UTC).isoformat()
                        stats.success_count += 1
                    else:
                        stats.failure_count += 1

                    # Update processing time statistics
                    end_time = datetime.now(UTC).timestamp()
                    stats.add_processing_time(end_time - start_time)

                except Exception as e:
                    logger.error(f"Error processing article {article.url}: {e}")
                    stats.failure_count += 1

                stats.processed_count += 1
                await asyncio.sleep(ARTICLE_SCRAPE_DELAY)  # Respectful delay
        return article

    return await asyncio.gather(
        *(process_single_article(article) for article in articles)
    )


async def save_progress(
    articles: list[Article], output_path: Path, stats: ProcessingStats
) -> None:
    """Save current progress to the JSONL file.

    Args:
        articles: List of processed articles
        output_path: Path to save the JSONL file
        stats: Processing stats for logging
    """
    temp_path = output_path.with_suffix(".jsonl.tmp")

    try:
        # Write to temporary file first
        with temp_path.open("w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(article.to_dict()) + "\n")

        # Rename temporary file to actual file
        temp_path.replace(output_path)
        stats.update_last_save()
        logger.info(
            f"Progress saved: {stats.processed_count}/{stats.total_articles} articles processed"
        )

    except Exception as e:
        logger.error(f"Error saving progress: {e}")
        if temp_path.exists():
            temp_path.unlink()


async def update_articles_with_content(jsonl_path: Path) -> None:
    """Update articles in the JSONL file with their content.

    Args:
        jsonl_path: Path to the JSONL file
    """
    # Read existing articles
    articles = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            article_data = json.loads(line)
            articles.append(Article.from_dict(article_data))

    # In test mode, only process a small subset
    if TEST_MODE:
        original_count = len(articles)
        articles = articles[:TEST_SAMPLE_SIZE]
        logger.info(
            f"TEST MODE: Processing {TEST_SAMPLE_SIZE} articles out of {original_count}"
        )

    logger.info(f"Found {len(articles)} articles to process")

    # Initialize processing stats
    stats = ProcessingStats(total_articles=len(articles))

    # Process articles in batches with controlled concurrency
    semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
    batch_size = 5
    processed_articles = []

    with tqdm(
        total=len(articles),
        desc="Processing articles",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ",
    ) as pbar:
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            processed_batch = await process_article_batch(batch, semaphore, stats)
            processed_articles.extend(processed_batch)
            pbar.update(len(batch))

            # Update progress description with detailed stats
            pbar.set_description(stats.get_progress_string())

            # Save progress periodically
            if stats.should_save_progress():
                if TEST_MODE:
                    # In test mode, save to a different file to avoid corrupting main data
                    test_path = jsonl_path.with_suffix(".test.jsonl")
                    await save_progress(processed_articles, test_path, stats)
                else:
                    await save_progress(processed_articles, jsonl_path, stats)

    # Final statistics
    logger.info("\nProcessing completed:")
    logger.info(stats.get_progress_string())
    logger.info(
        f"Average processing time per article: "
        f"{stats.format_time(stats.get_avg_processing_time())}"
    )

    # Save final results
    if TEST_MODE:
        test_path = jsonl_path.with_suffix(".test.jsonl")
        await save_progress(processed_articles, test_path, stats)
        logger.info(f"Test results saved to: {test_path}")
    else:
        await save_progress(processed_articles, jsonl_path, stats)
        logger.info(f"Updated {len(processed_articles)} articles with content")


def save_articles_to_jsonl(articles: Iterator[Article], output_path: Path) -> None:
    """Save articles to a JSONL file.

    Args:
        articles: Iterator of Article objects
        output_path: Path to save the JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article.to_dict()) + "\n")
            article_count += 1

    print(f"\nSaved {article_count} articles to {output_path}")


async def async_main() -> None:
    """Async main function to handle article processing."""
    output_path = Path("data/raw_sources/miami_herald_articles.jsonl")

    try:
        # Check if we need to fetch new articles first
        api = MiamiHeraldAPI()
        if not output_path.exists():
            # If file doesn't exist, fetch and save initial articles
            articles = api.get_all_articles("Carol Rosenberg")
            save_articles_to_jsonl(articles, output_path)

        # Now update articles with content
        if output_path.exists():
            if TEST_MODE:
                logger.info(
                    "Running in TEST MODE - processing small subset of articles"
                )
            await update_articles_with_content(output_path)

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def main() -> None:
    """Main entry point."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
