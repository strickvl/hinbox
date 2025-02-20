import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Dict, Iterator

import requests
from tqdm import tqdm


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

    @classmethod
    def from_api_response(cls, article_data: Dict[str, Any]) -> "Article":
        return cls(
            title=article_data.get("title", ""),
            url=article_data.get("url", ""),
            published_date=article_data.get("published_date", ""),
            scrape_timestamp=datetime.now(UTC).isoformat(),
        )


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


def save_articles_to_jsonl(articles: Iterator[Article], output_path: Path) -> None:
    """
    Save articles to a JSONL file.

    Args:
        articles: Iterator of Article objects
        output_path: Path to save the JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    article_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(asdict(article)) + "\n")
            article_count += 1

    print(f"\nSaved {article_count} articles to {output_path}")


def main() -> None:
    api = MiamiHeraldAPI()
    output_path = Path("data/raw_sources/miami_herald_articles.jsonl")

    try:
        articles = api.get_all_articles("Carol Rosenberg")
        save_articles_to_jsonl(articles, output_path)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
