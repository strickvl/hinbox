import json
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class SearchConfig:
    query: str
    start: int = 0
    rows: int = 11
    sort: str = "newest"


class MiamiHeraldAPI:
    BASE_URL = "https://publicapi.misitemgr.com/webapi-public/v2/publications/miamiherald/search"

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
        print("\n=== Request Details ===")
        print(f"URL: {self.BASE_URL}")
        payload = {
            "q": config.query,
            "start": config.start,
            "rows": config.rows,
            "sort": config.sort,
        }

        print("\n=== Request Payload ===")
        print(json.dumps(payload, indent=2))

        print("\n=== Request Headers ===")
        print(json.dumps(self.headers, indent=2))

        try:
            print("\n=== Making Request ===")
            response = requests.post(
                self.BASE_URL, headers=self.headers, data=json.dumps(payload)
            )

            print(f"\n=== Response Status: {response.status_code} ===")
            print("Response Headers:")
            print(json.dumps(dict(response.headers), indent=2))

            print("\n=== Response Content ===")
            print(
                response.text[:500] + "..."
                if len(response.text) > 500
                else response.text
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            raise


def main() -> None:
    api = MiamiHeraldAPI()
    config = SearchConfig(query="Carol Rosenberg")

    try:
        results = api.search_articles(config)

        # Print results in a readable format
        if "results" in results:
            print(f"\nFound {results.get('total', 0)} total articles")
            print("-" * 80)
            for article in results["results"]:
                print(f"\nTitle: {article.get('title', 'No title')}")
                print(f"Type: {article.get('type', 'No type')}")
                print(f"URL: {article.get('url', 'No URL')}")
                print(f"Published: {article.get('published_date', 'No date')}")
                print("-" * 80)
        else:
            print("No articles found in the response")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
