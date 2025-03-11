import json
from rich import print

ARTICLES_PATH = (
    "/home/strickvl/coding/hinbox/data/raw_sources/miami_herald_articles.jsonl"
)


with open(ARTICLES_PATH, "r") as f:
    entry = f.readline()
    loaded_entry = json.loads(entry)
    article = loaded_entry.get("content")

print(article)
