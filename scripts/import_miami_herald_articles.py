#!/usr/bin/env python3
"""
Script to import Miami Herald articles from a JSONL file into the database.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, select

from src.database import get_engine
from src.models import Article

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

JSONL_PATH = Path("data/raw_sources/miami_herald_articles.jsonl")


def parse_datetime(timestamp_str):
    """Parse datetime string or unix timestamp into datetime object."""
    if isinstance(timestamp_str, (int, float)):
        # Assume Unix timestamp
        return datetime.fromtimestamp(timestamp_str)
    else:
        # Try to parse various formats
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse datetime: {timestamp_str}")
            return datetime.now()


def import_articles():
    """Import articles from JSONL file into the database."""
    engine = get_engine()
    
    if not JSONL_PATH.exists():
        logger.error(f"JSONL file not found: {JSONL_PATH}")
        return
    
    # Count articles to import
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        total_articles = sum(1 for _ in f)
    
    logger.info(f"Found {total_articles} articles in {JSONL_PATH}")
    
    # Process articles
    articles_added = 0
    articles_skipped = 0
    articles_failed = 0
    
    # Use a new session for each batch to avoid transaction issues
    batch_size = 100
    current_batch = []
    
    # Read articles from JSONL file
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i % 100 == 0 or i == total_articles:
                logger.info(f"Processing article {i}/{total_articles}")
            
            try:
                article_data = json.loads(line.strip())
                
                # Skip articles with missing required fields
                if not article_data.get("content"):
                    logger.warning(f"Skipping article with missing content: {article_data['url']}")
                    articles_skipped += 1
                    continue
                
                # Create new article object
                article = Article(
                    title=article_data["title"],
                    url=article_data["url"],
                    published_date=parse_datetime(article_data["published_date"]),
                    content=article_data["content"],
                    scrape_timestamp=parse_datetime(article_data["scrape_timestamp"]),
                    content_scrape_timestamp=parse_datetime(article_data.get("content_scrape_timestamp"))
                )
                
                # Add to current batch
                current_batch.append(article)
                
                # Process batch if we've reached batch size
                if len(current_batch) >= batch_size:
                    processed = process_batch(engine, current_batch)
                    articles_added += processed
                    articles_failed += len(current_batch) - processed
                    current_batch = []
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON on line {i}")
                articles_failed += 1
            except KeyError as e:
                logger.error(f"Missing required field in article on line {i}: {e}")
                articles_failed += 1
            except Exception as e:
                logger.error(f"Error processing article on line {i}: {e}")
                articles_failed += 1
        
        # Process any remaining articles
        if current_batch:
            processed = process_batch(engine, current_batch)
            articles_added += processed
            articles_failed += len(current_batch) - processed
    
    logger.info(f"Import complete. Added {articles_added} articles, skipped {articles_skipped} articles, failed {articles_failed} articles.")


def process_batch(engine, articles):
    """Process a batch of articles, adding them to the database.
    
    Returns the number of articles successfully added.
    """
    successful = 0
    
    with Session(engine) as session:
        try:
            for article in articles:
                try:
                    # Check if article already exists
                    existing = session.exec(
                        select(Article).where(Article.url == article.url)
                    ).first()
                    
                    if existing:
                        logger.debug(f"Article already exists: {article.url}")
                        continue
                    
                    # Add article to session
                    session.add(article)
                    successful += 1
                except Exception as e:
                    logger.error(f"Error adding article {article.url}: {e}")
                    # Continue with next article
                    continue
            
            # Commit the batch
            session.commit()
            logger.info(f"Committed batch of {successful} articles")
            
        except Exception as e:
            # If the batch commit fails, roll back
            logger.error(f"Error committing batch: {e}")
            session.rollback()
            
    return successful


if __name__ == "__main__":
    logger.info("Starting import of Miami Herald articles")
    import_articles()
    logger.info("Import completed")