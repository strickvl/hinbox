#!/usr/bin/env python3
"""
Script to check the articles in the database.
"""

import logging
import sys
from datetime import datetime, timedelta

from sqlmodel import Session, func, select

from src.database import get_engine
from src.models import Article

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_articles():
    """Check the articles in the database."""
    engine = get_engine()

    with Session(engine) as session:
        # Count total articles
        total_count = session.exec(select(func.count()).select_from(Article)).one()
        logger.info(f"Total articles in database: {total_count}")

        # Get most recent article
        most_recent = session.exec(
            select(Article).order_by(Article.published_date.desc()).limit(1)
        ).first()

        if most_recent:
            logger.info(
                f"Most recent article: {most_recent.title} ({most_recent.published_date})"
            )

        # Get oldest article
        oldest = session.exec(
            select(Article).order_by(Article.published_date.asc()).limit(1)
        ).first()

        if oldest:
            logger.info(f"Oldest article: {oldest.title} ({oldest.published_date})")

        # Check articles in the last year
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_count = session.exec(
            select(func.count())
            .select_from(Article)
            .where(Article.published_date >= one_year_ago)
        ).one()

        logger.info(f"Articles published in the last year: {recent_count}")

        # Display a sample article
        if len(sys.argv) > 1 and sys.argv[1] == "--sample":
            sample = session.exec(select(Article).limit(1)).first()
            if sample:
                logger.info(f"\nSample article:\n")
                logger.info(f"Title: {sample.title}")
                logger.info(f"URL: {sample.url}")
                logger.info(f"Published: {sample.published_date}")
                logger.info(f"Content sample: {sample.content[:200]}...")


if __name__ == "__main__":
    logger.info("Checking articles in database")
    check_articles()
    logger.info("Check completed")
