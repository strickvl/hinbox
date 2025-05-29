#!/usr/bin/env python3
"""
Script to check the articles in the parquet file.
"""

import argparse
import os

import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from src.constants import ARTICLES_PATH

console = Console()


def check_articles(sample=False):
    """Check the articles in the parquet file."""

    if not os.path.exists(ARTICLES_PATH):
        console.print(f"[red]ERROR: Articles file not found at {ARTICLES_PATH}[/red]")
        return

    try:
        # Read the parquet file
        table = pq.read_table(ARTICLES_PATH)
        rows = table.to_pylist()

        # Basic statistics
        total_count = len(rows)
        console.print(f"\n[cyan]Total articles in parquet file: {total_count}[/cyan]")

        if total_count == 0:
            console.print("[yellow]No articles found in the file.[/yellow]")
            return

        # Count processed vs unprocessed
        processed_count = 0
        relevance_checked = 0

        for row in rows:
            metadata = row.get("processing_metadata", {})
            if metadata.get("processed", False):
                processed_count += 1
            if metadata.get("relevance_checked", False):
                relevance_checked += 1

        # Create summary table
        summary_table = Table(title="Article Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_column("Percentage", style="yellow")

        summary_table.add_row("Total Articles", str(total_count), "100%")
        summary_table.add_row(
            "Processed",
            str(processed_count),
            f"{(processed_count / total_count) * 100:.1f}%",
        )
        summary_table.add_row(
            "Unprocessed",
            str(total_count - processed_count),
            f"{((total_count - processed_count) / total_count) * 100:.1f}%",
        )
        summary_table.add_row(
            "Relevance Checked",
            str(relevance_checked),
            f"{(relevance_checked / total_count) * 100:.1f}%",
        )

        console.print(summary_table)

        # Find most recent and oldest articles
        articles_with_dates = [r for r in rows if r.get("published_date")]
        if articles_with_dates:
            sorted_articles = sorted(
                articles_with_dates, key=lambda x: x.get("published_date", "")
            )

            oldest = sorted_articles[0]
            newest = sorted_articles[-1]

            console.print(f"\n[blue]Date Range:[/blue]")
            console.print(
                f"  Oldest: {oldest['title'][:60]}... ({oldest['published_date']})"
            )
            console.print(
                f"  Newest: {newest['title'][:60]}... ({newest['published_date']})"
            )

        # Show sample article if requested
        if sample and rows:
            console.print(f"\n[magenta]Sample Article:[/magenta]")
            sample_article = rows[0]

            # Create article details table
            article_table = Table(show_header=False)
            article_table.add_column("Field", style="cyan")
            article_table.add_column("Value", style="white")

            article_table.add_row("Title", sample_article.get("title", "N/A"))
            article_table.add_row("URL", sample_article.get("url", "N/A"))
            article_table.add_row(
                "Published", sample_article.get("published_date", "N/A")
            )
            article_table.add_row("Author", sample_article.get("author", "N/A"))

            content = sample_article.get("content", "")
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                article_table.add_row("Content Preview", preview)

            # Show processing metadata if exists
            metadata = sample_article.get("processing_metadata", {})
            if metadata:
                article_table.add_row(
                    "Processed", str(metadata.get("processed", False))
                )
                article_table.add_row(
                    "Processing Date", metadata.get("processing_date", "N/A")
                )

            console.print(article_table)

    except Exception as e:
        console.print(f"[red]Error reading parquet file: {e}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check articles in parquet file")
    parser.add_argument(
        "--sample", action="store_true", help="Display a sample article"
    )
    args = parser.parse_args()

    console.print("[bold green]Checking articles in parquet file[/bold green]")
    check_articles(sample=args.sample)
    console.print("\n[bold green]Check completed[/bold green]")

