#!/usr/bin/env python3
"""Reset processing status for all articles in the Miami Herald dataset.

This utility script resets the processing status of all articles in the Miami Herald
articles dataset, allowing them to be reprocessed by the entity extraction pipeline.
Useful for testing, debugging, or when processing logic has been updated.
"""

import pandas as pd


def reset_processing_status() -> None:
    """Reset the processing status for all articles in the Miami Herald dataset.

    Reads the Miami Herald articles Parquet file, sets all articles' processing status
    to False (unprocessed), and saves the updated dataset back to the file.

    Raises:
        FileNotFoundError: If the Miami Herald articles Parquet file doesn't exist
        PermissionError: If the script lacks write permissions to the data file
        pd.errors.ParserError: If the Parquet file is corrupted or invalid

    Note:
        Modifies the file in-place. Ensure you have a backup if needed.
        The processing_metadata field is either updated or created if missing.
    """
    # Read the Parquet file
    file_path = "data/guantanamo/raw_sources/miami_herald_articles.parquet"
    df = pd.read_parquet(file_path)

    # Reset the processing status for all rows
    df["processing_metadata"] = df["processing_metadata"].apply(
        lambda x: {**x, "processed": False}
        if isinstance(x, dict)
        else {"processed": False}
    )

    # Save back to Parquet
    df.to_parquet(file_path, index=False)
    print(f"Reset processing status for {len(df)} articles")


if __name__ == "__main__":
    reset_processing_status()
