#!/usr/bin/env python3

import pandas as pd


def reset_processing_status():
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
