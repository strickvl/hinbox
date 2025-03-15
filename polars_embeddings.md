# Technical Guide: Using Parquet and Polars for Embedding Storage and Retrieval

This guide documents a practical approach for efficiently storing and working with text embeddings using Parquet files and the Polars library. This approach is particularly well-suited for projects where vector databases may be unnecessary overhead.

## Overview

When working with text embeddings (particularly from LLMs), efficient storage and retrieval are essential. This guide demonstrates how to use:

1. **Parquet** - A columnar file format that efficiently stores typed data, including nested arrays
2. **Polars** - A high-performance DataFrame library with native support for working with Parquet files and vector data

This approach is ideal for datasets of up to several hundred thousand embeddings before vector databases become necessary.

## Prerequisites

```python
import numpy as np
import polars as pl
```

## Storing Embeddings in Parquet Files

### Creating a DataFrame with Embeddings

```python
# Assume you have:
# - embeddings: a numpy matrix of shape (n_items, embedding_dim) with dtype=np.float32
# - metadata: a dictionary or dataframe with metadata for each item

# Create a polars DataFrame
df = pl.DataFrame({
    "id": range(len(embeddings)),
    "text": metadata["text"],  # or other metadata columns
    # Add additional metadata columns as needed
})

# Add the embeddings as a column (will be stored as an array of float32 values)
df = df.with_columns(embedding=embeddings)

# Save to Parquet
df.write_parquet("embeddings.parquet")
```

### Reading Embeddings from Parquet

```python
# Read the full DataFrame
df = pl.read_parquet("embeddings.parquet")

# Or select only specific columns for efficiency
df = pl.read_parquet("embeddings.parquet", columns=["id", "text", "embedding"])

# Extract embeddings matrix for similarity calculations (zero-copy operation)
embeddings_matrix = df["embedding"].to_numpy(allow_copy=False)
```

## Fast Similarity Search with Numpy

### Key Function for Similarity Search

```python
def fast_dot_product(query, matrix, k=3):
    """
    Find the top k most similar vectors to the query vector.
    
    Args:
        query: Normalized query vector (1D numpy array)
        matrix: Normalized embedding matrix (2D numpy array)
        k: Number of similar items to return
        
    Returns:
        idx: Indices of top k similar items
        score: Similarity scores for those items
    """
    # Compute dot products between query and all embeddings
    dot_products = query @ matrix.T
    
    # Find indices of top k results efficiently
    idx = np.argpartition(dot_products, -k)[-k:]
    idx = idx[np.argsort(dot_products[idx])[::-1]]
    
    # Get corresponding similarity scores
    score = dot_products[idx]
    
    return idx, score
```

### Performing Similarity Search

```python
# Normalize query embedding if not already normalized
query_embedding = query_embedding / np.linalg.norm(query_embedding)

# Get similar item indices and scores
idx, scores = fast_dot_product(query_embedding, embeddings_matrix, k=5)

# Retrieve the corresponding metadata for similar items
similar_items = df[idx]

# Display results with scores
for i, (item, score) in enumerate(zip(similar_items.rows(named=True), scores)):
    print(f"{i+1}. {item['text']} (Score: {score:.4f})")
```

## Dynamic Filtering Before Similarity Search

One of the advantages of this approach is the ability to filter the dataset on metadata before performing similarity searches.

```python
# Filter the DataFrame based on metadata criteria
filtered_df = df.filter(
    pl.col("category").is_in(["news", "article"]),
    pl.col("date") > "2023-01-01"
)

# Extract the filtered embeddings (this creates a copy)
filtered_embeddings = filtered_df["embedding"].to_numpy()

# Perform similarity search on the filtered subset
idx, scores = fast_dot_product(query_embedding, filtered_embeddings, k=5)

# Get the corresponding items from the filtered DataFrame
similar_items = filtered_df[idx]
```

## Handling Multiple Files and Appending Data

### Combining Multiple Embedding Sets

```python
# Read multiple embedding files
df1 = pl.read_parquet("embeddings_set1.parquet")
df2 = pl.read_parquet("embeddings_set2.parquet")

# Combine them (vertical concatenation)
combined_df = pl.concat([df1, df2])

# Save combined dataset
combined_df.write_parquet("combined_embeddings.parquet")
```

### Appending New Embeddings

```python
# Read existing embeddings
df = pl.read_parquet("embeddings.parquet")

# Create DataFrame with new embeddings
new_df = pl.DataFrame({
    "id": range(len(df), len(df) + len(new_embeddings)),
    "text": new_metadata["text"],
})
new_df = new_df.with_columns(embedding=new_embeddings)

# Append new embeddings
updated_df = pl.concat([df, new_df])

# Save updated dataset
updated_df.write_parquet("embeddings.parquet")
```

## Performance Considerations

### Memory Usage Estimates

For `float32` embeddings with dimension 768 (common for smaller LLM embeddings):
- Each embedding vector requires 768 * 4 bytes = 3.072 KB
- 100,000 embeddings would require ~300 MB of memory
- 1,000,000 embeddings would require ~3 GB of memory

### Optimization Tips

1. **Limit columns when reading**: Use the `columns` parameter with `pl.read_parquet()` to load only necessary data
2. **Normalize embeddings before storage**: Pre-normalize embeddings to unit vectors to simplify similarity calculations
3. **Use lazy evaluation**: For very large datasets, consider Polars' lazy API
   ```python
   df = pl.scan_parquet("large_embeddings.parquet").select(["id", "embedding"])
   ```
4. **Consider chunking**: For datasets too large to fit in memory, process in chunks:
   ```python
   chunk_size = 100000
   for i in range(0, len(query_embeddings), chunk_size):
       chunk = query_embeddings[i:i+chunk_size]
       # Process chunk...
   ```

## When to Move to a Vector Database

Consider migrating to a dedicated vector database when:

1. Your dataset exceeds several million embeddings
2. You need advanced features like approximate nearest neighbors (ANN) search
3. You require distributed queries across multiple machines
4. You need real-time updates with immediate query availability

## Conclusion

The Parquet + Polars approach offers a lightweight, efficient solution for working with embeddings that strikes a balance between simplicity and performance. It's particularly well-suited for projects where dedicated vector databases would add unnecessary complexity.

This method provides:
- Efficient storage with proper typing for embeddings
- Fast similarity calculations
- Easy filtering on metadata
- Portable files that work across environments

For many applications up to several hundred thousand embeddings, this approach
may be all you need.
