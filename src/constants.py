"""
Constants used throughout the application.
"""

import os

CLOUD_MODEL = "gemini/gemini-2.0-flash"
OLLAMA_MODEL = "ollama/gemma3:27b"  # Updated to include full path

# Embedding model names
CLOUD_EMBEDDING_MODEL = "jina_ai/jina-embeddings-v3"
LOCAL_EMBEDDING_MODEL = "huggingface/jinaai/jina-embeddings-v3"

# API endpoints
OLLAMA_API_URL = "http://192.168.178.175:11434/v1"
OLLAMA_API_KEY = "ollama"


# Helper function to get bare model name for Ollama API
def get_ollama_model_name(model: str) -> str:
    """Strip 'ollama/' prefix if present for Ollama API calls."""
    return model.replace("ollama/", "") if model.startswith("ollama/") else model


# File paths
ARTICLES_PATH = "data/raw_sources/miami_herald_articles.parquet"
OUTPUT_DIR = "data/entities"

PEOPLE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "people.parquet")
EVENTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "events.parquet")
LOCATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locations.parquet")
ORGANIZATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "organizations.parquet")

# Threshold for similarity matching
SIMILARITY_THRESHOLD = 0.75
