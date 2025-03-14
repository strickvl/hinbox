"""
Constants used throughout the application.
"""

import os

# LLM model names
GEMINI_MODEL = "gemini/gemini-2.0-flash"
OLLAMA_MODEL = "ollama/mistral-small"  # Updated to include full path

# Embedding model names
CLOUD_EMBEDDING_MODEL = "openai/text-embedding-3-large"
LOCAL_EMBEDDING_MODEL = "ollama/nomic-embed-text"

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
PROCESSED_ARTICLES_PATH = "data/processed/processed_articles.parquet"

PEOPLE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "people.parquet")
EVENTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "events.parquet")
LOCATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locations.parquet")
ORGANIZATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "organizations.parquet")

# Threshold for similarity matching
SIMILARITY_THRESHOLD = 0.9
