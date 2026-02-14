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
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1").strip()
OLLAMA_API_KEY = "ollama"


# Helper function to get bare model name for Ollama API
def get_ollama_model_name(model: str) -> str:
    """Strip 'ollama/' prefix if present for Ollama API calls."""
    return model.replace("ollama/", "") if model.startswith("ollama/") else model


# Threshold for similarity matching
SIMILARITY_THRESHOLD = 0.75

# Retry configuration for LLM API calls
MAX_RETRIES = 3
BASE_DELAY = 2.0

# LLM generation defaults
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0
MAX_ITERATIONS = 3

# Frontend configuration
DEFAULT_FRONTEND_PORT = 5001
HASH_TRUNCATE_LENGTH = 6

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Braintrust project id (optional)
BRAINTRUST_PROJECT_ID = os.getenv("BRAINTRUST_PROJECT_ID", "").strip() or None

# Profile versioning feature flag
ENABLE_PROFILE_VERSIONING = (
    os.getenv("ENABLE_PROFILE_VERSIONING", "true").lower() == "true"
)
