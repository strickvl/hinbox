"""
Constants used throughout the application.
"""

import os

# Model names
GEMINI_MODEL = "openrouter/google/gemini-2.0-flash-001"
OLLAMA_MODEL = "mistral-small"

# API endpoints
OLLAMA_API_URL = "http://192.168.178.175:11434/v1"
OLLAMA_API_KEY = "ollama"

# File paths
ARTICLES_PATH = "data/raw_sources/miami_herald_articles.jsonl"
OUTPUT_DIR = "data/entities"
PROCESSED_ARTICLES_PATH = "data/processed/processed_articles.jsonl"

PEOPLE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "people.jsonl")
EVENTS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "events.jsonl")
LOCATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "locations.jsonl")
ORGANIZATIONS_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "organizations.jsonl")
