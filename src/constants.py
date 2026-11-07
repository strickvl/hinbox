"""
Constants used throughout the application.
"""

import os

# GPT-6 Luna: scored on every eval (relevance, merge, profile update); Gemini 2.0
# Flash, the old default, never was
# What profile prompts call one entity of each table ("the person X", not
# "the peopl X", which ``entity_type[:-1]`` gave)
ENTITY_SINGULAR = {
    "people": "person",
    "organizations": "organization",
    "locations": "location",
    "events": "event",
}

CLOUD_MODEL = os.getenv("HINBOX_CLOUD_MODEL", "openai/gpt-6-luna")
OLLAMA_MODEL = os.getenv("HINBOX_OLLAMA_MODEL", "ollama/qwen2.5:32b-instruct-q5_K_M")
# Sent to Ollama as ``reasoning_effort``. "none" turns thinking off for models
# that think by default (e.g. Qwen3.6); set to "" to leave the model's default.
OLLAMA_REASONING_EFFORT = os.getenv("HINBOX_OLLAMA_REASONING_EFFORT", "none").strip()
# Thinking spends output tokens before the answer; with the default 2048 limit
# Nemotron 3.5 Bolt ran out on 8 of 161 merge checks. Used as the minimum token
# limit for Ollama calls whenever thinking isn't switched off.
OLLAMA_THINKING_MAX_TOKENS = int(os.getenv("HINBOX_OLLAMA_THINKING_MAX_TOKENS", "8192"))

# Embedding model names
CLOUD_EMBEDDING_MODEL = "jina_ai/jina-embeddings-v3"
LOCAL_EMBEDDING_MODEL = "huggingface/jinaai/jina-embeddings-v3"

# API endpoints
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/v1").strip()

# Threshold for similarity matching
SIMILARITY_THRESHOLD = 0.75

# Retry configuration for LLM API calls
MAX_RETRIES = 3
BASE_DELAY = 2.0

# LLM generation defaults
DEFAULT_MAX_TOKENS = 2048
# A profile is rewritten whole on every update, so its output grows with it.
# At 2048 tokens profiles stop at roughly 6-7k characters: the JSON is cut off
# mid-way and the update fails.
PROFILE_MAX_TOKENS = int(os.getenv("HINBOX_PROFILE_MAX_TOKENS", "16384"))
# Extraction returns every entity of one type in one reply. For long articles
# GPT-6 Luna's events list ran past 2048 tokens, all 3 tries were cut off and
# the article was saved with no events (3 of the first ~10 articles of a run).
EXTRACTION_MAX_TOKENS = int(os.getenv("HINBOX_EXTRACTION_MAX_TOKENS", "8192"))
# How update_profile() changes an existing profile. "edit": one call returns a
# list of add/replace edits and code applies them (src/engine/profile_edits.py).
# "rewrite": the model writes the whole profile again, then up to 3
# reflect/regenerate rounds. On the profile-update eval, edit scored 0.89 vs
# 0.29 on Luna and ran ~10x faster (2.2s vs 21s per update).
PROFILE_UPDATE_APPROACH = os.getenv("HINBOX_PROFILE_UPDATE_APPROACH", "edit").strip()
# A hub profile ("United States", ~50k characters from 83 articles) was sent
# whole on every update and every merge check against it. Over these sizes
# the model sees a shortened copy instead (src/engine/profile_view.py: the
# opening, the headings, and the parts closest to the new article); the
# stored profile keeps everything. 0 turns the shortening off.
PROFILE_VIEW_CHARS = int(os.getenv("HINBOX_PROFILE_VIEW_CHARS", "16000"))
MATCH_PROFILE_CHARS = int(os.getenv("HINBOX_MATCH_PROFILE_CHARS", "8000"))
# How create_profile() writes a first profile. "check" (default): one draft,
# code fixes, at most one repair call (src/engine/profile_create.py). "reflect":
# a draft, then up to 3 rounds of a model reviewer (which never sees the
# article) and a fresh redraft. On 6 labelled first articles both stated the
# labelled facts (Luna 26/26 each, Qwen 25/26 each) with 1/2 (Luna) to 1/3
# (Qwen) of the calls.
PROFILE_CREATE_APPROACH = os.getenv("HINBOX_PROFILE_CREATE_APPROACH", "check").strip()
DEFAULT_TEMPERATURE = 0
# Output limit for reflection verdicts (valid, reasoning, issues)
REFLECTION_MAX_TOKENS = int(os.getenv("HINBOX_REFLECTION_MAX_TOKENS", "1024"))
MAX_ITERATIONS = 3

# Merge dispute / gray-band thresholds
MERGE_GRAY_BAND_DELTA = 0.05  # similarity within ±delta of threshold is "gray band"
MERGE_UNCERTAIN_CONFIDENCE_CUTOFF = (
    0.7  # match-check confidence below this is "uncertain"
)

# Quality control thresholds
QC_MIN_NAME_LENGTH = 3
PROFILE_QC_MIN_TEXT_LENGTH = 100
PROFILE_QC_MIN_TAG_COUNT = 1

# Frontend configuration
DEFAULT_FRONTEND_PORT = 5001
HASH_TRUNCATE_LENGTH = 6

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Profile versioning feature flag
ENABLE_PROFILE_VERSIONING = (
    os.getenv("ENABLE_PROFILE_VERSIONING", "true").lower() == "true"
)
