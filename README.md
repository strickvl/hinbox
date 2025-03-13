# `hinbox`

A tool for processing and extracting information from articles related to Guantánamo Bay detention facility.

## Overview

This project processes articles to extract relevant information about Guantánamo Bay, including people, events, locations, and organizations mentioned in the articles. The processing is done in two main steps:

1. Extract information from raw articles using LLMs (Gemini or Ollama)
2. Process the extracted information to create separate entity files

## Scripts

### Article Processing (`src/v2/run.py`)

This script processes raw articles from a JSONL file, checks if they are relevant to Guantánamo Bay, and extracts information about people, events, locations, and organizations using either Gemini or Ollama LLMs.

#### Usage

```bash
python src/v2/run.py [options]
```

#### Options

- `--local`: Use only local models (Ollama) instead of Gemini
- `--people`: Only extract and print people information
- `--places`: Only extract and print location information
- `--orgs`: Only extract and print organization information
- `--events`: Only extract and print event information
- `--tags`: Only extract and print article tags
- `--show-article`: Show the article text during processing
- `--limit N`: Limit the number of articles to process (default: 5)
- `--output PATH`: Path to the output file (default: data/processed/processed_articles.jsonl)

#### Example

Process 10 articles using Gemini and extract all information:

```bash
python src/v2/run.py --limit 10
```

Process 5 articles using Ollama and only extract people and organizations:

```bash
python src/v2/run.py --local --people --orgs
```

### Entity Extraction (`src/v2/extract_entities.py`)

This script reads the processed articles from a JSONL file and extracts people, events, locations, and organizations into separate JSONL files for easier analysis.

#### Usage

```bash
python src/v2/extract_entities.py [options]
```

#### Options

- `--input PATH`: Path to the input JSONL file (default: data/processed/processed_articles.jsonl)
- `--output-dir DIR`: Directory to store output files (default: data/entities)
- `--limit N`: Limit the number of articles to process (default: process all)

#### Example

Extract entities from all processed articles:

```bash
python src/v2/extract_entities.py
```

Extract entities from a specific file and save to a custom directory:

```bash
python src/v2/extract_entities.py --input custom_articles.jsonl --output-dir custom_entities
```

## Output Files

The entity extraction script creates the following output files:

- `data/entities/people.jsonl`: Information about people mentioned in the articles
- `data/entities/events.jsonl`: Information about events mentioned in the articles
- `data/entities/locations.jsonl`: Information about locations mentioned in the articles
- `data/entities/organizations.jsonl`: Information about organizations mentioned in the articles

Each entity includes information about the article it came from (ID, title, URL, published date) as well as entity-specific information.

## Development

### Dependencies

This project uses Python 3.9+ and requires the following packages:
- litellm
- pydantic
- rich

### Setup

1. Clone the repository
2. Install dependencies using uv:
   ```bash
   uv add litellm pydantic rich
   ```

### Workflow

1. Process raw articles using `src/v2/run.py`
2. Extract entities from processed articles using `src/v2/extract_entities.py`
3. Analyze the extracted entities as needed 
