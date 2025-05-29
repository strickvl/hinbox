# Domain Configuration Guide

This directory contains domain-specific configurations for the Hinbox entity extraction system. Each domain defines its own entity types, extraction prompts, and processing settings.

## Quick Start

> **Note**: This project supports both `./run.py` commands and `just` commands. Use whichever you prefer!

### Creating a New Domain

1. **Initialize a new domain**:
   ```bash
   ./run.py init football
   # OR using justfile:
   just init football
   ```
   This copies the template to `configs/football/`

2. **Configure your domain**:
   - Edit `configs/football/config.yaml` - Set domain name, description, and data paths
   - Edit `configs/football/categories/*.yaml` - Define entity types relevant to your domain
   - Edit `configs/football/prompts/*.md` - Customize extraction prompts

3. **Process articles**:
   ```bash
   ./run.py process --domain football
   # OR using justfile:
   just process-domain football
   ```

### Configuration Files

Each domain contains these files:

#### `config.yaml`
Main configuration including domain name, data paths, and processing settings.

#### `categories/`
YAML files defining entity types:
- `people.yaml` - Types of people relevant to your domain
- `organizations.yaml` - Types of organizations
- `locations.yaml` - Types of locations  
- `events.yaml` - Types of events and tags

#### `prompts/`
Markdown files with extraction instructions:
- `people.md` - How to extract and categorize people
- `organizations.md` - How to extract organizations
- `locations.md` - How to extract locations
- `events.md` - How to extract events
- `relevance.md` - How to determine article relevance

## Example Domains

### Guantánamo Bay (`configs/guantanamo/`)
The default domain focusing on detention, legal proceedings, and human rights issues.

**Key entity types**:
- People: detainee, military, government, lawyer, journalist
- Organizations: military, intelligence, legal, humanitarian, advocacy
- Locations: detention_facility, military_base, country, city
- Events: detention, legal, military_operation, policy_change, protest

### Template (`configs/template/`)
Starting point for new domains with generic categories and prompts.

## Customization Tips

### 1. Entity Types
- Keep categories specific to your domain
- Provide clear descriptions and examples
- Remove irrelevant types, add domain-specific ones
- Use lowercase, underscore-separated names

### 2. Prompts
- Write in natural language - these are instructions for AI models
- Be specific about output format requirements
- Include examples that match your domain
- Provide context about what's important in your domain

### 3. Data Sources
- Update `data_sources.default_path` in config.yaml
- Ensure your articles are in Parquet format with required columns:
  - `title`, `content`, `url`, `published_date`

### 4. Output Paths
- Configure `output.directory` to organize results by domain
- Results will be saved to separate Parquet files for each entity type

## File Format Requirements

### Categories YAML Structure
```yaml
entity_types:
  type_name:
    description: "Clear description of this type"
    examples: ["Example 1", "Example 2"]
```

### Prompt Markdown Structure
```markdown
# Entity Type Extraction Prompt

Instructions for the AI model...

## Output Format
JSON format requirements...
```

### Config YAML Structure
```yaml
domain: "domain_name"
description: "Domain description"
data_sources:
  default_path: "path/to/articles.parquet"
output:
  directory: "data/entities/domain_name"
similarity_threshold: 0.75
```

## Troubleshooting

### Common Issues

1. **Prompt too generic**: Make prompts specific to your domain with relevant examples
2. **Categories too broad**: Create specific entity types that make sense for your domain
3. **Poor relevance filtering**: Update relevance criteria to be more specific
4. **Data path errors**: Ensure paths in config.yaml match your actual data location

### Testing Your Configuration

1. Start with a small number of articles:
   ```bash
   ./run.py process --domain football --limit 2
   # OR:
   just process-domain football --limit 2
   ```
2. Use verbose mode to see extraction details:
   ```bash
   ./run.py process --domain football --limit 2 --verbose
   # OR:
   just process-domain football --limit 2 --verbose
   ```
3. Check the output files in your configured directory
4. Iterate on prompts and categories based on results

### Available Commands

**Using run.py:**
- `./run.py domains` - List available domains
- `./run.py init <domain>` - Create new domain
- `./run.py process --domain <domain>` - Process articles

**Using justfile:**
- `just domains` - List available domains  
- `just init <domain>` - Create new domain
- `just process-domain <domain>` - Process articles for domain

## Advanced Configuration

### Custom Processing Settings
```yaml
processing:
  relevance_check: true      # Filter irrelevant articles
  batch_size: 5             # Articles to process at once
  
similarity_threshold: 0.75   # Entity deduplication threshold (0.0-1.0)
```

### Multiple Data Sources
```yaml
data_sources:
  default_path: "data/raw_sources/main_articles.parquet"
  secondary_path: "data/raw_sources/backup_articles.parquet"
```

## Getting Help

- Check existing domain configurations for examples
- Review the template files for structure guidance
- Run `./run.py --help` for CLI options
- Use `--verbose` flag to debug extraction issues