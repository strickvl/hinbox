# Research Domain Configuration Guide

This directory contains domain-specific configurations for the Hinbox entity extraction system designed for historical and academic research. Each research domain defines its own entity types, extraction prompts, and processing settings tailored to specific historical periods, regions, or research topics.

## Quick Start

> **Note**: This project supports both `./run.py` commands and `just` commands. Use whichever you prefer!

### Creating a New Research Domain

1. **Initialize a new research domain**:
   ```bash
   ./run.py init palestine_food_history
   # OR using justfile:
   just init afghanistan_1980s
   ```
   This copies the template to `configs/palestine_food_history/`

2. **Configure your research domain**:
   - Edit `configs/palestine_food_history/config.yaml` - Set research focus, description, and data paths
   - Edit `configs/palestine_food_history/categories/*.yaml` - Define entity types relevant to your historical research
   - Edit `configs/palestine_food_history/prompts/*.md` - Customize extraction prompts for your sources

3. **Process your historical sources**:
   ```bash
   ./run.py process --domain palestine_food_history
   # OR using justfile:
   just process-domain afghanistan_1980s
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
- `relevance.md` - How to determine source relevance to your research

## Example Research Domains

### Guantánamo Bay Research (`configs/guantanamo/`)
The original domain focusing on detention, legal proceedings, and human rights issues.

**Key entity types**:
- People: detainee, military, government, lawyer, journalist
- Organizations: military, intelligence, legal, humanitarian, advocacy
- Locations: detention_facility, military_base, country, city
- Events: detention, legal, military_operation, policy_change, protest

### Historical Food Studies
Example configuration for researching food history:

**Key entity types**:
- People: farmers, traders, cookbook_authors, anthropologists, community_leaders
- Organizations: agricultural_cooperatives, food_companies, research_institutions, markets
- Locations: farms, markets, kitchens, regions, trade_routes
- Events: harvests, famines, recipe_documentation, cultural_exchanges, trade_agreements

### Conflict Studies (Soviet-Afghan War)
Example for military/political history research:

**Key entity types**:
- People: military_leaders, diplomats, commanders, journalists, civilians
- Organizations: military_units, intelligence_agencies, tribal_groups, international_bodies
- Locations: provinces, military_bases, refugee_camps, strategic_locations
- Events: battles, negotiations, refugee_movements, diplomatic_meetings

### Template (`configs/template/`)
Starting point for new research domains with generic categories and prompts suitable for academic research.

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
- Ensure your historical sources are in Parquet format with required columns:
  - `title`: Document/source title
  - `content`: Full text content
  - `url`: Source URL (if applicable)
  - `published_date`: Publication/creation date
  - `source_type`: "book_chapter", "journal_article", "news_article", "archival_document", etc.

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

1. **Prompt too generic**: Make prompts specific to your historical period/region with relevant examples
2. **Categories too broad**: Create specific entity types that make sense for your research area
3. **Poor relevance filtering**: Update relevance criteria to focus on your specific research questions
4. **Data path errors**: Ensure paths in config.yaml match your actual historical source location
5. **Mixed source types**: Different document types (books vs. articles) may need different extraction approaches

### Testing Your Configuration

1. Start with a small number of sources:
   ```bash
   ./run.py process --domain palestine_food_history --limit 2
   # OR:
   just process-domain afghanistan_1980s --limit 2
   ```
2. Use verbose mode to see extraction details:
   ```bash
   ./run.py process --domain palestine_food_history --limit 2 --verbose
   # OR:
   just process-domain afghanistan_1980s --limit 2 --verbose
   ```
3. Check the output files in your configured directory
4. Iterate on prompts and categories based on results

### Available Commands

**Using run.py:**
- `./run.py domains` - List available domains
- `./run.py init <domain>` - Create new domain
- `./run.py process --domain <domain>` - Process historical sources

**Using justfile:**
- `just domains` - List available domains  
- `just init <domain>` - Create new domain
- `just process-domain <domain>` - Process historical sources for domain

## Advanced Configuration

### Custom Processing Settings
```yaml
processing:
  relevance_check: true      # Filter irrelevant sources
  batch_size: 5             # Sources to process at once
  
similarity_threshold: 0.75   # Entity deduplication threshold (0.0-1.0)
```

### Multiple Data Sources
```yaml
data_sources:
  default_path: "data/domain/raw_sources/primary_sources.parquet"
  secondary_path: "data/domain/raw_sources/supplementary_sources.parquet"
```

## Getting Help

- Check existing research domain configurations for examples
- Review the template files for structure guidance
- Run `./run.py --help` for CLI options
- Use `--verbose` flag to debug extraction issues
- Consider the historical context when defining entity categories
- Test with different source types (books, articles, archival documents) to refine prompts