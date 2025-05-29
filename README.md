# Hinbox

Hinbox is a flexible, domain-configurable entity extraction system designed for historians and researchers. It processes historical documents, academic papers, news articles, and book chapters to extract structured information about people, organizations, locations, and events. Originally designed for Guantánamo Bay coverage analysis, it now supports any historical or research domain through a simple configuration system.

## 🎯 Key Features

- **Research-Focused**: Designed for historians, academics, and researchers
- **Flexible Sources**: Process historical documents, academic papers, news articles, book chapters
- **Domain-Agnostic**: Configure for any historical period, region, or research topic
- **Multiple AI Models**: Support for both cloud (Gemini default, but supports
  anything that `litellm` supports) and local (Ollama default, but works with
  `litellm`) models  
- **Entity Extraction**: Automatically extract people, organizations, locations, and events
- **Smart Deduplication**: Uses embeddings to merge similar entities across sources
- **Web Interface**: FastHTML-based UI for exploring research findings
- **Easy Setup**: Simple configuration files, no Python coding required

## 🚀 Quick Start

> **Note**: This project supports both `./run.py` commands and `just` commands. Use whichever you prefer!

### 1. List Available Domains
```bash
./run.py domains
# OR: just domains
```

### 2. Create a New Research Domain
```bash
./run.py init palestine_food_history
# OR: just init afghanistan_1980s
```

### 3. Configure Your Research Domain
Edit the generated files in `configs/palestine_food_history/`:
- `config.yaml` - Research domain settings and data paths
- `prompts/*.md` - Extraction instructions tailored to your sources
- `categories/*.yaml` - Entity type definitions relevant to your research

### 4. Process Your Sources
```bash
./run.py process --domain palestine_food_history --limit 5
# OR: just process-domain afghanistan_1980s --limit 5
```

### 5. Explore Results
```bash
./run.py frontend
# OR: just frontend
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- uv (for dependency management)
- Optional: Ollama (for local model support)
- Optional: just (for easier command running)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/hinbox.git
   cd hinbox
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   # Optional for local processing:
   export OLLAMA_API_URL="http://localhost:11434/v1"
   ```

4. **Verify installation:**
   ```bash
   ./run.py domains
   ```

## 📚 Research Domain Examples

### History of Food in Palestine
```bash
just init palestine_food_history
# Edit configs/palestine_food_history/ to focus on:
# - People: farmers, traders, cookbook authors, anthropologists
# - Organizations: agricultural cooperatives, food companies, research institutions
# - Events: harvests, famines, recipe documentation, cultural exchanges
# - Locations: villages, markets, agricultural regions, refugee camps
```

### Soviet-Afghan War (1980s)
```bash
just init afghanistan_1980s
# Configure for:
# - People: military leaders, diplomats, journalists, mujahideen commanders
# - Organizations: military units, intelligence agencies, NGOs, tribal groups
# - Events: battles, negotiations, refugee movements, arms shipments
# - Locations: provinces, military bases, refugee camps, border crossings
```

### Medieval Trade Networks
```bash
just init medieval_trade
# Set up for:
# - People: merchants, rulers, scholars, travelers
# - Organizations: trading companies, guilds, monasteries, courts
# - Events: trade agreements, diplomatic missions, market fairs
# - Locations: trading posts, cities, trade routes, ports
```

## 🛠 Advanced Usage

### Processing Historical Sources
```bash
# Process with different options
./run.py process --domain afghanistan_1980s -n 20 --verbose
just process-domain palestine_food_history --limit 10 --relevance

# Use local models (requires Ollama) - useful for sensitive historical research
./run.py process --domain medieval_trade --local

# Force reprocessing when you update your configuration
./run.py process --domain afghanistan_1980s --force
```

### Web Interface
```bash
./run.py frontend
# OR: just frontend
```
Explore extracted entities at http://localhost:5001

### Data Management
```bash
# Check processing status
./run.py check

# Reset processing status
./run.py reset

# View available domains
./run.py domains
```

## 📂 Project Structure

```
configs/
├── guantanamo/              # Example: Guantánamo Bay research
├── palestine_food_history/  # Your historical research domain
├── template/                # Template for new research domains
└── README.md               # Configuration guide

src/
├── config_loader.py    # Domain configuration system
├── dynamic_models.py   # Dynamic Pydantic model generation
├── people.py          # People extraction
├── organizations.py   # Organization extraction
├── locations.py       # Location extraction
├── events.py         # Event extraction
└── frontend/         # Web interface

data/
├── guantanamo/              # Guantánamo research data
│   ├── raw_sources/         # Historical documents, articles
│   └── entities/           # Extracted entities
└── {research_domain}/      # Each research domain has its own directory
    ├── raw_sources/        # Books, papers, articles, documents
    └── entities/          # Research findings
```

## 🔧 Configuration

### Domain Configuration
Each domain has its own `configs/{domain}/` directory with:

**config.yaml** - Main settings:
```yaml
domain: "palestine_food_history"
description: "Historical analysis of Palestinian food culture and agriculture"
data_sources:
  default_path: "data/palestine_food_history/raw_sources/historical_sources.parquet"
output:
  directory: "data/palestine_food_history/entities"
```

**categories/*.yaml** - Entity type definitions:
```yaml
person_types:
  player:
    description: "Professional football players"
    examples: ["Lionel Messi", "Cristiano Ronaldo"]
```

**prompts/*.md** - Extraction instructions (plain English!):
```markdown
You are an expert at extracting people from historical documents about Palestinian food culture.
Focus on farmers, traders, cookbook authors, researchers, and community leaders...
```

### Data Format
Historical sources should be in Parquet format with columns:
- `title`: Document/article title
- `content`: Full text content  
- `url`: Source URL (if applicable)
- `published_date`: Publication/creation date
- `source_type`: "book_chapter", "journal_article", "news_article", "archival_document", etc.

## 🏗 Architecture

### Processing Pipeline
1. **Configuration Loading**: Read domain-specific settings
2. **Source Loading**: Process historical documents in Parquet format
3. **Relevance Filtering**: Domain-specific content filtering for research focus
4. **Entity Extraction**: Extract people, organizations, locations, events from historical sources
5. **Smart Deduplication**: Merge similar entities using embeddings
6. **Profile Generation**: Create comprehensive entity profiles

### Key Features
- **Domain-Agnostic**: Easy to configure for any topic
- **Multiple AI Models**: Cloud (Gemini) and local (Ollama) support
- **Smart Processing**: Automatic relevance filtering and deduplication
- **Modern Interface**: FastHTML-based web UI
- **Robust Pipeline**: Error handling and progress tracking

## Development

### Code Quality
```bash
# Format code
./scripts/format.sh

# Run linting
./scripts/lint.sh

# Both together
just check-code
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New domain templates and examples
- Additional language model integrations
- Enhanced web interface features
- Performance optimizations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙋 Support

For questions about:
- **Configuration**: See `configs/README.md`
- **Setup**: Check installation steps above
- **Usage**: Try `./run.py --help` or `just --list`
- **Issues**: Open a GitHub issue

---

**Built for**: Historians, researchers, and academics working with large document collections

**Built with**: Python, Pydantic, FastHTML, LiteLLM, Jina Embeddings
