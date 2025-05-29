# Hinbox

Hinbox is a flexible, domain-configurable entity extraction system that processes news articles and extracts structured information about people, organizations, locations, and events. Originally designed for Guantánamo Bay coverage analysis, it now supports any domain through a simple configuration system.

## 🎯 Key Features

- **Domain-Agnostic**: Configure for any topic (politics, sports, business, etc.)
- **Multiple AI Models**: Support for both cloud (Gemini) and local (Ollama) models  
- **Entity Extraction**: Automatically extract people, organizations, locations, and events
- **Smart Deduplication**: Uses embeddings to merge similar entities
- **Web Interface**: FastHTML-based UI for exploring results
- **Easy Setup**: Simple configuration files, no Python coding required

## 🚀 Quick Start

> **Note**: This project supports both `./run.py` commands and `just` commands. Use whichever you prefer!

### 1. List Available Domains
```bash
./run.py domains
# OR: just domains
```

### 2. Create a New Domain
```bash
./run.py init football
# OR: just init football
```

### 3. Configure Your Domain
Edit the generated files in `configs/football/`:
- `config.yaml` - Domain settings and data paths
- `prompts/*.md` - Extraction instructions (in plain English!)
- `categories/*.yaml` - Entity type definitions

### 4. Process Articles
```bash
./run.py process --domain football --limit 5
# OR: just process-domain football --limit 5
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

## 📁 Domain Examples

### Sports (Football)
```bash
just init football
# Edit configs/football/ to focus on:
# - People: players, coaches, referees
# - Organizations: teams, leagues, federations  
# - Events: matches, transfers, tournaments
# - Locations: stadiums, cities, countries
```

### Business
```bash
just init business
# Configure for:
# - People: executives, investors, analysts
# - Organizations: companies, funds, banks
# - Events: mergers, earnings, launches
# - Locations: headquarters, markets, regions
```

### Politics
```bash
just init politics
# Set up for:
# - People: politicians, officials, activists
# - Organizations: parties, governments, NGOs
# - Events: elections, policies, debates
# - Locations: capitals, districts, countries
```

## 🛠 Advanced Usage

### Processing Articles
```bash
# Process with different options
./run.py process --domain football -n 20 --verbose
just process-domain football --limit 10 --relevance

# Use local models (requires Ollama)
./run.py process --domain football --local

# Force reprocessing
./run.py process --domain football --force
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
├── guantanamo/          # Example: Guantánamo Bay domain
├── football/            # Your sports domain
├── template/            # Template for new domains
└── README.md           # Configuration guide

src/
├── config_loader.py    # Domain configuration system
├── dynamic_models.py   # Dynamic Pydantic model generation
├── people.py          # People extraction
├── organizations.py   # Organization extraction
├── locations.py       # Location extraction
├── events.py         # Event extraction
└── frontend/         # Web interface

data/
├── guantanamo/      # Guantánamo domain data
│   ├── raw_sources/ # Input articles (Parquet format)
│   └── entities/    # Extracted entities
└── {domain}/        # Each domain has its own directory
    ├── raw_sources/
    └── entities/
```

## 🔧 Configuration

### Domain Configuration
Each domain has its own `configs/{domain}/` directory with:

**config.yaml** - Main settings:
```yaml
domain: "football"
description: "Football news and analysis"
data_sources:
  default_path: "data/football/raw_sources/football_articles.parquet"
output:
  directory: "data/football/entities"
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
You are an expert at extracting people from football articles.
Focus on players, coaches, and officials...
```

### Data Format
Articles should be in Parquet format with columns:
- `title`, `content`, `url`, `published_date`

## 🏗 Architecture

### Processing Pipeline
1. **Configuration Loading**: Read domain-specific settings
2. **Article Loading**: Process Parquet files
3. **Relevance Filtering**: Domain-specific content filtering  
4. **Entity Extraction**: Extract people, organizations, locations, events
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

**Built with**: Python, Pydantic, FastHTML, LiteLLM, Jina Embeddings