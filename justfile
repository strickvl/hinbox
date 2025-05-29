# Hinbox task runner

# Default command - show available commands
default:
    @just --list

# Process articles and extract entities (default: guantanamo domain)
process *args:
    python -m src.process_and_extract {{args}}

# Start the web interface
frontend:
    @echo "Starting web interface..."
    @echo "Open http://localhost:5001 in your browser"
    python src/frontend/frontend.py

# Alias for frontend
web: frontend

# Alias for frontend  
ui: frontend

# Check article database statistics
check *args:
    python scripts/check_articles_parquet.py {{args}}

# Reset processing status of all articles
reset:
    #!/usr/bin/env bash
    read -p "This will reset the processing status of ALL articles. Are you sure? (y/N): " response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        python scripts/reset_processing_status.py
    else
        echo "Reset cancelled."
    fi

# Fetch Miami Herald articles
fetch-miami:
    python scripts/get_miami_herald_articles.py

# Import Miami Herald articles
import-miami:
    python scripts/import_miami_herald_articles.py

# Format code
format:
    ./scripts/format.sh

# Run linting
lint:
    ./scripts/lint.sh

# Run both format and lint
check-code: format lint

# Show article statistics
stats: check

# Process a single article (useful for testing)
process-one:
    python -m src.process_and_extract --limit 1

# Process articles with verbose output
process-verbose:
    python -m src.process_and_extract -v

# Quick test run - process one article with verbose output
test-run:
    python -m src.process_and_extract --limit 1 -v

# Initialize a new domain configuration
init domain:
    python run.py init {{domain}}

# List available domain configurations
domains:
    python run.py domains

# Process articles for a specific domain
process-domain domain *args:
    python -m src.process_and_extract --domain {{domain}} {{args}}