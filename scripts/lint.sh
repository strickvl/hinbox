#!/bin/bash

# Run ruff linter (uses pyproject.toml config)
echo "Running ruff check..."
ruff check src data scripts . --fix

# Run ruff formatter
echo "Running ruff formatter..."
ruff format src data scripts .

echo "Linting complete!"
