#!/bin/bash

# Run ruff to fix import-related issues
echo "Running ruff check to fix import issues..."
ruff check src data scripts . --select I --fix --ignore D

# Run ruff formatter
echo "Running ruff formatter..."
ruff format src data scripts .

echo "Linting complete!" 
