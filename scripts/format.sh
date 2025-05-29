#!/usr/bin/env bash
set -x

# Initialize default source directories
default_src="data ."
# Initialize SRC as an empty string
SRC=""

# Process arguments
for arg in "$@"
do
    # If it's not a flag, treat it as a source directory
    # Append the argument to SRC, separated by space
    if [ -z "$SRC" ]; then
        SRC="$arg"
    else
        SRC="$SRC $arg"
    fi
done

# If no source directories were provided, use the default
if [ -z "$SRC" ]; then
    SRC="$default_src"
fi


# autoflake replacement: removes unused imports and variables
ruff check $SRC --select F401,F841 --fix --exclude "__init__.py" --isolated

# sorts imports
ruff check $SRC --select I --fix --ignore D
ruff format $SRC
