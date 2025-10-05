#!/bin/bash
set -e

# --- Initialize variables ---
PR_NUMBER=""
COMMIT_ID=""
FILE_PATH=""
BODY=""
LINE=""
START_LINE=""

# --- Parse command-line arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pr-number) PR_NUMBER="$2"; shift ;;
        --commit-id) COMMIT_ID="$2"; shift ;;
        --path) FILE_PATH="$2"; shift ;;
        --line) LINE="$2"; shift ;;
        --start-line) START_LINE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
    esac
    shift
done

# --- Read the comment body from standard input ---
BODY=$(cat)

# --- Validate required arguments ---
if [ -z "$PR_NUMBER" ] || [ -z "$COMMIT_ID" ] || [ -z "$FILE_PATH" ] || [ -z "$BODY" ] || [ -z "$LINE" ]; then
    echo "Error: Missing required arguments. PR_NUMBER, COMMIT_ID, FILE_PATH, BODY (via stdin), and LINE are required." >&2
    exit 1
fi

# --- Construct the gh api command arguments ---
ARGS=(
    "--method" "POST"
    "-H" "Accept: application/vnd.github+json"
    "/repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/comments"
    "-f" "commit_id=${COMMIT_ID}"
    "-f" "path=${FILE_PATH}"
    "-F" "line=${LINE}"
    "-f" "side=RIGHT"
)

# Add parameters for multi-line comments if start_line is provided
if [ -n "$START_LINE" ]; then
    ARGS+=("-F" "start_line=${START_LINE}" "-f" "start_side=RIGHT")
fi

# --- Execute the command, passing the body from stdin ---
echo "Posting inline comment to ${FILE_PATH} on PR #${PR_NUMBER}..."
echo "$BODY" | gh api "${ARGS[@]}" --input -

echo "Successfully posted inline comment."
