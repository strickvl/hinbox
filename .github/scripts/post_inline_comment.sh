#!/bin/bash
set -e # Exit on error

# --- Parse Arguments ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --pr-number) PR_NUMBER="$2"; shift ;;
        --commit-id) COMMIT_ID="$2"; shift ;;
        --path) FILE_PATH="$2"; shift ;;
        --body) BODY="$2"; shift ;;
        --line) LINE="$2"; shift ;;
        --start-line) START_LINE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# --- Validate Required Arguments ---
if [ -z "$PR_NUMBER" ] || [ -z "$COMMIT_ID" ] || [ -z "$FILE_PATH" ] || [ -z "$BODY" ] || [ -z "$LINE" ]; then
    echo "Error: Missing required arguments: --pr-number, --commit-id, --path, --body, and --line are required."
    exit 1
fi

# --- Construct gh api command ---
ARGS=(
    "--method" "POST"
    "-H" "Accept: application/vnd.github+json"
    "/repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/comments"
    "-f" "body=${BODY}"
    "-f" "commit_id=${COMMIT_ID}"
    "-f" "path=${FILE_PATH}"
    "-F" "line=${LINE}"
    "-f" "side=RIGHT"
)

# Add parameters for multi-line comments if start_line is provided
if [ -n "$START_LINE" ]; then
    ARGS+=("-F" "start_line=${START_LINE}" "-f" "start_side=RIGHT")
fi

# --- Execute Command ---
echo "Posting inline comment to ${FILE_PATH} at line ${LINE}..."
gh api "${ARGS[@]}"

echo "Successfully posted inline comment."
