#!/bin/bash
# =================================================================
# Tool: git_status
# Description: Check for modified files in git repository
# =================================================================
#
# LLM Instructions:
# This tool checks which files have been modified in the git repository
# since the last commit. It's used to determine whether a file index
# needs to be updated before performing a search.
#
# Usage:
#   git_status [--path /path/to/repo]
#
# Input:
#   --path: Path to git repository (default: current directory)
#
# Output:
#   JSON array: [{"file": "path/to/file", "status": "M|A|D"}]
#   Status codes: M=modified, A=added, D=deleted
#
# Resource Metrics:
#   Cost: {{ GIT_STATUS_COST | default: 0.0 }} USD (free)
#   Latency: {{ GIT_STATUS_LATENCY_MS | default: 50 }} ms
#
# Error Codes:
#   0 - Success
#   128 - Not a git repository
# =================================================================

set -e

# Default values
REPO_PATH="."

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path)
            REPO_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Change to repo directory
cd "$REPO_PATH" 2>/dev/null || {
    echo "Error: Directory not found: $REPO_PATH" >&2
    exit 1
}

# Check if it's a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository: $REPO_PATH" >&2
    exit 128
fi

# Get status in porcelain format
STATUS_OUTPUT=$(git status --porcelain)

# Parse and convert to JSON
RESULTS="["
FIRST=true

while IFS= read -r line; do
    if [[ -z "$line" ]]; then
        continue
    fi

    # Extract status code and file path
    STATUS="${line:0:2}"
    FILE="${line:3}"

    # Normalize status code
    case "${STATUS:0:1}" in
        M|A|D|R|C)
            STATUS_CODE="${STATUS:0:1}"
            ;;
        *)
            STATUS_CODE="M"
            ;;
    esac

    if [[ "$FIRST" == "false" ]]; then
        RESULTS+=","
    fi
    FIRST=false

    RESULTS+=$(jq -n \
        --arg file "$FILE" \
        --arg status "$STATUS_CODE" \
        '{file: $file, status: $status}')

done <<< "$STATUS_OUTPUT"

RESULTS+="]"

echo "$RESULTS"
