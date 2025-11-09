#!/bin/bash
# =================================================================
# Tool: local_indexer
# Description: Indexes and searches files using local embeddings
# =================================================================
#
# LLM Instructions:
# This tool creates local embeddings for text files using sentence-transformers
# and performs semantic search. It's a free, fast alternative to cloud-based
# retrieval that works entirely on local compute.
#
# Usage:
#   local_indexer --query "search query" --files file1.txt file2.txt --top-k 25
#
# Input:
#   --query: The search query string
#   --files: Space-separated list of files to index and search
#   --top-k: Number of top results to return (default: 25)
#   --reindex: Force re-indexing even if cache exists
#
# Output:
#   JSON array: [{"file": "path", "chunk": "text", "score": 0.0-1.0}]
#
# Resource Metrics:
#   Cost: {{ LOCAL_INDEXER_COST | default: 0.0 }} USD (free, local compute)
#   Latency: {{ LOCAL_INDEXER_LATENCY_MS | default: 500 }} ms per file
#
# Error Codes:
#   0 - Success
#   1 - No files found
#   2 - Index missing or corrupted
#   5 - Invalid input
# =================================================================

set -e

# Default values
TOP_K=25
QUERY=""
FILES=()
REINDEX=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)
            QUERY="$2"
            shift 2
            ;;
        --files)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                FILES+=("$1")
                shift
            done
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --reindex)
            REINDEX=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 5
            ;;
    esac
done

# Validate inputs
if [[ -z "$QUERY" ]]; then
    echo "Error: --query is required" >&2
    exit 5
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: --files is required" >&2
    exit 5
fi

# Check if files exist
for file in "${FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: File not found: $file" >&2
        exit 1
    fi
done

# Stub implementation: In production, this would call a Python script
# that uses sentence-transformers to create embeddings and search
#
# For now, we'll simulate the output with a simple grep-based scoring
# This demonstrates the interface without requiring the full embedding infrastructure

RESULTS="["
FIRST=true

for file in "${FILES[@]}"; do
    # Simple scoring based on keyword matches (stub logic)
    MATCHES=$(grep -i -o -E "\w*${QUERY}\w*" "$file" 2>/dev/null | wc -l || echo "0")

    if [[ $MATCHES -gt 0 ]]; then
        # Extract a sample chunk around the first match
        CHUNK=$(grep -i -C 2 "$QUERY" "$file" 2>/dev/null | head -n 5 | tr '\n' ' ' || echo "")

        # Normalize score to 0.0-1.0 range (stub logic)
        SCORE=$(echo "scale=2; if ($MATCHES > 10) 0.95 else $MATCHES / 10" | bc)

        if [[ "$FIRST" == "false" ]]; then
            RESULTS+=","
        fi
        FIRST=false

        RESULTS+=$(jq -n \
            --arg file "$file" \
            --arg chunk "$CHUNK" \
            --arg score "$SCORE" \
            '{file: $file, chunk: $chunk, score: ($score | tonumber)}')
    fi
done

RESULTS+="]"

# Sort by score and return top-k
echo "$RESULTS" | jq -c "sort_by(-.score) | .[:$TOP_K]"
