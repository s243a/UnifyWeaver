#!/bin/bash
# =================================================================
# Tool: gemini_reranker
# Description: Re-ranks text snippets by relevance using Gemini Flash
# =================================================================
#
# LLM Instructions:
# This tool takes a search query and a list of text chunks, then uses
# the Gemini Flash API to re-rank them by relevance. It returns the
# top-k most relevant chunks with confidence scores and reasoning.
#
# Usage:
#   gemini_reranker --query "search query" --chunks chunk1.txt chunk2.txt --top-k 10
#
# Input:
#   --query: The search query string
#   --chunks: Space-separated list of files containing text chunks
#   --top-k: Number of top results to return (default: 10)
#
# Output:
#   JSON array: [{"chunk": "text", "confidence": 0.0-1.0, "reasoning": "explanation"}]
#
# Resource Metrics:
#   Cost: {{ GEMINI_FLASH_COST | default: 0.002 }} USD per call
#   Latency: {{ GEMINI_RERANKER_LATENCY_MS | default: 2500 }} ms
#
# Error Codes:
#   0 - Success
#   3 - API quota exceeded
#   4 - Network error
#   5 - Invalid input
# =================================================================

set -e

# Default values
TOP_K=10
QUERY=""
CHUNKS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)
            QUERY="$2"
            shift 2
            ;;
        --chunks)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                CHUNKS+=("$1")
                shift
            done
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
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

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
    echo "Error: --chunks is required" >&2
    exit 5
fi

# Build JSON payload for chunks
CHUNKS_JSON="["
for i in "${!CHUNKS[@]}"; do
    CHUNK_TEXT=$(cat "${CHUNKS[$i]}" | jq -Rs .)
    if [[ $i -gt 0 ]]; then
        CHUNKS_JSON+=","
    fi
    CHUNKS_JSON+="$CHUNK_TEXT"
done
CHUNKS_JSON+="]"

# Call the Gemini retriever service
# Note: This assumes the gemini_retriever service is running on localhost:8000
# In production, this would be configurable via environment variable
RESPONSE=$(curl -s -X POST http://localhost:8000/gemini-flash-retrieve \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": $(echo "$QUERY" | jq -Rs .),
        \"shard_name\": \"combined\",
        \"shard_docs\": $CHUNKS_JSON
    }" 2>&1)

# Check for network errors
if [[ $? -ne 0 ]]; then
    echo "Error: Network error connecting to Gemini service" >&2
    exit 4
fi

# Check for API quota errors
if echo "$RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error')
    if [[ "$ERROR_MSG" =~ "quota" ]] || [[ "$ERROR_MSG" =~ "rate limit" ]]; then
        echo "$RESPONSE" >&2
        exit 3
    else
        echo "$RESPONSE" >&2
        exit 4
    fi
fi

# Extract and format the results
echo "$RESPONSE" | jq -c ".context[:$TOP_K] | map({chunk: ., confidence: .confidence // 0.5})"
