#!/bin/bash
# =================================================================
# Tool: gemini_full_retrieval
# Description: Full-document semantic search using Gemini Flash
# =================================================================
#
# LLM Instructions:
# This tool performs high-accuracy semantic search on a single large document
# using Gemini Flash. It's more expensive than local indexing but provides
# better semantic understanding and can handle complex queries.
#
# Usage:
#   gemini_full_retrieval --query "search query" --file large_doc.txt --top-k 5
#
# Input:
#   --query: The search query string
#   --file: Path to the document to search
#   --top-k: Number of top results to return (default: 5)
#
# Output:
#   JSON array: [{"chunk": "text", "confidence": 0.0-1.0}]
#
# Resource Metrics:
#   Cost: {{ GEMINI_FLASH_COST | default: 0.002 }} USD per call
#   Latency: {{ GEMINI_FULL_RETRIEVAL_LATENCY_MS | default: 3000 }} ms
#
# Error Codes:
#   0 - Success
#   3 - API quota exceeded
#   4 - Network error
#   5 - Invalid input / file too large
# =================================================================

set -e

# Default values
TOP_K=5
QUERY=""
FILE=""
MAX_FILE_SIZE=1000000  # 1MB limit for single file retrieval

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query)
            QUERY="$2"
            shift 2
            ;;
        --file)
            FILE="$2"
            shift 2
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

if [[ -z "$FILE" ]]; then
    echo "Error: --file is required" >&2
    exit 5
fi

if [[ ! -f "$FILE" ]]; then
    echo "Error: File not found: $FILE" >&2
    exit 5
fi

# Check file size
FILE_SIZE=$(wc -c < "$FILE")
if [[ $FILE_SIZE -gt $MAX_FILE_SIZE ]]; then
    echo "Error: File too large ($FILE_SIZE bytes, max $MAX_FILE_SIZE)" >&2
    exit 5
fi

# Read file content and chunk it (simple line-based chunking for demo)
# In production, this would use smarter chunking logic
FILE_CONTENT=$(cat "$FILE")

# Call Gemini retriever service
RESPONSE=$(curl -s -X POST http://localhost:8000/gemini-full-retrieval \
    -H "Content-Type: application/json" \
    -d "{
        \"query\": $(echo "$QUERY" | jq -Rs .),
        \"file_content\": $(echo "$FILE_CONTENT" | jq -Rs .),
        \"top_k\": $TOP_K
    }" 2>&1)

# Check for errors
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

# Return results
echo "$RESPONSE" | jq -c ".context[:$TOP_K]"
