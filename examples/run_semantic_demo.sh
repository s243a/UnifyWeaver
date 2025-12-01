#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "1. Generating Python scripts..."
swipl -g main -t halt examples/semantic_playbook.pl

echo "2. Running Indexer (Parsing & Embedding)..."
# Check if model exists
if [ ! -f models/model.onnx ]; then
    echo "Model not found. Please run 'curl ...' (see previous steps) or expect failure."
    # Attempt download if missing?
    mkdir -p models
    curl -L -o models/model.onnx https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
    curl -L -o models/vocab.txt https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
fi

echo "{}" | python3 run_index.py

echo "3. Running Search..."
echo "{}" | python3 run_search.py | head -n 5

echo "Done."
