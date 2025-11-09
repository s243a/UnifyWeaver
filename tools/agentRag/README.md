# AgentRag - Retrieval Services for Economic Agent Workflows

This directory contains the core implementation of the AgentRag retrieval services, which provide semantic search and re-ranking capabilities for the Economic Agent playbooks.

## Overview

AgentRag is a multi-component retrieval system that supports the Economic Agent model by providing:

- **Gemini Flash Retrieval**: Cloud-based semantic search and re-ranking using Google's Gemini API
- **Local Embedding Service**: CPU/GPU-based local indexing using sentence-transformers
- **Code-Aware Retrieval**: Specialized retrieval for source code with syntax awareness
- **Distributed Retrieval**: Multi-shard parallel search across large codebases

## Core Components

### `src/agent_rag/core/`

- **`gemini_retriever.py`**: Flask service for Gemini Flash-based retrieval
  - Endpoints: `/gemini-flash-retrieve`, `/gemini-multi-retrieve`, `/analyze-context`
  - Used by: `proposals/llm_workflows/tools/gemini_reranker.sh`

- **`embedding_service.py`**: Local embedding generation and vector search
  - Provides free, local-compute alternative to cloud APIs

- **`code_aware_retriever.py`**: Code-specific semantic search
  - Handles programming language syntax and structure

- **`orchestrator.py`**: Coordinates multiple retrieval strategies
  - Implements the data pipeline model discussed in the Economic Agent philosophy

## Quick Start

### Installation

```bash
cd tools/agentRag
pip install -r requirements.txt
```

### Running the Gemini Retriever Service

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key-here"

# Start the Flask service
python -m agent_rag.core.gemini_retriever
```

The service will run on `http://localhost:8000` and be accessible to the Economic Agent playbook tools.

### Configuration

The services can be configured via:
- Environment variables (`GEMINI_API_KEY`, `RETRIEVER_API_KEY`)
- Configuration file: `src/agent_rag/config/__init__.py`

## Integration with Economic Agent Playbooks

The tools in `proposals/llm_workflows/tools/` use these services:

- `gemini_reranker.sh` → `gemini_retriever.py` (port 8000)
- `local_indexer.sh` → `embedding_service.py` (to be implemented)
- `gemini_full_retrieval.sh` → `gemini_retriever.py` (port 8000)

## Architecture

AgentRag implements a **data pipeline model** where specialized LLM components handle distinct stages:
- Retriever LLM → finds candidate chunks
- Combiner LLM → merges results from multiple sources
- Generator LLM → produces final output

This contrasts with the **Economic Agent model** where a single high-level LLM strategically orchestrates these components based on resource constraints.

## Status

This is a working implementation extracted from the standalone AgentRag project. Additional components (testing infrastructure, platform-specific scripts, documentation) may be added after review.

## License

See project root LICENSE file.
