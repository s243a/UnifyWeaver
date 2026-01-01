# Chapter 16: Bookmark Filing Assistant

This chapter covers the bookmark filing assistant that combines semantic search with LLM reasoning to recommend folders for new bookmarks.

## Overview

The bookmark filing pipeline:

```
New Bookmark → Semantic Search → Top-K Candidates → LLM Selection → Filing
             (93% Recall@1)     (Merged Tree)      (Final Choice)
```

## Architecture

### Federated Model

The semantic search uses a federated projection model:
- **51 clusters** of semantically similar folders
- **1 shared W matrix per cluster** (not per-query)
- **Query-level routing** to find similar training queries
- **160MB total** model size

See [Model Format Documentation](../../docs/design/FEDERATED_MODEL_FORMAT.md) for detailed file structure.

### LLM Integration

Multiple LLM backends supported:

| Provider | Command | Notes |
|----------|---------|-------|
| Claude CLI | `--provider claude` | Cheapest with subscription |
| Gemini CLI | `--provider gemini` | Gemini headless |
| OpenAI API | `--provider openai` | Requires OPENAI_API_KEY |
| Anthropic API | `--provider anthropic` | Requires ANTHROPIC_API_KEY |
| Ollama | `--provider ollama` | Local models |

## Usage

### Quick Semantic Search
```bash
python3 scripts/infer_pearltrees_federated.py \
  --model models/pearltrees_federated_single.pkl \
  --query "Your bookmark title" \
  --top-k 10 --tree
```

Output:
```
└── s243a
    └── s243a_groups @s243a_groups
        └── s243a
            └── STEM
                └── AI & Machine Learning ★ #10 [0.280]
                    ├── Deep Learning ★ #2 [0.328]
                    └── Neural network architectures ★ #1 [0.328]
```

### Full LLM-Assisted Filing
```bash
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "Neural network tutorial" \
  --provider claude \
  --top-k 10
```

### Slash Command
```
/file-bookmark "Your bookmark title"
```

### Agent Launcher
```bash
./scripts/launch_bookmark_filing_agent.sh "Optional bookmark title"
```

## Fuzzy Boost

Fuzzy logic boosting allows fine-tuning candidate rankings with term matching:

### CLI Options
```bash
python3 scripts/bookmark_filing_assistant.py \
  --bookmark "bash reduce function" \
  --boost-or "bash:0.9,shell:0.5,scripting:0.3" \
  --filter "in_subtree:Unix" \
  --blend-alpha 0.8
```

| Option | Description |
|--------|-------------|
| `--boost-and` | AND boost - all terms must match (product t-norm) |
| `--boost-or` | OR boost - any term can match (distributed OR) |
| `--filter` | Filter by predicate (e.g., `in_subtree:Unix`, `is_type:tree`) |
| `--blend-alpha` | Blend weight 0-1 (default 0.7, higher = more boost influence) |

### Interactive Commands
In interactive mode (`--interactive`):
- `boost bash:0.9,shell:0.5` - Set OR boost
- `boost-and python:0.9` - Set AND boost
- `filter in_subtree:Unix` - Add filter
- `clear` - Clear all boosts/filters
- `status` - Show current settings

### API Mode
```bash
# Start REST API server
python3 scripts/api_bookmark_filing_server.py --port 5000

# Query with fuzzy boost
curl -X POST http://localhost:5000/api/candidates \
  -H "Content-Type: application/json" \
  -d '{"bookmark_title": "bash tutorial", "boost_or": "bash:0.9"}'
```

## MCP Integration

An MCP server exposes these tools:

```python
# Start MCP server
python3 scripts/mcp_bookmark_filing_server.py
```

Tools exposed:
- `get_filing_candidates` - Semantic search candidates (with fuzzy boost support)
- `get_dual_objective_candidates` - Dual-objective scoring
- `file_bookmark` - Full LLM recommendation (with fuzzy boost support)

All tools support fuzzy boost parameters: `boost_and`, `boost_or`, `filters`, `blend_alpha`.

## Decision Process

The LLM sees the merged tree and considers:

1. **Specificity match** - File in most specific matching folder
2. **Hierarchical context** - Parent vs child folder appropriateness
3. **Account boundaries** - Personal (s243a) vs shared (s243a_groups)
4. **Score proximity** - Close scores may warrant parent folder

## Performance

| Metric | Value |
|--------|-------|
| Semantic Recall@1 | 93% |
| Semantic Recall@5 | 99% |
| Model Size | 160MB |
| Inference Time | ~100ms |
| LLM Call | ~2-5s |

## Files

| File | Purpose |
|------|---------|
| `scripts/infer_pearltrees_federated.py` | Semantic search |
| `scripts/bookmark_filing_assistant.py` | LLM-assisted filing (CLI + interactive) |
| `scripts/mcp_bookmark_filing_server.py` | MCP server |
| `scripts/api_bookmark_filing_server.py` | REST API server |
| `scripts/fuzzy_boost.py` | Fuzzy logic boost module |
| `scripts/launch_bookmark_filing_agent.sh` | Claude launcher |
| `skills/skill_bookmark_filing.md` | Skill documentation |
| `docs/ai-skills/bookmark-filing-agent.md` | Agent role |

## Example Session

User wants to file: "Introduction to Transformer models in NLP"

Semantic search returns:
```
└── AI & Machine Learning
    ├── Machine Learning ★ #3 [0.318]
    │   └── Deep Learning ★ #1 [0.328]
    │       └── Transformers ★ #2 [0.325]
    └── NLP ★ #4 [0.310]
```

LLM reasoning:
> "Deep Learning (#1) is recommended because Transformers are a deep learning architecture.
> The 'Transformers' subfolder (#2) would also be excellent for more specific organization.
> NLP (#4) is too broad for this specific architecture tutorial."

## Next Steps

- Integration with browser extension for auto-filing
- Batch filing for bookmark imports
- Learning from user corrections
