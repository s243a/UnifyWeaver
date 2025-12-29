---
description: File a bookmark into Pearltrees using semantic search
---

# File Bookmark Slash Command

## Usage
`/file-bookmark "Your bookmark title here"`

## Steps

1. **Get semantic candidates** with tree output:
   ```bash
   python3 scripts/infer_pearltrees_federated.py \
     --model models/pearltrees_federated_single.pkl \
     --query "{bookmark_title}" \
     --top-k 10 --tree
   ```

2. **Analyze the tree output**:
   - Candidates are marked with ★ 
   - Format: `★ #rank [score]`
   - Higher scores = more semantically similar

3. **Recommend the best folder** based on:
   - Specificity match (specific topic vs general category)
   - Hierarchical context (parent vs child folder)
   - Account context (personal s243a vs shared s243a_groups)

4. **Present recommendation** with:
   - Selected folder name
   - Rank and score
   - Brief reasoning

## Example

User: `/file-bookmark "Introduction to Transformer models in NLP"`

Run the semantic search, then respond with something like:

> **Recommended folder: Deep Learning ★ #2 [0.328]**
> 
> Reasoning: Transformers are a deep learning architecture. While "Neural network architectures" ranked #1, 
> "Deep Learning" is more specific to this tutorial-style content.
> 
> Alternative: "Transformers" (#3) if you have a dedicated subfolder.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `--tree` | Merged hierarchical view |
| `--json` | Structured JSON output |
| `--top-k N` | Number of candidates |
