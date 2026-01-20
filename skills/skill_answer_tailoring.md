# Skill: Answer Tailoring

Reword and tailor Q&A answers using LLMs to create natural variation while preserving semantic equivalence.

## When to Use

- User asks "how do I create answer variations?"
- User wants to augment training data with reworded answers
- User needs to tailor answers to specific questions
- User asks about generate_tailored_answers.py

## Overview

Answer tailoring takes existing (question, base_answer) pairs and generates reworded answers that:

1. **Directly address** the specific question
2. **Maintain semantic equivalence** with the base answer
3. **Have natural variation** in phrasing

This creates diverse training data from a single base answer.

## Quick Start

```bash
# Process all files in expanded directory
python scripts/generate_tailored_answers.py \
  --input training-data/expanded

# Process single file
python scripts/generate_tailored_answers.py \
  --file training-data/expanded/mindmap/pairs.jsonl

# Use different provider/model
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --provider gemini \
  --model gemini-2.5-flash-preview
```

## Command Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | training-data/expanded | Input directory with JSONL files |
| `--output` | input/tailored/ | Output directory |
| `--file` | - | Process single file instead of all |
| `--provider` | claude | LLM provider (claude, gemini) |
| `--model` | sonnet | Model to use |
| `--batch-size` | 10 | Pairs per batch |
| `--delay` | 0.5 | Seconds between API calls |
| `--no-skip` | false | Don't skip already processed pairs |

## How It Works

### Input Format

Expects JSONL with question/answer pairs:

```json
{
  "pair_id": "mindmap_001_q1",
  "cluster_id": "mindmap_001",
  "question": "How do I organize my mindmaps into folders?",
  "answer": "Use MST folder grouping to create semantic hierarchies..."
}
```

### Processing

1. For each (question, answer) pair:
2. Send to LLM with tailoring prompt
3. LLM rewrites answer to directly address the question
4. Preserve original answer as `original_answer`
5. Mark record with `answer_tailored: true`

### Output Format

```json
{
  "pair_id": "mindmap_001_q1",
  "cluster_id": "mindmap_001",
  "question": "How do I organize my mindmaps into folders?",
  "answer": "To organize your mindmaps into folders, use the MST folder grouping tool...",
  "original_answer": "Use MST folder grouping to create semantic hierarchies...",
  "answer_tailored": true
}
```

## Resumable Processing

The tool supports resumable processing:

- Checks output file for already-processed `pair_id` values
- Skips pairs with `answer_tailored: true`
- Saves progress after each batch

To reprocess everything:

```bash
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --no-skip
```

## LLM Prompt

The tailoring prompt instructs the LLM to:

```
Given a question and a base answer, rewrite the answer to:
1. Directly address the specific question (start by addressing what they asked)
2. Keep all the essential technical information from the base answer
3. Use slightly different wording/phrasing for diversity
4. Be concise but complete
5. If mentioning where to find more info, reference specific paths
```

## Batch Processing

For large datasets, adjust batch size and delay:

```bash
# Smaller batches, longer delay (rate limit safe)
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --batch-size 5 \
  --delay 1.0

# Larger batches, faster (if rate limits allow)
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --batch-size 20 \
  --delay 0.2
```

## Provider Comparison

| Provider | Model | Speed | Quality | Cost |
|----------|-------|-------|---------|------|
| Claude | haiku | Fast | Good | Low |
| Claude | sonnet | Medium | Better | Medium |
| Claude | opus | Slow | Best | High |
| Gemini | flash | Fast | Good | Low |

Recommendation: Use `haiku` or `gemini flash` for bulk processing, `sonnet` for quality-sensitive data.

## Workflow Example

Complete answer tailoring workflow:

```bash
# 1. Start with base Q&A data
ls training-data/expanded/mindmap/

# 2. Generate tailored versions
python scripts/generate_tailored_answers.py \
  --input training-data/expanded/mindmap \
  --output training-data/tailored/mindmap

# 3. Check results
head -1 training-data/tailored/mindmap/pairs.jsonl | jq .

# 4. Verify tailoring
grep '"answer_tailored": true' training-data/tailored/mindmap/pairs.jsonl | wc -l
```

## Troubleshooting

### Timeout errors

Increase timeout or reduce batch size:

```bash
python scripts/generate_tailored_answers.py \
  --batch-size 5 \
  --delay 1.0
```

### Rate limiting

Add longer delays between calls:

```bash
python scripts/generate_tailored_answers.py --delay 2.0
```

### Poor quality rewrites

- Try a better model (sonnet instead of haiku)
- Check that base answers have enough content
- Review the source_file field for context

### Resuming interrupted run

Just run the same command again - it will skip already-processed pairs:

```bash
python scripts/generate_tailored_answers.py --input training-data/expanded
```

## Related

**Parent Skill:**
- `skill_synthetic_data.md` - Synthetic data sub-master

**Sibling Skills:**
- `skill_qa_generation.md` - Generate Q&A pairs
- `skill_pearl_dataset.md` - Pearltrees training data

**Upstream Tools:**
- `scripts/expand_clusters_to_pairs.py` - Prepare input pairs

**Code:**
- `scripts/generate_tailored_answers.py`
