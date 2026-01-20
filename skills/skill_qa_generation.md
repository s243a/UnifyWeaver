# Skill: Q&A Generation

Generate question-answer training pairs from source documents (skills, documentation, source files) using LLMs.

## When to Use

- User asks "how do I create Q&A training data?"
- User wants to generate training pairs from skill files
- User needs to create quickstart agent data
- User asks about SOURCE_MAPPING.md or skills-generated.jsonl

## Overview

Two complementary tools for Q&A generation:

| Tool | Source | Best For |
|------|--------|----------|
| `generate_qa_from_skills.py` | Skill markdown files | Task-oriented Q&A |
| `generate_quickstart_qa.py` | SOURCE_MAPPING.md | Capability-based Q&A |

Both use LLMs (Claude or Gemini) to generate structured JSONL output.

## Generate Q&A from Skills

### Basic Usage

```bash
# Single skill
python training-data/scripts/generate_qa_from_skills.py \
  --skill skill_mindmap_linking.md

# All skills
python training-data/scripts/generate_qa_from_skills.py --all

# With specific model
python training-data/scripts/generate_qa_from_skills.py \
  --all \
  --provider gemini \
  --model gemini-2.5-flash-preview
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--skill` | - | Process specific skill file |
| `--all` | - | Process all skill_*.md files |
| `--coverage` | - | Analyze skill coverage gaps |
| `--provider` | claude | LLM provider (claude, gemini) |
| `--model` | sonnet | Model name |
| `--pairs` | 4 | Q&A pairs per skill |
| `--output` | by-topic | Output directory |

### How It Works

1. Reads skill markdown file
2. Extracts "When to Use" section for question ideas
3. Extracts commands/examples for answer content
4. Prompts LLM to generate Q&A pairs
5. Infers topic from skill name (mindmap, compilation, etc.)
6. Outputs to `by-topic/<topic>/skills-generated.jsonl`

### Coverage Analysis

Check which capabilities have corresponding skills:

```bash
python training-data/scripts/generate_qa_from_skills.py --coverage
```

Output shows:
- Existing skills with checkmarks
- Missing skills (capability tree items without skills)
- Suggested new skills to create

## Generate Q&A from Source Mapping

### Basic Usage

```bash
# Single topic
python training-data/scripts/generate_quickstart_qa.py \
  --topic compilation

# All topics
python training-data/scripts/generate_quickstart_qa.py --all
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--topic` | - | Generate for specific topic |
| `--all` | - | Generate for all topics |
| `--provider` | claude | LLM provider |
| `--model` | sonnet | Model name |
| `--pairs` | 3 | Q&A pairs per section |
| `--output` | by-topic | Output directory |

### SOURCE_MAPPING.md Format

The tool reads `training-data/by-topic/SOURCE_MAPPING.md`:

```markdown
## Compilation

### Targets (Level 2-3)

| Target | Source Files |
|--------|--------------|
| Bash | `src/unifyweaver/targets/bash_target.pl`, `education/book-02-bash-target/` |
| Go | `src/unifyweaver/targets/go_target.pl` |
```

### How It Works

1. Parses SOURCE_MAPPING.md for topics/sections
2. Reads referenced source files (truncated to 150 lines each)
3. Prompts LLM with source content
4. Generates Q&A with appropriate level tags
5. Outputs to `by-topic/<topic>/<topic>-generated.jsonl`

## Output Format

Both tools produce JSONL with this structure:

```json
{
  "id": "skill_mindmap_linking_001",
  "question": "How do I enrich mindmaps with Pearltrees links?",
  "question_variants": [
    "How to add Pearltrees references to mindmaps?",
    "Can I link my mindmaps to Pearltrees folders?"
  ],
  "level": 2,
  "tree_path": ["Mindmap", "Mindmap Linking"],
  "answer": "Use the link_pearltrees.py script to enrich SimpleMind exports...",
  "related_skills": ["skill_mindmap_linking.md"],
  "related_docs": ["docs/QUICKSTART_MINDMAP_LINKING.md"],
  "tags": ["mindmap", "pearltrees", "linking", "enrichment"]
}
```

### Level Meanings

| Level | Description | Example |
|-------|-------------|---------|
| 0 | Identity | "What is UnifyWeaver?" |
| 1 | Capabilities | "What can UnifyWeaver compile to?" |
| 2 | General task | "How do I organize mindmaps?" |
| 3 | Specific task | "How do I use MST clustering?" |
| 4 | Details | "What's the --target-size parameter?" |

## Best Practices

### Question Quality

- Questions should be user-centric (what a new user would ask)
- Don't mention skill names in questions (users don't know them)
- Include 2-3 question variants for training diversity

### Answer Quality

- Keep answers concise but complete
- Include code examples where helpful
- Reference paths for "learn more" (e.g., "See education/book-02-bash-target/")

### Workflow

```bash
# 1. Generate initial Q&A
python training-data/scripts/generate_qa_from_skills.py --all --pairs 4

# 2. Review generated files
ls training-data/by-topic/*/skills-generated.jsonl

# 3. Edit/curate as needed

# 4. Expand clusters to pairs
python scripts/expand_clusters_to_pairs.py \
  --input training-data/by-topic/ \
  --output training-data/expanded/

# 5. Tailor answers for variation
python scripts/generate_tailored_answers.py \
  --input training-data/expanded/
```

## Troubleshooting

### "CLI error" or timeout

- Check that `claude` or `gemini` CLI is installed and authenticated
- Try a smaller model (haiku is faster)
- Increase timeout if needed

### Poor quality output

- Review the skill file - clear "When to Use" sections help
- Try a different model (opus for quality, haiku for speed)
- Reduce pairs per skill and generate more iterations

### Missing topics

Run coverage analysis to identify gaps:

```bash
python training-data/scripts/generate_qa_from_skills.py --coverage
```

## Related

**Parent Skill:**
- `skill_synthetic_data.md` - Synthetic data sub-master

**Sibling Skills:**
- `skill_answer_tailoring.md` - Reword generated answers
- `skill_pearl_dataset.md` - Pearltrees training data

**Code:**
- `training-data/scripts/generate_qa_from_skills.py`
- `training-data/scripts/generate_quickstart_qa.py`
- `training-data/by-topic/SOURCE_MAPPING.md`
