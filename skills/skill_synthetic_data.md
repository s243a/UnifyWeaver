# Skill: Synthetic Data (Sub-Master)

Tools for generating synthetic training data using LLMs, including Q&A pairs, reworded answers, and structured datasets from source documents.

## When to Use

- User asks "how do I generate training data?"
- User wants to create Q&A pairs from documentation
- User needs to expand or augment existing training data
- User asks about generating data from skills or source files
- User wants to create Pearltrees training datasets

## Skill Hierarchy

```
skill_data_tools.md (parent)
└── skill_synthetic_data.md (this file)
    ├── skill_qa_generation.md - Generate Q&A from skills/docs
    ├── skill_answer_tailoring.md - Reword answers with LLM
    └── skill_pearl_dataset.md - Pearltrees training data from RDF
```

## Quick Start

### Generate Q&A from Skills

```bash
# Generate Q&A pairs from a single skill
python training-data/scripts/generate_qa_from_skills.py \
  --skill skill_mindmap_linking.md \
  --model haiku

# Generate from all skills
python training-data/scripts/generate_qa_from_skills.py \
  --all \
  --provider gemini
```

### Tailor/Reword Answers

```bash
# Reword answers in expanded Q&A files
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --model sonnet

# Use Gemini instead
python scripts/generate_tailored_answers.py \
  --input training-data/expanded \
  --provider gemini
```

### Generate Pearltrees Dataset

```bash
# Generate training targets from Pearltrees RDF
python scripts/generate_pearl_dataset.py \
  --rdf data/s243a.rdf \
  --output reports/pearltrees_targets.jsonl \
  --query-style locate
```

## The Synthetic Data Pipeline

```
1. Source Documents
   ├── Skills (skill_*.md)
   ├── Documentation (docs/, education/)
   └── Pearltrees RDF exports

2. Q&A Generation (LLM-based)
   └── generate_qa_from_skills.py
   └── generate_quickstart_qa.py

3. Answer Tailoring (LLM-based)
   └── generate_tailored_answers.py

4. Dataset Expansion
   └── expand_clusters_to_pairs.py
   └── generate_tree_refpearls.py

5. Training Data (JSONL)
   └── training-data/by-topic/
   └── reports/pearltrees_targets.jsonl
```

## LLM Provider Support

All synthetic data tools support multiple LLM backends:

| Provider | CLI Tool | Models |
|----------|----------|--------|
| Claude | `claude` | sonnet, opus, haiku |
| Gemini | `gemini` | gemini-2.5-flash-preview, gemini-3-flash-preview |

```bash
# Claude (default)
--provider claude --model sonnet

# Gemini
--provider gemini --model gemini-2.5-flash-preview
```

## Output Format

All tools output JSONL with consistent structure:

```json
{
  "id": "skill_mindmap_linking_001",
  "question": "How do I link mindmaps to Pearltrees?",
  "question_variants": ["How to connect mindmaps with Pearltrees?"],
  "level": 2,
  "tree_path": ["Mindmap", "Linking"],
  "answer": "Use link_pearltrees.py to enrich mindmaps...",
  "related_skills": ["skill_mindmap_linking.md"],
  "related_docs": ["docs/QUICKSTART_MINDMAP_LINKING.md"],
  "tags": ["mindmap", "pearltrees", "linking"]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (topic_section_NNN) |
| `question` | Primary question text |
| `question_variants` | Alternative phrasings |
| `level` | Specificity (0=identity, 2=general task, 3=specific task) |
| `tree_path` | Hierarchical categorization |
| `answer` | Answer text |
| `related_skills` | Skill files for more info |
| `related_docs` | Documentation paths |
| `tags` | Searchable keywords |

## Common Workflows

### Bootstrap Training Data for New Topic

```bash
# 1. Create skill document
# skills/skill_new_feature.md

# 2. Generate Q&A from skill
python training-data/scripts/generate_qa_from_skills.py \
  --skill skill_new_feature.md \
  --pairs 6

# 3. Review and edit generated pairs
# training-data/by-topic/<topic>/skills-generated.jsonl

# 4. Expand to individual pairs
python scripts/expand_clusters_to_pairs.py \
  --input training-data/by-topic/<topic>/ \
  --output training-data/expanded/

# 5. Generate tailored answers
python scripts/generate_tailored_answers.py \
  --input training-data/expanded/<topic>/
```

### Augment Existing Training Data

```bash
# Generate reworded versions of existing answers
python scripts/generate_tailored_answers.py \
  --file training-data/expanded/mindmap/pairs.jsonl \
  --batch-size 5
```

### Coverage Analysis

```bash
# Check which capabilities have skills
python training-data/scripts/generate_qa_from_skills.py --coverage
```

## Child Skills

- `skill_qa_generation.md` - Generate Q&A pairs from source documents
- `skill_answer_tailoring.md` - Reword/tailor answers using LLM
- `skill_pearl_dataset.md` - Generate Pearltrees training datasets

## Related

**Parent Skill:**
- `skill_data_tools.md` - Data tools master

**Sibling Sub-Masters:**
- `skill_query_patterns.md` - Query and aggregation
- `skill_ml_tools.md` - Embeddings, training, inference
- `skill_data_sources.md` - JSON/JSONL data sources

**Code:**
- `scripts/generate_tailored_answers.py` - Answer tailoring
- `training-data/scripts/generate_qa_from_skills.py` - Q&A from skills
- `training-data/scripts/generate_quickstart_qa.py` - Q&A from source mapping
- `scripts/generate_pearl_dataset.py` - Pearltrees dataset generation
- `scripts/expand_clusters_to_pairs.py` - Cluster expansion
