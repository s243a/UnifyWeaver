# Quickstart Agent Specification

Technical specification for the quickstart agent's retrieval system, database schema, and skill integration.

## Overview

The quickstart agent retrieves from two sources:

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Intent Classification          │
│  (understanding vs action vs both)       │
└─────────────────────────────────────────┘
    │                              │
    ▼                              ▼
┌─────────────┐              ┌─────────────┐
│ Q&A Database│              │Skills Index │
│ (answers)   │              │ (workflows) │
└─────────────┘              └─────────────┘
    │                              │
    └──────────┬───────────────────┘
               ▼
┌─────────────────────────────────────────┐
│           Response Assembly              │
│  - Direct answer (if understanding)      │
│  - Skill reference (if action)           │
│  - Both (if hybrid)                      │
└─────────────────────────────────────────┘
```

## Intent Classification

### Intent Types

| Intent | Signal Words | Primary Source |
|--------|--------------|----------------|
| **Understanding** | "what is", "explain", "why", "how does X work" | Q&A Database |
| **Action** | "do", "run", "organize", "compile", "generate" | Skills Index |
| **Navigation** | "where", "find", "show me" | Both |
| **Hybrid** | "how do I" (understanding + action) | Q&A → Skill |

### Classification Logic

```python
def classify_intent(query: str) -> Intent:
    query_lower = query.lower()

    # Pure understanding
    if any(w in query_lower for w in ["what is", "explain", "why does"]):
        return Intent.UNDERSTANDING

    # Pure action
    if any(w in query_lower for w in ["organize my", "compile my", "run the"]):
        return Intent.ACTION

    # Hybrid (most "how do I" questions)
    if "how do i" in query_lower or "how can i" in query_lower:
        return Intent.HYBRID

    # Default to hybrid
    return Intent.HYBRID
```

## Q&A Database Schema

### Entry Format

```jsonl
{
  "id": "qa_001",
  "question": "How do I compile Prolog to Bash?",
  "question_variants": [
    "compile prolog to shell script",
    "generate bash from prolog",
    "convert logic to shell"
  ],
  "level": 3,
  "tree_path": ["Compilation", "Targets", "bash"],
  "answer": "Use compiler_driver.pl with --target bash:\n\n```bash\nswipl -g \"compile('input.pl', bash, 'output.sh')\" -t halt src/compiler_driver.pl\n```\n\nThis generates a shell script with Unix pipes for data flow.",
  "prerequisites": ["qa_general_compile"],
  "related_skills": ["skill_unifyweaver_compile.md"],
  "related_docs": ["docs/EXTENDED_README.md#bash-target", "docs/ENHANCED_PIPELINE_CHAINING.md"],
  "go_deeper": ["qa_bash_recursion", "qa_bash_chaining"],
  "verify_command": "ls output.sh",
  "tags": ["compilation", "bash", "getting-started"]
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | yes | Unique identifier |
| `question` | string | yes | Canonical question text |
| `question_variants` | string[] | no | Alternative phrasings for embedding |
| `level` | int | yes | 0=identity, 1=capabilities, 2=general task, 3=specific task, 4=details |
| `tree_path` | string[] | yes | Position in capability tree |
| `answer` | string | yes | Markdown answer text |
| `prerequisites` | string[] | no | IDs of questions user should know first |
| `related_skills` | string[] | no | Skill files relevant to this question |
| `related_docs` | string[] | no | Documentation files for deeper understanding |
| `go_deeper` | string[] | no | IDs of follow-up questions |
| `verify_command` | string | no | Command to verify answer works |
| `tags` | string[] | no | Searchable tags |

### Level Definitions

| Level | Name | Example | Count Target |
|-------|------|---------|--------------|
| 0 | Identity | "What is UnifyWeaver?" | 1-3 |
| 1 | Capabilities | "What can I compile to?" | 5-10 |
| 2 | General Task | "How do I compile?" | 10-20 |
| 3 | Specific Task | "How do I compile to Bash?" | 50-100 |
| 4 | Details | "What flags does Bash target support?" | unlimited |

## Skills Index Schema

### Entry Format

Skills are markdown files. The index extracts metadata:

```jsonl
{
  "skill_id": "skill_mst_folder_grouping",
  "file": "skills/skill_mst_folder_grouping.md",
  "title": "MST Folder Grouping",
  "triggers": [
    "organize mindmaps",
    "cluster mindmaps",
    "group by topic",
    "semantic clustering"
  ],
  "tree_path": ["Mindmap", "Organization", "MST folder grouping"],
  "prerequisites": ["federated model exists", "embeddings computed"],
  "quick_command": "python3 scripts/mindmap/mst_folder_grouping.py --help",
  "description": "Organize mindmaps into folders using MST-based semantic clustering"
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `skill_id` | string | yes | Unique identifier (derived from filename) |
| `file` | string | yes | Path to skill markdown file |
| `title` | string | yes | Human-readable title |
| `triggers` | string[] | yes | Phrases that should load this skill |
| `tree_path` | string[] | yes | Position in capability tree |
| `prerequisites` | string[] | no | What must be true before using skill |
| `quick_command` | string | no | Command to show help/verify setup |
| `description` | string | yes | One-line description |

### Trigger Extraction

Triggers come from:
1. `SKILL_INDEX.md` "User says..." table
2. Skill file "When to Use" section
3. Manual additions

## Retrieval Flow

### Understanding Query

```
User: "What is Procrustes projection?"
    │
    ▼
Intent: UNDERSTANDING
    │
    ▼
Search Q&A Database (semantic + keyword)
    │
    ▼
Return: Direct answer
    │
    ▼
Offer: "Want to see how to use it? → skill_folder_suggestion.md"
```

### Action Query

```
User: "Organize my mindmaps by topic"
    │
    ▼
Intent: ACTION
    │
    ▼
Search Skills Index (trigger matching)
    │
    ▼
Return: skill_mst_folder_grouping.md
    │
    ▼
Agent: Loads skill, follows its workflow
```

### Hybrid Query

```
User: "How do I organize mindmaps?"
    │
    ▼
Intent: HYBRID
    │
    ▼
Search Q&A Database → found qa_organize_mindmaps
Search Skills Index → found skill_mst_folder_grouping.md
    │
    ▼
Return: Answer + Skill reference
    │
    ▼
Answer: "UnifyWeaver can organize mindmaps using MST-based
        semantic clustering. This groups similar mindmaps
        into topic folders based on embedding similarity."

Skill: "To do this now, I'll follow skill_mst_folder_grouping.md:
        1. Check prerequisites (model, embeddings)
        2. Run mst_folder_grouping.py
        3. Review and adjust groupings"
```

## Embedding Strategy

### Question Embeddings

For Q&A retrieval, embed:
- Canonical question
- All question variants
- Tags (weighted lower)

Store as single vector (mean pooling) or multiple vectors (max similarity).

### Skill Trigger Embeddings

For skill matching, embed:
- All triggers
- Description
- Title

### Model Choice

Use same model as general Q&A database for consistency:
- Default: `nomic-embed-text-v1.5`
- Fallback: `sentence-transformers/all-MiniLM-L6-v2`

## Ranking and Selection

### Q&A Ranking

```python
def rank_qa_results(query_embedding, candidates):
    scores = []
    for qa in candidates:
        # Semantic similarity
        sim = cosine_similarity(query_embedding, qa.embedding)

        # Level bonus (prefer lower levels for broad queries)
        level_bonus = 0.1 * (4 - qa.level) if is_broad_query else 0

        # Prerequisite penalty (if user hasn't seen prereqs)
        prereq_penalty = 0.2 * len(unmet_prerequisites(qa))

        scores.append(sim + level_bonus - prereq_penalty)

    return sorted(zip(candidates, scores), key=lambda x: -x[1])
```

### Skill Ranking

```python
def rank_skill_results(query, candidates):
    scores = []
    for skill in candidates:
        # Trigger match (exact or fuzzy)
        trigger_score = max(
            fuzzy_match(query, trigger)
            for trigger in skill.triggers
        )

        # Tree path relevance
        path_score = tree_path_similarity(query, skill.tree_path)

        scores.append(0.7 * trigger_score + 0.3 * path_score)

    return sorted(zip(candidates, scores), key=lambda x: -x[1])
```

## Response Assembly

### Template: Understanding

```markdown
## {question}

{answer}

---
**Go deeper:**
- {go_deeper[0]}
- {go_deeper[1]}

**Related:**
- Skill: {related_skills[0]}
- Docs: {related_docs[0]}
```

### Template: Action

```markdown
Loading skill: **{skill.title}**

{skill.description}

**Prerequisites:**
{skill.prerequisites}

**Quick start:**
```bash
{skill.quick_command}
```

Following skill workflow...
```

### Template: Hybrid

```markdown
## {question}

{answer}

---
**To do this now:**

Loading skill: **{skill.title}**

{skill workflow continues...}
```

## Coverage Tracking

### Tree Coverage Report

```
Capability Tree Coverage
========================

Compilation [85%]
├── General [100%] ✓
├── Targets [90%]
│   ├── bash [100%] ✓
│   ├── go [100%] ✓
│   ├── rust [50%] !
│   └── ...
├── Recursion Patterns [80%]
└── Data [70%]

Mindmap [95%]
├── Linking [100%] ✓
├── Organization [100%] ✓
└── Cross-References [85%]

...

Legend: ✓ = fully covered, ! = needs attention
```

### Gap Detection

```python
def find_coverage_gaps(tree, qa_entries, skills):
    gaps = []
    for node in tree.all_nodes():
        qa_count = count_entries_for_path(qa_entries, node.path)
        skill_count = count_skills_for_path(skills, node.path)

        if qa_count == 0 and skill_count == 0:
            gaps.append(CoverageGap(
                path=node.path,
                severity="high",
                suggestion=f"Add Q&A or skill for {node.name}"
            ))
        elif qa_count == 0:
            gaps.append(CoverageGap(
                path=node.path,
                severity="medium",
                suggestion=f"Add Q&A entry (skill exists: {skill_count})"
            ))

    return gaps
```

## File Locations

| Component | Path |
|-----------|------|
| Q&A Database | `data/quickstart_qa.jsonl` |
| Skills Index | `data/skills_index.jsonl` |
| Embeddings | `models/quickstart_embeddings.pkl` |
| Coverage Report | `reports/quickstart_coverage.md` |
| Skills (source) | `skills/*.md` |

## Related Documents

- `QUICKSTART_AGENT_PROPOSAL.md` - Problem statement and capability tree
- `QUICKSTART_AGENT_PHILOSOPHY.md` - Design principles
- `QUICKSTART_AGENT_IMPLEMENTATION.md` - Build plan
