# Quickstart Agent Implementation Plan

Step-by-step plan for building the quickstart agent and task-oriented Q&A database.

## Phase 1: Foundation

### 1.1 Create Database Files

Create empty/template files:

```bash
# Q&A database
touch data/quickstart_qa.jsonl

# Skills index (auto-generated from skills/*.md)
touch data/skills_index.jsonl

# Coverage tracking
touch reports/quickstart_coverage.md
```

### 1.2 Build Skills Indexer

Script to extract metadata from skill files:

**File:** `scripts/build_skills_index.py`

```python
"""
Extract triggers, prerequisites, and tree paths from skills/*.md
Output: data/skills_index.jsonl
"""

def extract_skill_metadata(skill_path: Path) -> dict:
    # Parse markdown
    # Extract "When to Use" section → triggers
    # Extract "Prerequisites" section
    # Extract "Quick Start" command
    # Infer tree_path from filename/content

def main():
    skills = glob("skills/skill_*.md")
    index = [extract_skill_metadata(s) for s in skills]
    write_jsonl("data/skills_index.jsonl", index)
```

### 1.3 Seed Level 0-1 Questions

Create foundational Q&A entries manually:

**Level 0 (Identity):**
- "What is UnifyWeaver?"
- "What is this project for?"

**Level 1 (Capabilities):**
- "What can UnifyWeaver do?"
- "What can I compile to?"
- "What are the mindmap tools?"
- "Can I generate apps from Prolog?"
- "How does semantic search work?"

**Target:** 10-15 seed entries covering major branches.

## Phase 2: Q&A Content

### 2.1 Level 2-3 Questions by Branch

Work through capability tree, writing Q&A entries for each branch:

| Branch | Level 2 (General) | Level 3 (Specific) |
|--------|-------------------|-------------------|
| Compilation | "How do I compile?" | "How do I compile to Bash/Go/SQL/...?" |
| Mindmap | "How do I organize mindmaps?" | "How do I link to Pearltrees/cross-link/...?" |
| GUI | "How do I generate an app?" | "How do I generate Vue/Flutter/...?" |
| Semantic | "How do I train a model?" | "How do I use MST clustering/...?" |
| Security | "How do I secure my app?" | "How do I add OAuth/TLS/...?" |

**Target:** 50-80 entries covering all major branches.

### 2.2 Link Entries to Skills and Docs

For each Q&A entry, populate:

- `related_skills`: Which skill file helps accomplish this?
- `related_docs`: Which doc explains the concepts?
- `prerequisites`: Which Q&A entries should the user know first?
- `go_deeper`: What follow-up questions are natural?

### 2.3 Add Verification Commands

For each entry, add a `verify_command` that:
- Confirms prerequisites are met, OR
- Demonstrates the answer worked

Example:
```jsonl
{
  "question": "How do I compile to Bash?",
  "verify_command": "swipl -g 'use_module(src/compiler_driver)' -t halt && echo 'Compiler loaded OK'"
}
```

## Phase 3: Embedding and Retrieval

### 3.1 Embed Q&A Entries

Script to embed questions and variants:

**File:** `scripts/embed_quickstart_qa.py`

```python
"""
Embed all questions and variants from quickstart_qa.jsonl
Output: models/quickstart_qa_embeddings.pkl
"""

def embed_qa_entries(qa_path: str, model_name: str):
    entries = load_jsonl(qa_path)

    for entry in entries:
        texts = [entry["question"]] + entry.get("question_variants", [])
        embeddings = embed_texts(texts, model_name)
        entry["embedding"] = mean_pool(embeddings)

    save_embeddings("models/quickstart_qa_embeddings.pkl", entries)
```

### 3.2 Embed Skills Index

Similar process for skill triggers:

**File:** `scripts/embed_skills_index.py`

```python
"""
Embed skill triggers and descriptions
Output: models/skills_index_embeddings.pkl
"""
```

### 3.3 Build Retrieval Functions

**File:** `scripts/quickstart_retrieval.py`

```python
"""
Retrieval functions for quickstart agent
"""

def retrieve_qa(query: str, top_k: int = 5) -> List[QAEntry]:
    """Semantic search over Q&A database"""

def retrieve_skill(query: str, top_k: int = 3) -> List[SkillEntry]:
    """Match query to skill triggers"""

def classify_intent(query: str) -> Intent:
    """Determine if query is understanding/action/hybrid"""

def retrieve(query: str) -> RetrievalResult:
    """Main entry point - returns Q&A and/or skills based on intent"""
```

## Phase 4: Agent Integration

### 4.1 Create Quickstart Skill

**File:** `skills/skill_quickstart.md`

```markdown
# Skill: Quickstart

Help new users find the right starting point.

## When to Use

- User says "help", "where do I start", "new here"
- User asks a broad question about capabilities
- User seems lost or overwhelmed

## Workflow

1. Classify user intent (understanding vs action)
2. Search quickstart Q&A database
3. Search skills index
4. Present answer with related skills/docs
5. Offer to go deeper or take action
```

### 4.2 Add Router Entry

Update `skills/SKILL_ROUTER.md`:

```markdown
## 0.0 Quickstart (New Users)

- 0.1 IF user says "help" OR "get started" OR "new here" OR "what can"
  - 0.1.1 LOAD `skill_quickstart.md`
- 0.2 IF question seems broad or unfocused
  - 0.2.1 LOAD `skill_quickstart.md`
```

### 4.3 Update Agent Instructions

Update `AGENT_INSTRUCTIONS.md`:

```markdown
## Quickstart System

For new users or broad questions, use the quickstart retrieval:

1. Check `data/quickstart_qa.jsonl` for direct answers
2. Check `data/skills_index.jsonl` for relevant skills
3. Present answer with pointers to skills and docs
```

## Phase 5: Coverage and Quality

### 5.1 Build Coverage Report

**File:** `scripts/quickstart_coverage.py`

```python
"""
Generate coverage report: which tree nodes have Q&A/skills?
Output: reports/quickstart_coverage.md
"""

def generate_coverage_report():
    tree = load_capability_tree()
    qa_entries = load_jsonl("data/quickstart_qa.jsonl")
    skills = load_jsonl("data/skills_index.jsonl")

    for node in tree.all_nodes():
        qa_count = count_for_path(qa_entries, node.path)
        skill_count = count_for_path(skills, node.path)
        # Generate markdown report with coverage %
```

### 5.2 Gap Analysis

Run coverage report to find:
- Tree nodes with no Q&A entries
- Tree nodes with no skills
- Orphaned entries (not in tree)

### 5.3 Quality Checklist

For each entry, verify:
- [ ] Question is phrased in user language (no jargon)
- [ ] Answer includes concrete next action
- [ ] `related_skills` populated (if applicable)
- [ ] `related_docs` populated (if applicable)
- [ ] `verify_command` works
- [ ] `prerequisites` are accurate

## Phase 6: Future Enhancements

### 6.1 Merge with General Q&A Database

Once quickstart DB is stable:

```python
# Merge databases
merged = quickstart_qa + general_qa

# Re-embed with unified model
embeddings = embed_all(merged)

# Or keep as priority tier
def retrieve(query):
    quickstart_results = search_quickstart(query)
    if quickstart_results.confidence > 0.8:
        return quickstart_results
    return search_general(query)
```

### 6.2 Distill to Transformer

Train a small model on Q&A pairs:

```python
# Fine-tune on (question, answer) pairs
model = train_qa_model(
    questions=[e["question"] for e in qa_entries],
    answers=[e["answer"] for e in qa_entries],
    base_model="distilbert-base-uncased"
)
```

### 6.3 Interactive Refinement

Add feedback loop:
- Track which Q&A entries are used
- Track user satisfaction (did they accomplish goal?)
- Surface low-quality entries for review

## Timeline

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Foundation | 1-2 hours | None |
| Phase 2: Q&A Content | 4-8 hours | Phase 1 |
| Phase 3: Embedding | 2-3 hours | Phase 2 |
| Phase 4: Agent Integration | 2-3 hours | Phase 3 |
| Phase 5: Coverage | 1-2 hours | Phase 4 |
| Phase 6: Enhancements | Ongoing | Phase 5 |

## Files Created/Modified

| File | Action | Phase |
|------|--------|-------|
| `data/quickstart_qa.jsonl` | Create | 1 |
| `data/skills_index.jsonl` | Create | 1 |
| `scripts/build_skills_index.py` | Create | 1 |
| `scripts/embed_quickstart_qa.py` | Create | 3 |
| `scripts/embed_skills_index.py` | Create | 3 |
| `scripts/quickstart_retrieval.py` | Create | 3 |
| `scripts/quickstart_coverage.py` | Create | 5 |
| `skills/skill_quickstart.md` | Create | 4 |
| `skills/SKILL_ROUTER.md` | Modify | 4 |
| `AGENT_INSTRUCTIONS.md` | Modify | 4 |
| `reports/quickstart_coverage.md` | Create | 5 |
| `models/quickstart_qa_embeddings.pkl` | Create | 3 |
| `models/skills_index_embeddings.pkl` | Create | 3 |

## Related Documents

- `QUICKSTART_AGENT_PROPOSAL.md` - Problem statement and capability tree
- `QUICKSTART_AGENT_PHILOSOPHY.md` - Design principles
- `QUICKSTART_AGENT_SPECIFICATION.md` - Database schema and retrieval logic
