# Chapter Review Playbook

**Purpose:** Review an education chapter, test code examples, identify gaps, generate Q&A pairs, and map prerequisites.

**Output:** JSONL file for `training-data/education/`

## Prerequisites

Before running this playbook:
- Access to the chapter markdown file
- Access to UnifyWeaver source code (for verification)
- SWI-Prolog installed (for Prolog examples)
- Python 3.9+ (for Python examples)

## Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `CHAPTER_PATH` | Path to chapter markdown | `education/UnifyWeaver_Education/book-01-foundations/02_prolog_fundamentals.md` |
| `BOOK_NAME` | Book identifier | `book-01-foundations` |
| `CHAPTER_NUM` | Chapter number | `02` |
| `OUTPUT_DIR` | Output directory | `training-data/education/book-01-foundations/` |

## Step 1: Parse Chapter Structure

Read the chapter and extract:

```bash
# Read the chapter
cat "$CHAPTER_PATH"
```

Identify:
- [ ] Chapter title (H1 heading)
- [ ] Section headings (H2, H3)
- [ ] Code blocks with language tags
- [ ] Navigation links (previous/next chapters)
- [ ] Any "Prerequisites" or "What You'll Learn" sections

**Expected output:**
```
Title: Chapter 2: Prolog Fundamentals for UnifyWeaver
Sections: [The Building Blocks: Terms, Facts: Stating What Is True, Rules: Defining New Knowledge, ...]
Code blocks: 8 (prolog: 7, text: 1)
Navigation: prev=01_introduction, next=03_unifyweaver_architecture
```

## Step 2: Extract Code Blocks

For each code block:

1. **Identify language** from markdown fence (```prolog, ```python, ```bash)
2. **Determine if executable** - is it a complete example or a fragment?
3. **Extract context** - what text precedes/follows the code?

Create a list:
```json
{
  "block_id": 1,
  "language": "prolog",
  "code": "file_dependency('main.o', 'main.c').\nfile_dependency('main.o', 'utils.h').",
  "line_start": 62,
  "line_end": 68,
  "context_before": "Example: Defining a file dependency graph",
  "executable": true,
  "is_query": false
}
```

## Step 3: Test Code Blocks

For each executable code block:

### Prolog Code

```bash
# Create temp file with code
cat > /tmp/test_ch${CHAPTER_NUM}_block${BLOCK_ID}.pl << 'EOF'
% Code from block
file_dependency('main.o', 'main.c').
file_dependency('main.o', 'utils.h').
EOF

# Test loading
swipl -g "consult('/tmp/test_ch${CHAPTER_NUM}_block${BLOCK_ID}.pl')" -t halt
```

**If error occurs:**
- [ ] Document the error
- [ ] Identify missing imports (e.g., `use_module`)
- [ ] Note required context from earlier in chapter
- [ ] Suggest fix if possible

### Prolog Queries

For query examples (lines starting with `?-`):

```bash
# Combine facts + query
swipl -g "consult('/tmp/test_facts.pl'), file_dependency('main.o', Dep), writeln(Dep)" -t halt
```

### Python Code

```bash
python3 -c "
# Code from block
from typing import Iterator
def read_stream(stream):
    for line in stream:
        yield line
"
```

### Bash Code

```bash
# Test bash examples
bash -n << 'EOF'
# Code from block
EOF
```

**Record results:**
```json
{
  "block_id": 1,
  "test_status": "pass",
  "output": "",
  "modifications_needed": null
}
```

Or if modifications needed:
```json
{
  "block_id": 3,
  "test_status": "fail",
  "error": "Undefined procedure: file_dependency/2",
  "modifications_needed": "Add facts from block 1 before this query",
  "suggested_fix": "Combine with previous example or add ':- consult(facts).'"
}
```

## Step 4: Identify Documentation Gaps

Review the chapter for:

### Missing Explanations
- [ ] Are all terms defined before use?
- [ ] Are concepts introduced in logical order?
- [ ] Are there unexplained jumps in complexity?

### Missing Examples
- [ ] Does each concept have at least one example?
- [ ] Are edge cases covered?
- [ ] Are common mistakes shown?

### Missing Prerequisites
- [ ] What must the reader know before this chapter?
- [ ] Are there links to prerequisite chapters?
- [ ] Is environment setup explained or referenced?

**Document gaps:**
```json
{
  "gap_type": "missing_explanation",
  "location": "Section 'Unification'",
  "description": "The occurs check is mentioned but not explained",
  "severity": "medium",
  "suggested_addition": "Add a sentence explaining what the occurs check prevents"
}
```

## Step 5: Generate Reader Questions

Based on the content, generate questions a reader would likely ask:

### Concept Questions
- "What is X?"
- "How does X work?"
- "Why do we need X?"

### Practical Questions
- "How do I X?"
- "What's the syntax for X?"
- "Show me an example of X"

### Troubleshooting Questions
- "Why does X fail?"
- "What if X doesn't work?"
- "How do I debug X?"

### Connecting Questions
- "How does X relate to Y?"
- "When should I use X vs Y?"
- "What's the difference between X and Y?"

**Generate 5-10 questions per major section:**
```json
{
  "section": "Facts: Stating What Is True",
  "questions": [
    {
      "text": "How do I define a Prolog fact?",
      "type": "practical",
      "length": "short"
    },
    {
      "text": "What is the difference between a fact and a rule in Prolog?",
      "type": "connecting",
      "length": "medium"
    }
  ]
}
```

## Step 6: Provide Answers

For each generated question:

1. **Check if chapter answers it** - quote the relevant text
2. **If not in chapter, check source code** - reference the implementation
3. **If not in source, check other documentation** - link to related docs
4. **If unanswerable, mark as gap**

```json
{
  "question": "How do I define a Prolog fact?",
  "answer_source": "chapter",
  "answer_text": "A fact is a complex term that is asserted to be true. Facts are the foundation of a Prolog program; they are the raw data the program works with.",
  "answer_code": "file_dependency('main.o', 'main.c').",
  "source_location": "02_prolog_fundamentals.md:55-68"
}
```

## Step 7: Map Prerequisites

Identify what the reader needs before this chapter:

### Knowledge Prerequisites
- Concepts from earlier chapters
- External knowledge (e.g., "basic programming")

### Environment Prerequisites
- Tools that must be installed
- Configuration that must be done

### Content Prerequisites
- Specific chapters that must be read first
- Playbooks that should be completed

**Create prerequisite mapping:**
```json
{
  "chapter": "book-01-ch02",
  "prerequisites": {
    "knowledge": [
      {
        "concept": "What UnifyWeaver is",
        "source": "book-01-ch01",
        "relation": "foundational"
      }
    ],
    "environment": [
      {
        "requirement": "SWI-Prolog installed",
        "setup_ref": "prerequisites/prolog-installation"
      }
    ],
    "chapters": [
      {
        "chapter_id": "book-01-ch01",
        "relation": "preliminary"
      }
    ]
  }
}
```

## Step 8: Suggest Relations

Based on content analysis, suggest relations to other content:

```json
{
  "relations": [
    {
      "from": "book-01-ch02-facts",
      "to": "book-01-ch02-rules",
      "type": "preliminary",
      "reason": "Facts must be understood before rules"
    },
    {
      "from": "book-01-ch02-transitive",
      "to": "src/unifyweaver/core/transitive_closure.pl",
      "type": "implementation",
      "reason": "Source implements the transitive closure pattern shown"
    },
    {
      "from": "book-01-ch02",
      "to": "book-02-ch01",
      "type": "foundational",
      "reason": "Prolog fundamentals are foundation for Bash target"
    }
  ]
}
```

## Step 9: Generate Output JSONL

Combine all findings into the output format:

```bash
mkdir -p "$OUTPUT_DIR"
```

Write `ch${CHAPTER_NUM}_*.jsonl`:

```json
{
  "cluster_id": "book-01-ch02-facts",
  "source_file": "education/book-01-foundations/02_prolog_fundamentals.md",
  "source_type": "education",
  "section": "Facts: Stating What Is True",
  "answer": {
    "text": "A fact is a complex term that is asserted to be true...",
    "code_blocks": [
      {
        "language": "prolog",
        "code": "file_dependency('main.o', 'main.c').",
        "executable": true,
        "test_status": "pass",
        "line_start": 62
      }
    ]
  },
  "questions": [
    {"text": "How do I define a Prolog fact?", "type": "practical", "length": "short"},
    {"text": "What is the syntax for stating something is true?", "type": "practical", "length": "short"}
  ],
  "gaps": [],
  "prerequisites": ["book-01-ch01"],
  "relations": [
    {"to": "book-01-ch02-rules", "type": "preliminary"}
  ]
}
```

## Step 10: Verify Output

```bash
# Validate JSONL
python3 -c "
import json
with open('${OUTPUT_DIR}/ch${CHAPTER_NUM}_clusters.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
print('Validation complete')
"
```

## Success Criteria

- [ ] All code blocks tested
- [ ] Failed tests documented with suggested fixes
- [ ] 5+ questions generated per major section
- [ ] All questions have answers or are marked as gaps
- [ ] Prerequisites mapped
- [ ] Relations to other content identified
- [ ] Valid JSONL output generated

## Example Execution

```
Input: education/UnifyWeaver_Education/book-01-foundations/02_prolog_fundamentals.md
Output: training-data/education/book-01-foundations/ch02_prolog_fundamentals.jsonl

Summary:
- Sections processed: 6
- Code blocks tested: 8 (7 pass, 1 needs context)
- Questions generated: 24
- Gaps identified: 2
- Prerequisites mapped: 3
- Relations suggested: 5
```
