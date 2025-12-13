# Proposal: LLM-Generated Mappings for Code Example RAG

## Summary

This proposal describes a system where LLMs generate multiple "mapping" representations for each code example and documentation page. These mappings serve as query expansion on the document side, making it more likely that diverse user queries will match relevant content.

## Problem

Traditional RAG systems embed documents directly and hope user queries semantically match the content. This fails when:

1. **Vocabulary mismatch**: User says "filter rows" but doc says "predicate selection"
2. **Language mismatch**: User thinks in Python/SQL but examples are in Prolog/Bash
3. **Abstraction mismatch**: User describes high-level intent, docs contain low-level implementation
4. **Query style variance**: Some users ask questions, others paste error messages, others describe desired output

## Proposed Solution

Generate multiple **mapping documents** for each source document. Each mapping is a different representation designed to match different query styles:

```
Source Document
     │
     ├── Question Mappings     "How do I...?", "What is...?"
     ├── Task Mappings         "filter CSV rows", "parse XML stream"
     ├── Pseudocode Mappings   "read file → filter → output"
     ├── Code Analog Mappings  Same concept in Python, SQL, etc.
     ├── Error Mappings        Common errors this doc solves
     └── Keyword Mappings      Relevant terms and synonyms
```

At query time, we search mappings (not raw docs), then return the linked source documents.

## Mapping Schema

### YAML Format

```yaml
# mappings/examples/csv_data_source.yaml
source: playbooks/examples_library/csv_examples.md
record_id: unifyweaver.execution.csv_data_source
source_type: example  # example | documentation | playbook

mappings:
  # Natural language questions users might ask
  - type: question
    text: "How do I read a CSV file in UnifyWeaver?"

  - type: question
    text: "How can I filter CSV rows by column value?"

  - type: question
    text: "What's the UnifyWeaver equivalent of pandas read_csv?"

  # Task descriptions (imperative)
  - type: task
    text: "Read CSV data and query by field name"

  - type: task
    text: "Compile a CSV source to a bash function"

  # Pseudocode showing the pattern
  - type: pseudocode
    text: |
      source(csv, users, [file: 'users.csv', has_header: true])
      get_user_age(Name, Age) :- users(_, Name, Age)
      compile → bash function

  # Code in other languages that accomplishes similar task
  # Users thinking in these languages will match
  - type: code_analog
    language: python
    text: |
      import pandas as pd
      df = pd.read_csv('users.csv')
      age = df[df['name'] == 'Alice']['age']

  - type: code_analog
    language: sql
    text: |
      SELECT age FROM users WHERE name = 'Alice'

  - type: code_analog
    language: bash
    text: |
      awk -F, 'NR>1 && $2=="Alice" {print $3}' users.csv

  # Errors this example helps solve
  - type: error
    text: "csv_source not found"

  - type: error
    text: "undefined predicate users/3"

  # Keywords and synonyms for broad matching
  - type: keywords
    text: "csv comma separated values tabular data filter column row header delimiter tsv"

  # Concepts and patterns
  - type: concepts
    text: "data source, external data, file parsing, declarative query"
```

### Mapping Types

| Type | Purpose | Example |
|------|---------|---------|
| `question` | Match user questions | "How do I read CSV files?" |
| `task` | Match imperative descriptions | "Filter rows by column value" |
| `pseudocode` | Match abstract patterns | "read → filter → output" |
| `code_analog` | Match thinking in other languages | Python pandas, SQL queries |
| `error` | Match error messages | "undefined predicate users/3" |
| `keywords` | Broad term matching | "csv tabular column row" |
| `concepts` | Abstract pattern matching | "data source, declarative query" |

## Directory Structure

```
mappings/
├── examples/
│   ├── csv_data_source.yaml
│   ├── xml_streaming.yaml
│   ├── parallel_execution.yaml
│   └── ...
├── playbooks/
│   ├── csv_data_source_playbook.yaml
│   ├── csharp_codegen_playbook.yaml
│   └── ...
├── documentation/
│   ├── GO_TARGET.yaml
│   ├── RECURSION_PATTERN_THEORY.yaml
│   └── ...
└── schema/
    └── mapping_schema.json
```

## RAG Architecture

### Indexing Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Source Document │────▶│ LLM Generation  │────▶│ Mapping YAML    │
│ (example, doc)  │     │ (generate maps) │     │ (multiple reps) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Embed Each      │
                                                │ Mapping Text    │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Vector Store    │
                                                │ (mapping → src) │
                                                └─────────────────┘
```

### Query Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ User Query      │────▶│ Embed Query     │────▶│ Search Mappings │
│                 │     │                 │     │ (top-k similar) │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Dedupe by       │
                                                │ Source Document │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Return Source   │
                                                │ + Matched Maps  │
                                                └─────────────────┘
```

### Query Response

```json
{
  "query": "How do I filter CSV rows in Python style?",
  "results": [
    {
      "source": "playbooks/examples_library/csv_examples.md",
      "record_id": "unifyweaver.execution.csv_data_source",
      "relevance_score": 0.92,
      "matched_mappings": [
        {
          "type": "code_analog",
          "language": "python",
          "similarity": 0.94
        },
        {
          "type": "question",
          "text": "How can I filter CSV rows by column value?",
          "similarity": 0.89
        }
      ]
    }
  ]
}
```

## LLM Mapping Generation

### Method 1: Playbook-Integrated Generation (Preferred)

When running playbooks with LLMs, add a final step asking the LLM to generate search queries:

```markdown
## Step N: Generate Search Mappings (Optional)

Now that you've completed this task, imagine you needed to find this
example using a search system. What queries would you have tried?

Generate:
1. **Questions you might ask** (3-5)
2. **How you'd describe this task** (2-3 short phrases)
3. **Code snippets in other languages** that do similar things
4. **Error messages** this example helped you avoid or solve
5. **Keywords** you'd search for
```

**Benefits of playbook-integrated generation:**
- Mappings come from actual LLM "users" of the examples
- Diverse query styles from different models (Gemini, Claude, GPT)
- Natural integration into existing playbook testing workflow
- Real-world query patterns, not synthetic guesses
- Can compare queries across model tiers to find universal patterns

**Workflow:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ LLM Runs        │────▶│ Task Succeeds   │────▶│ "What queries   │
│ Playbook        │     │                 │     │  would find     │
└─────────────────┘     └─────────────────┘     │  this?"         │
                                                └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │ Append to       │
                                                │ mappings file   │
                                                └─────────────────┘
```

**Example playbook addition:**

```markdown
## Step 6: Generate Search Mappings

You successfully compiled a CSV source to bash. If you had to find this
example again using a semantic search system, what would you search for?

Think about:
- Questions a developer might ask
- How someone thinking in Python/SQL would describe this
- Error messages that would lead here
- Abstract concepts involved

Format your response as:
- Questions: ...
- Task descriptions: ...
- Code analogs: ...
- Errors: ...
- Keywords: ...
```

### Method 2: Batch Generation

For documents not covered by playbooks, use direct prompting:

### Prompt Template

```markdown
You are generating search mappings for a code example. These mappings help users find this example when searching with different query styles.

## Source Document
{document_content}

## Generate Mappings

Generate mappings in these categories:

1. **Questions** (3-5): Natural language questions this example answers
2. **Tasks** (2-3): Imperative task descriptions
3. **Pseudocode** (1-2): Abstract pattern representation
4. **Code Analogs** (2-4): Same concept in Python, SQL, JavaScript, etc.
5. **Errors** (1-3): Error messages this example helps solve
6. **Keywords** (1): Space-separated relevant terms
7. **Concepts** (1): Abstract patterns and concepts

Output as YAML following this schema:
{schema}
```

### Generation Workflow

1. **Batch processing**: Process all examples/docs in a batch
2. **Review step**: Human reviews generated mappings for quality
3. **Incremental updates**: When docs change, regenerate affected mappings
4. **Multi-model**: Optionally use multiple LLMs for diverse mappings

## Implementation Plan

### Phase 1: Schema and Tooling
- [ ] Define JSON schema for mapping files
- [ ] Create validation script for mapping files
- [ ] Build mapping generation prompt template

### Phase 2: Initial Corpus
- [ ] Generate mappings for `examples_library/` (20+ examples)
- [ ] Generate mappings for core playbooks (15+ playbooks)
- [ ] Human review and refinement

### Phase 3: Indexing Infrastructure
- [ ] Extend Go embedder to index mapping files
- [ ] Build mapping → source document linking
- [ ] Create deduplication logic for multi-mapping matches

### Phase 4: Query Interface
- [ ] CLI tool: `unifyweaver search "how do I filter CSV?"`
- [ ] API endpoint for programmatic access
- [ ] Integration with existing semantic search

### Phase 5: LLM Integration
- [ ] Tool definition for LLM to query mappings
- [ ] Response formatting for LLM consumption
- [ ] Feedback loop for mapping quality improvement

## Benefits

1. **Higher recall**: Multiple representations catch diverse queries
2. **Cross-language discovery**: Python user finds Prolog example via code analog
3. **Error-driven search**: Users can paste errors to find solutions
4. **Explainability**: "Matched via Python code analog" explains why result was returned
5. **Incremental improvement**: Add mappings based on failed queries

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Mapping generation cost | Batch process, cache results, only regenerate on doc change |
| Low-quality mappings | Human review step, feedback mechanism |
| Index bloat | Mappings are small text; storage is minimal |
| Stale mappings | CI check that mappings exist for all sources |

## Success Metrics

- **Query success rate**: % of queries that return relevant results
- **Cross-language matches**: % of matches via code_analog mappings
- **User satisfaction**: Feedback on result relevance
- **Coverage**: % of examples/docs with mappings

## Open Questions

1. Should mappings be versioned with source docs?
2. How many mappings per document is optimal?
3. Should users be able to contribute mappings?
4. Should we weight mapping types differently in ranking?

## Related Work

- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical docs for queries; we do the inverse
- **Query expansion**: Traditional IR technique; we expand on document side
- **Multi-vector retrieval**: ColBERT etc.; similar idea of multiple representations

## Appendix: Example Mapping File

See `mappings/examples/csv_data_source.yaml` for a complete example.
