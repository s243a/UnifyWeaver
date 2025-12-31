# Source Review Playbook

**Purpose:** Review a UnifyWeaver source module, extract predicates, generate Q&A pairs, and map relations to education content.

**Output:** JSONL file for `training-data/source/`

## Prerequisites

Before running this playbook:
- Access to the source file
- Access to UnifyWeaver education content (for linking)
- SWI-Prolog installed (for testing)

## Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `SOURCE_PATH` | Path to source file | `src/unifyweaver/targets/python_target.pl` |
| `MODULE_TYPE` | Type of module | `targets`, `core`, `glue`, `sources` |
| `OUTPUT_DIR` | Output directory | `training-data/source/targets/` |

## Step 1: Parse Module Structure

Read the source file and extract:

```bash
cat "$SOURCE_PATH"
```

Identify:
- [ ] Module declaration (`:- module(Name, Exports).`)
- [ ] Exported predicates (public API)
- [ ] Internal predicates (private)
- [ ] Module imports (`:- use_module(...)`)
- [ ] Comments and documentation

**Expected output:**
```
Module: python_target
Exports: 45 predicates
Imports: [common_generator, firewall, service_validation, ...]
Documentation: Module header comment present
```

## Step 2: Extract Exported Predicates

For each exported predicate:

1. **Find the declaration** in the export list
2. **Find the implementation** (all clauses)
3. **Extract documentation** (comments above predicate)
4. **Identify arity and argument names**

Create predicate entries:
```json
{
  "predicate": "compile_enhanced_pipeline/3",
  "module": "python_target",
  "export_line": 45,
  "implementation_lines": [8650, 8720],
  "documentation": "Compile an enhanced pipeline with fan-out, merge, routing stages.",
  "arguments": [
    {"name": "Stages", "type": "list", "description": "Pipeline stage specifications"},
    {"name": "Options", "type": "list", "description": "Compilation options"},
    {"name": "Code", "type": "string", "description": "Generated Python code"}
  ],
  "examples_in_source": true
}
```

## Step 3: Test Exported Predicates

For key predicates, verify they load and can be called:

```bash
# Test module loads
swipl -g "use_module('$SOURCE_PATH')" -t halt

# Test specific predicate exists
swipl -g "use_module('$SOURCE_PATH'), current_predicate(compile_enhanced_pipeline/3)" -t halt
```

For predicates with test clauses (e.g., `test_*`):
```bash
swipl -g "use_module('$SOURCE_PATH'), test_enhanced_pipeline_chaining" -t halt
```

**Record results:**
```json
{
  "predicate": "compile_enhanced_pipeline/3",
  "loads": true,
  "has_tests": true,
  "test_result": "pass"
}
```

## Step 4: Identify Predicate Categories

Group predicates by function:

### Code Generation
- `compile_*/3` - Main compilation entry points
- `generate_*/3` - Generate specific constructs

### Helpers
- `format_*/2` - Formatting utilities
- `extract_*/2` - Data extraction

### Infrastructure
- `*_helpers/1` - Helper code generation
- `*_infrastructure/1` - Service/runtime infrastructure

### Testing
- `test_*/0` - Self-test predicates

```json
{
  "categories": {
    "code_generation": ["compile_enhanced_pipeline/3", "compile_python_semantic/3"],
    "helpers": ["format_stage_list/2", "extract_stage_names/2"],
    "infrastructure": ["enhanced_pipeline_helpers/1", "service_infrastructure/1"],
    "testing": ["test_enhanced_pipeline_chaining/0"]
  }
}
```

## Step 5: Generate Questions

For each exported predicate, generate:

### Usage Questions
- "How do I use X?"
- "What does X do?"
- "What are the arguments to X?"

### Conceptual Questions
- "What is the purpose of X?"
- "When should I use X vs Y?"

### Example Questions
- "Show me an example of X"
- "How do I X with options Y?"

**Generate 3-5 questions per major predicate:**
```json
{
  "predicate": "compile_enhanced_pipeline/3",
  "questions": [
    {"text": "How do I compile an enhanced pipeline to Python?", "type": "practical"},
    {"text": "What stages can I use in an enhanced pipeline?", "type": "concept"},
    {"text": "Show me a fan-out merge pipeline example", "type": "example"},
    {"text": "What options does compile_enhanced_pipeline accept?", "type": "practical"}
  ]
}
```

## Step 6: Extract Code Examples

Find examples in:
- Test predicates
- Comments with example queries
- Module documentation

```json
{
  "predicate": "compile_enhanced_pipeline/3",
  "examples": [
    {
      "source": "test_enhanced_pipeline_chaining/0",
      "code": "compile_enhanced_pipeline([\n    extract/1,\n    fan_out([validate/1, enrich/1]),\n    merge,\n    transform/1\n], [pipeline_name(test)], Code)",
      "description": "Fan-out merge pipeline"
    }
  ]
}
```

## Step 7: Map to Education Content

Find related education chapters:

```bash
# Search for module mentions in education
grep -r "python_target" ../education/UnifyWeaver_Education/ --include="*.md"
```

Create relations:
```json
{
  "relations": [
    {
      "from": "src/targets/python_target.pl:compile_enhanced_pipeline/3",
      "to": "book-05-ch03-enhanced-pipelines",
      "type": "implementation",
      "reason": "Source implements concepts taught in chapter"
    },
    {
      "from": "book-05-ch03-enhanced-pipelines",
      "to": "src/targets/python_target.pl:compile_enhanced_pipeline/3",
      "type": "example",
      "reason": "Chapter uses this predicate in examples"
    }
  ]
}
```

## Step 8: Map to Playbooks

Find related playbooks:

```bash
grep -r "python_target\|compile_enhanced" ../playbooks/ --include="*.md"
```

```json
{
  "playbook_relations": [
    {
      "source_predicate": "compile_enhanced_pipeline/3",
      "playbook": "playbooks/python_source_playbook.md",
      "relation": "used_by"
    }
  ]
}
```

## Step 9: Identify Dependencies

Map module dependencies:

```prolog
% From the source file
:- use_module('../core/service_validation').
:- use_module('../glue/network_glue').
```

```json
{
  "dependencies": {
    "imports": [
      {"module": "service_validation", "path": "src/unifyweaver/core/service_validation.pl"},
      {"module": "network_glue", "path": "src/unifyweaver/glue/network_glue.pl"}
    ],
    "imported_by": [
      {"module": "prolog_service_target", "path": "src/unifyweaver/targets/prolog_service_target.pl"}
    ]
  }
}
```

## Step 10: Generate Output JSONL

Write clusters for each major predicate:

```bash
mkdir -p "$OUTPUT_DIR"
```

```json
{
  "cluster_id": "python_target-compile_enhanced_pipeline",
  "source_file": "src/unifyweaver/targets/python_target.pl",
  "source_type": "source",
  "module": "python_target",
  "predicate": "compile_enhanced_pipeline/3",
  "answer": {
    "text": "compile_enhanced_pipeline/3 compiles a pipeline specification with enhanced stages (fan-out, merge, routing, filter) to Python code.",
    "signature": "compile_enhanced_pipeline(+Stages, +Options, -Code)",
    "arguments": [
      {"name": "Stages", "mode": "+", "type": "list"},
      {"name": "Options", "mode": "+", "type": "list"},
      {"name": "Code", "mode": "-", "type": "string"}
    ],
    "code_blocks": [
      {
        "language": "prolog",
        "code": "compile_enhanced_pipeline([\n    extract/1,\n    fan_out([validate/1, enrich/1]),\n    merge,\n    transform/1\n], [pipeline_name(my_pipeline)], Code).",
        "executable": true
      }
    ]
  },
  "questions": [
    {"text": "How do I compile an enhanced pipeline to Python?", "type": "practical"},
    {"text": "What stages can I use in compile_enhanced_pipeline?", "type": "concept"}
  ],
  "relations": [
    {"to": "book-05-ch03-enhanced-pipelines", "type": "implementation"}
  ],
  "dependencies": ["service_validation", "common_generator"]
}
```

## Step 11: Generate Module Summary

Create a summary cluster for the entire module:

```json
{
  "cluster_id": "python_target-module",
  "source_file": "src/unifyweaver/targets/python_target.pl",
  "source_type": "source",
  "is_module_summary": true,
  "answer": {
    "text": "The python_target module generates Python code from Prolog specifications. It supports basic pipelines, enhanced pipelines with fan-out/merge/routing, semantic search integration, and client-server service infrastructure.",
    "exports_count": 45,
    "categories": {
      "code_generation": 12,
      "helpers": 18,
      "infrastructure": 10,
      "testing": 5
    }
  },
  "questions": [
    {"text": "What does python_target do?", "type": "concept"},
    {"text": "How do I generate Python code with UnifyWeaver?", "type": "practical"},
    {"text": "What features does the Python target support?", "type": "concept"}
  ],
  "main_predicates": [
    "compile_python/3",
    "compile_enhanced_pipeline/3",
    "compile_python_semantic/3"
  ]
}
```

## Step 12: Verify Output

```bash
# Validate JSONL
python3 -c "
import json
with open('${OUTPUT_DIR}/$(basename $SOURCE_PATH .pl).jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
print('Validation complete')
"
```

## Success Criteria

- [ ] Module loads without errors
- [ ] All exported predicates documented
- [ ] 3+ questions per major predicate
- [ ] Relations to education content identified
- [ ] Dependencies mapped
- [ ] Valid JSONL output generated

## Example Execution

```
Input: src/unifyweaver/targets/python_target.pl
Output: training-data/source/targets/python_target.jsonl

Summary:
- Exported predicates: 45
- Clusters generated: 48 (45 predicates + 1 module + 2 category summaries)
- Questions generated: 156
- Relations to education: 12
- Relations to playbooks: 4
```
