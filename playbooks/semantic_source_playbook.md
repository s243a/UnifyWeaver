# Playbook: Semantic/Vector Search Source

## Audience
This playbook is a high-level guide for coding agents. It demonstrates semantic search using embedding vectors and similarity through UnifyWeaver's semantic_source plugin.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "semantic_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use semantic source"


## Workflow Overview
Use UnifyWeaver for semantic search:
1. Declare a semantic source with vector store and embedding backend
2. Configure similarity threshold and top-k parameters
3. UnifyWeaver generates code that queries vectors and returns similar documents

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** – `semantic_basic`, `semantic_threshold`, `semantic_topk`, `semantic_info` in `playbooks/examples_library/semantic_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/semantic_source.pl`
3. **Extraction Tool** – `scripts/extract_records.pl`

## Execution Guidance

**IMPORTANT**: Records contain **BASH SCRIPTS**. Extract and run with `bash`, not `swipl`.

**NOTE**: These examples demonstrate code generation but require embedding backend setup (python_onnx, go_service, rust_candle, or csharp_native) to execute fully.

### Example 1: Basic Semantic Search

**Step 1: Navigate and extract**
```bash
cd /path/to/UnifyWeaver
perl scripts/extract_records.pl playbooks/examples_library/semantic_source_examples.md \
    semantic_basic > tmp/semantic_basic.sh
chmod +x tmp/semantic_basic.sh
bash tmp/semantic_basic.sh
```

**Expected Output:**
```
Compiling semantic source: find_papers/3
Generated: tmp/find_papers.sh
Note: To run, you need:
  1. Python 3.8+ with numpy and onnxruntime
  2. Vector store file with embeddings
  3. ONNX model for embeddings
Success: Semantic source compiled (backend setup required to execute)
```

### Example 2: Similarity Threshold

Extract and run `semantic_threshold` query for high-similarity filtering.

### Example 3: Top-K Results

Extract and run `semantic_topk` query for retrieving top N matches.

### Example 4: Plugin Info

Extract and run `semantic_info` query to check plugin metadata.

## Expected Outcome
- Semantic sources compile successfully
- Code generation verified (execution requires backend)
- Configuration options understood
- Exit code 0

## Citations
[1] playbooks/examples_library/semantic_source_examples.md
[2] src/unifyweaver/sources/semantic_source.pl
[3] scripts/extract_records.pl
