# Semantic Source Examples

This file contains executable records for the semantic_source playbook.

**Note**: These examples require an embedding backend to be installed. The examples show the structure but may need backend setup to run.

## Example 1: Basic Semantic Search

Query: `semantic_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Semantic Source Demo: Basic Search ==="
echo "Note: Requires python_onnx backend with numpy and onnxruntime"

mkdir -p tmp
cat > tmp/semantic_basic.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Declare semantic source with python_onnx backend
:- dynamic_source(find_papers/3, semantic, [
    vector_store('tmp/papers_vectors.json'),
    embedding_backend(python_onnx),
    similarity_threshold(0.7),
    top_k(5)
]).

main :-
    format("Compiling semantic source: find_papers/3~n"),
    compile_dynamic_source(find_papers/3, [target(bash)], 'tmp/find_papers.sh'),
    format("Generated: tmp/find_papers.sh~n"),
    format("~nNote: To run, you need:~n"),
    format("  1. Python 3.8+ with numpy and onnxruntime~n"),
    format("  2. Vector store file with embeddings~n"),
    format("  3. ONNX model for embeddings~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/semantic_basic.pl
echo "Success: Semantic source compiled (backend setup required to execute)"
```

## Example 2: Threshold Filtering

Query: `semantic_threshold`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Semantic Source Demo: Threshold ==="

cat > tmp/semantic_threshold.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% High similarity threshold
:- dynamic_source(similar_docs/2, semantic, [
    vector_store('tmp/docs_vectors.json'),
    embedding_backend(python_onnx),
    similarity_threshold(0.75)
]).

main :-
    format("Compiling semantic source: similar_docs/2~n"),
    compile_dynamic_source(similar_docs/2, [target(bash)], 'tmp/similar_docs.sh'),
    format("Generated: tmp/similar_docs.sh~n"),
    format("Threshold set to 0.75 (only high-similarity matches)~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/semantic_threshold.pl
echo "Success: Threshold-based semantic source compiled"
```

## Example 3: Top-K Results

Query: `semantic_topk`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Semantic Source Demo: Top-K ==="

cat > tmp/semantic_topk.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Return top 5 most similar
:- dynamic_source(top_matches/3, semantic, [
    vector_store('tmp/vectors.json'),
    embedding_backend(python_onnx),
    top_k(5),
    similarity_metric(cosine)
]).

main :-
    format("Compiling semantic source: top_matches/3~n"),
    compile_dynamic_source(top_matches/3, [target(bash)], 'tmp/top_matches.sh'),
    format("Generated: tmp/top_matches.sh~n"),
    format("Will return top 5 matches using cosine similarity~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/semantic_topk.pl
echo "Success: Top-K semantic source compiled"
```

## Example 4: Check Plugin Info

Query: `semantic_info`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Semantic Source Info ==="

swipl -g "
    use_module('src/unifyweaver/sources/semantic_source'),
    semantic_source:source_info(Info),
    Info = info(Name, Version, Desc, _, _, _, _),
    format('Name: ~w~n', [Name]),
    format('Version: ~w~n', [Version]),
    format('Description: ~w~n', [Desc]),
    halt
"

echo "Success: Semantic source plugin loaded"
```
