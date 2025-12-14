<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# Semantic Search Usage Guide

This guide covers how to use UnifyWeaver's semantic search capabilities for finding relevant playbook examples and documentation.

## Overview

The semantic search system uses multi-head LDA projection to improve retrieval accuracy. It projects query embeddings into answer space, making it easier to find semantically similar content even when the wording differs.

## Prolog API

### Basic Usage

```prolog
% Load the semantic search module
:- use_module('src/unifyweaver/runtime/semantic_search').

% Find top 5 examples matching a query
?- find_examples("How do I read a CSV file?", 5, Examples).
Examples = [[answer_id(3), score(0.89), text("..."), ...], ...].

% Search with minimum score threshold
?- find_examples("database query", 10, [min_score(0.7)], Examples).
```

### Available Predicates

#### find_examples/3

```prolog
find_examples(+Query, +TopK, -Examples)
```

Find playbook examples matching the query.

- **Query**: Natural language query (atom or string)
- **TopK**: Number of results to return (integer)
- **Examples**: List of result dicts with `answer_id`, `score`, `text`, `record_id`, `source_file`

#### find_examples/4

```prolog
find_examples(+Query, +TopK, +Options, -Examples)
```

Find examples with options.

**Options:**
- `component(Name)` - Use a specific search component instance
- `use_projection(Bool)` - Enable/disable LDA projection (default: true)
- `min_score(Score)` - Filter results below this similarity threshold

#### semantic_search/3, semantic_search/4

Lower-level search predicates with the same signature as `find_examples`.

### Search Modes

The system supports three search modes:

1. **Direct similarity** - Standard cosine similarity without projection
2. **Multi-head projection** - Per-cluster routing with softmax (recommended)
3. **Global projection** - Single W matrix projection

Mode is determined by component configuration:

```prolog
% Multi-head mode (default when mh_projection_id is set)
Config = [db_path('lda.db'), mh_projection_id(2)].

% Global projection mode
Config = [db_path('lda.db'), projection_id(1)].

% Direct mode (no projection)
Config = [db_path('lda.db')].
% Or override with option:
find_examples("query", 5, [use_projection(false)], Results).
```

### Component Initialization

```prolog
% Initialize a named search component
semantic_search:init_component(my_search, [
    db_path('playbooks/lda-training-data/lda.db'),
    model_name('all-MiniLM-L6-v2'),
    mh_projection_id(2)
]).

% Use the named component
find_examples("query", 5, [component(my_search)], Results).

% Shutdown when done
semantic_search:shutdown_component(my_search).
```

## Go API

The Go projection module provides native multi-head LDA projection without Python overhead.

### Loading a Projection

```go
import "unifyweaver/targets/go_runtime/projection"

// Load from directory with centroid_*.npy and answer_emb_*.npy files
mh, err := projection.LoadMultiHead(projection.Config{
    DataDir:     "playbooks/lda-training-data/trained/mh_projection_2/",
    Temperature: 0.1,
})
if err != nil {
    log.Fatal(err)
}

// Or load specific files
mh, err := projection.LoadMultiHead(projection.Config{
    Temperature: 0.1,
    HeadFiles: map[int]projection.HeadFilePair{
        1: {CentroidPath: "centroid_1.npy", AnswerEmbPath: "answer_emb_1.npy"},
        2: {CentroidPath: "centroid_2.npy", AnswerEmbPath: "answer_emb_2.npy"},
    },
})
```

### Projecting Queries

```go
// Project a query embedding
queryEmb := []float32{...} // 384-dimensional embedding
projected, err := mh.Project(queryEmb)
if err != nil {
    log.Fatal(err)
}

// Project with routing weights (for debugging/analysis)
projected, weights, err := mh.ProjectWithWeights(queryEmb)
// weights maps cluster ID to softmax weight
fmt.Printf("Routing: %v\n", weights) // e.g., {1: 0.85, 2: 0.10, 3: 0.05}
```

### Search with Projection

```go
import (
    "unifyweaver/targets/go_runtime/projection"
    "unifyweaver/targets/go_runtime/search"
)

// Load projection
mh, _ := projection.LoadMultiHead(projection.Config{
    DataDir:     "path/to/heads/",
    Temperature: 0.1,
})

// Search with projection
results, err := search.SearchWithOptions(store, queryVec, 10, search.SearchOptions{
    Projection:            mh,
    UseProjection:         true,
    IncludeRoutingWeights: true,
})

for _, r := range results {
    fmt.Printf("%s: %.3f (routing: %v)\n", r.ID, r.Score, r.RoutingWeights)
}
```

## Temperature Tuning

The softmax temperature controls routing sharpness:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.1 | Sharp routing (winner-take-all) | Production retrieval |
| 0.5 | Moderate blending | Exploratory search |
| 1.0 | Diffuse routing | Diverse results |

Lower temperatures (0.1) typically provide better recall for precise queries.

## Example: Playbook Assistant

```prolog
% Find relevant examples for a user task
suggest_examples(UserTask, Suggestions) :-
    find_examples(UserTask, 3, [min_score(0.6)], Results),
    maplist(format_suggestion, Results, Suggestions).

format_suggestion(Result, Suggestion) :-
    member(text(Text), Result),
    member(score(Score), Result),
    member(source_file(File), Result),
    format(string(Suggestion), "~w (score: ~2f, from: ~w)", [Text, Score, File]).

% Usage:
?- suggest_examples("How to parse JSON in Python?", S).
```

## Testing

```bash
# Run Prolog tests
swipl tests/core/test_semantic_search.pl

# Run Go tests
cd src/unifyweaver/targets/go_runtime && go test ./projection/... -v
```

## See Also

- [SEMANTIC_PROJECTION_LDA.md](../proposals/SEMANTIC_PROJECTION_LDA.md) - Theory
- [MULTI_HEAD_PROJECTION_THEORY.md](../proposals/MULTI_HEAD_PROJECTION_THEORY.md) - Multi-head design
- [TODO_LDA_PROJECTION.md](../TODO_LDA_PROJECTION.md) - Implementation status
