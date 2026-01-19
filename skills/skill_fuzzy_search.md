# Skill: Fuzzy Search

Combine scores from multiple sources using fuzzy logic operations and fusion techniques.

## When to Use

- User asks "how do I combine search scores?"
- User wants to blend semantic and lexical search
- User asks about weighted score combination
- User needs rank fusion (RRF)
- User wants to implement soft matching
- User asks about f_and, f_or, or fuzzy logic DSL

## Quick Start

### Basic Score Blending

```prolog
:- use_module('src/unifyweaver/fuzzy/fuzzy').

% Blend two score lists (70% semantic, 30% keyword)
combined_search(Query, Items, Combined) :-
    semantic_search(Query, SemanticScores),
    keyword_search(Query, KeywordScores),
    blend_scores(0.7, SemanticScores, KeywordScores, Combined).
```

### Fuzzy AND/OR

```prolog
% All terms must match (product of scores)
strict_match(Terms, Score) :-
    f_and(Terms, Score).

% Any term can match (probabilistic sum)
flexible_match(Terms, Score) :-
    f_or(Terms, Score).
```

## Fuzzy Logic Operations

### Core Operations

| Operation | Formula | Use Case |
|-----------|---------|----------|
| `f_and` | w1*t1 * w2*t2 * ... | All terms must match |
| `f_or` | 1 - (1-w1*t1)(1-w2*t2)... | Any term can match |
| `f_dist_or` | Distributed OR | Base score into each term |
| `f_union` | Base * OR result | Non-distributed combination |
| `f_not` | 1 - Score | Negation/complement |

### Weighted Terms

Use `w(Term, Weight)` to assign importance:

```prolog
% "bash" is more important than "shell"
search_expr(Expr) :-
    Expr = f_and([w(bash, 0.9), w(shell, 0.5)]).

% Alternative colon syntax
search_expr2(Expr) :-
    Expr = f_and([bash:0.9, shell:0.5]).
```

### Evaluating Expressions

```prolog
% With explicit term scores
?- eval_fuzzy_expr(
       f_and([w(bash, 0.9), w(shell, 0.5)]),
       [bash-0.8, shell-0.6],  % Term-Score pairs
       Result
   ).
Result = 0.216  % 0.9*0.8 * 0.5*0.6
```

### Distributed vs Non-Distributed OR

```prolog
% f_dist_or: Base distributed into each term
% Formula: 1 - (1-Base*w1*t1)(1-Base*w2*t2)...
f_dist_or(0.5, [bash, shell], Result).

% f_union: Base multiplied by OR result
% Formula: Base * f_or(Terms)
f_union(0.5, [bash, shell], Result).
```

## Score Fusion Techniques

### Score Fusion (Weighted Blend)

Combines normalized scores with weights:

```prolog
% 70% model A, 30% model B
blend_scores(0.7, ScoresA, ScoresB, Combined).
```

```python
# Python equivalent
def blend_scores(alpha, scores_a, scores_b):
    result = {}
    for item in scores_a:
        s1 = scores_a.get(item, 0)
        s2 = scores_b.get(item, 0)
        result[item] = alpha * s1 + (1 - alpha) * s2
    return result
```

### Rank Fusion (RRF)

Reciprocal Rank Fusion combines rankings, not scores:

```python
# From scripts/experiment_ensemble_blend.py
def rrf_blend_scores(scores_list, k=60):
    """
    Reciprocal Rank Fusion.
    RRF(d) = sum(1 / (k + rank_i(d)))
    """
    rrf_scores = {}
    for scores in scores_list:
        # Sort by score descending to get ranks
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        for rank, (item, _) in enumerate(ranked, start=1):
            rrf_scores[item] = rrf_scores.get(item, 0) + 1 / (k + rank)
    return rrf_scores
```

**When to use RRF:**
- Combining results from different models with incompatible score scales
- Ensemble of diverse retrieval methods
- When relative ranking matters more than absolute scores

### Score Multiplication

Element-wise multiplication for combining independent signals:

```prolog
% Combine relevance and recency
multiply_scores(RelevanceScores, RecencyScores, Combined).
```

## Boolean Filters

Combine fuzzy scores with hard boolean filters:

```prolog
% Boolean AND (all conditions must be true)
b_and([is_type(pearl), has_account(main)], Result).

% Boolean OR (any condition can be true)
b_or([has_tag(important), has_parent(root)], Result).

% Apply filter to scored items
apply_filter(ScoredItems, is_type(pearl), Filtered).
```

### Available Predicates

| Predicate | Description |
|-----------|-------------|
| `is_type(T)` | Item is of type T |
| `has_account(A)` | Item belongs to account A |
| `has_parent(P)` | Item's parent is P |
| `in_subtree(Root)` | Item is under Root |
| `has_tag(Tag)` | Item has tag |
| `child_of(P)` | Direct child of P |
| `descendant_of(P)` | Any descendant of P |
| `has_depth(D)` | Item at depth D |
| `depth_between(Min, Max)` | Depth in range |

## Score Boosting

Apply multiplicative boosts based on item properties:

```prolog
% Boost recent items
apply_boost(ScoredItems, recency_boost, Boosted).

% Custom boost expression
apply_boost(
    ScoredItems,
    f_or([w(featured, 1.5), w(popular, 1.2)]),
    Boosted
).
```

## Top-K Selection

Get the highest-scoring items:

```prolog
% Get top 10 results
top_k(ScoredItems, 10, TopResults).
```

## Complete Pipeline Example

```prolog
:- use_module('src/unifyweaver/fuzzy/fuzzy').

% Full search pipeline
search_pipeline(Query, TopResults) :-
    % 1. Get scores from multiple sources
    semantic_search(Query, SemanticScores),
    keyword_search(Query, KeywordScores),

    % 2. Blend scores (60% semantic, 40% keyword)
    blend_scores(0.6, SemanticScores, KeywordScores, Blended),

    % 3. Apply type filter
    apply_filter(Blended, is_type(pearl), Filtered),

    % 4. Boost by recency
    apply_boost(Filtered, recency_factor, Boosted),

    % 5. Get top 20
    sort(2, @>=, Boosted, Sorted),
    top_k(Sorted, 20, TopResults).
```

## Python Ensemble Search

For more complex ensemble scenarios, use the Python experiment script:

```python
# scripts/experiment_ensemble_blend.py
from experiment_ensemble_blend import blend_scores, rrf_blend_scores

# Load scores from different models
bge_scores = load_scores("bge_results.json")
minilm_scores = load_scores("minilm_results.json")
nomic_scores = load_scores("nomic_results.json")

# Score fusion with weights
blended = blend_scores([
    (bge_scores, 0.5),
    (minilm_scores, 0.3),
    (nomic_scores, 0.2)
])

# Or rank fusion
rrf_combined = rrf_blend_scores([bge_scores, minilm_scores, nomic_scores])
```

## Related

**Parent Skill:**
- `skill_aggregation_patterns.md` - Overview of aggregation approaches

**Other Skills:**
- `skill_semantic_search.md` - Semantic search basics
- `skill_train_model.md` - Training embedding models
- `skill_embedding_models.md` - Model selection

**Documentation:**
- `education/book-13-semantic-search/` - Semantic search concepts

**Code:**
- `src/unifyweaver/fuzzy/fuzzy.pl` - Main fuzzy module
- `src/unifyweaver/fuzzy/core.pl` - Core operations
- `src/unifyweaver/fuzzy/eval.pl` - Evaluation engine
- `src/unifyweaver/fuzzy/boolean.pl` - Boolean filters
- `src/unifyweaver/fuzzy/predicates.pl` - Filter predicates
- `scripts/experiment_ensemble_blend.py` - Python ensemble experiments
