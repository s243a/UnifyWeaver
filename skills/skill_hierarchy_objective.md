# Skill: Hierarchy Objective Function

Evaluate and optimize hierarchical tree structures using the objective function J = D/(1+H).

## When to Use

- User asks "how do I measure if my hierarchy is good?"
- User wants to understand MST clustering quality
- User asks about "entropy" or "distance" in hierarchies
- User wants to tune hierarchy construction parameters
- User needs to compare different tree structures
- User asks "what makes a good folder organization?"

## Theory: The J = D/(1+H) Objective

Good hierarchies have two properties:
1. **Tight clusters** - nodes close to their parent (low D)
2. **Informative splits** - each level adds meaning (high H)

The objective function balances these:

```
J = D / (1 + H)
```

Where:
- **D (Distance)** = average semantic distance from nodes to parents
- **H (Entropy gain)** = information gained at each level split

**Lower J = better hierarchy.**

### What D Measures

D captures semantic coherence - how tightly nodes cluster under their parents:
- Low D: Children are semantically similar to their parent
- High D: Children are scattered, loosely related

Normalized by depth: deeper nodes are expected to be more specific (tighter).

### What H Measures

H captures hierarchy informativeness - how much each split tells you:
- High H: Clear separation between sibling groups (informative)
- Low H: Siblings blend together (uninformative)

Computed as the ratio of between-cluster to within-cluster variance (Fisher criterion).

## Quick Start

### Evaluate an Existing Hierarchy

```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree hierarchy.json \
  --embeddings embeddings.npy
```

### Build a J-Guided Tree (Alternative to MST)

```python
from scripts.mindmap.hierarchy_objective import build_j_guided_tree

tree, stats, suggestions = build_j_guided_tree(
    embeddings,
    titles=folder_names,
    verbose=True
)
print(f"Objective J: {stats.objective:.4f}")
```

## Commands

### Basic Evaluation
```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree "HIERARCHY.json" \
  --embeddings "EMBEDDINGS.npy"
```

### With Depth Normalization
```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree "HIERARCHY.json" \
  --embeddings "EMBEDDINGS.npy" \
  --depth-decay 0.5
```

### Using BERT-Based Entropy (More Accurate, Slower)
```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree "HIERARCHY.json" \
  --embeddings "EMBEDDINGS.npy" \
  --entropy-source logits \
  --entropy-model answerdotai/ModernBERT-base
```

### Save Results to JSON
```bash
python3 scripts/mindmap/hierarchy_objective.py \
  --tree "HIERARCHY.json" \
  --embeddings "EMBEDDINGS.npy" \
  --output stats.json
```

## Entropy Source Options

| Source | Speed | Accuracy | Requirements |
|--------|-------|----------|--------------|
| `fisher` | Fast | Good | Embeddings only (default) |
| `logits` | Slow | Best | Text + transformer model |

**Fisher** (default): Uses geometric proxy (between/within cluster variance ratio). Works with any embedding model.

**Logits**: Computes actual Shannon entropy from transformer output. Requires text for each node. More theoretically pure but slower.

```bash
# Fisher (fast, embeddings only)
--entropy-source fisher

# Logits (accurate, needs text)
--entropy-source logits \
--entropy-model answerdotai/ModernBERT-base
```

## Distance Metrics

| Metric | Range | Best For |
|--------|-------|----------|
| `euclidean` | [0, 2] | Default, good resolution |
| `angular` | [0, π] | Linear in angle |
| `cosine` | [0, 2] | Traditional, poor small-angle resolution |
| `sqeuclidean` | [0, 4] | Avoids sqrt, fast |

```python
from scripts.mindmap.hierarchy_objective import JGuidedTreeBuilder

builder = JGuidedTreeBuilder(
    embeddings,
    distance_metric='euclidean'  # or 'angular', 'cosine', 'sqeuclidean'
)
```

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--depth-decay` | 0.5 | Higher = stricter at deep levels |
| `--smoothing` | dirichlet | Probability smoothing method |
| `--alpha` | 1.0 | Smoothing strength |
| `--combine` | product | How to combine D and H |

### Depth Decay

Controls how distance penalties scale with depth:

```
Expected_distance(depth) = base * exp(-λ * depth)
```

- `λ = 0.5`: Moderate (default)
- `λ = 1.0`: Strict - deep nodes must be very tight
- `λ = 0.2`: Lenient - allows looser deep clusters

## Interpreting Results

```
Hierarchy Statistics:
  Nodes: 156
  Levels: 5
  Semantic Distance D: 0.2341 (raw: 0.1872)
  Entropy Gain H: 1.4532 (raw: 1.4532)
  Objective J: 0.0955

Per-level stats:
  Level 0: D=0.0000, H=1.2341, n=1
  Level 1: D=0.1523, H=1.5678, n=8
  Level 2: D=0.2145, H=1.4123, n=24
  Level 3: D=0.2876, H=1.3456, n=67
  Level 4: D=0.3012, H=0.0000, n=56
```

**Good signs:**
- J < 0.2: Well-structured hierarchy
- H increases with depth (more specific concepts deeper)
- D roughly constant or slightly increasing

**Warning signs:**
- J > 0.5: Poor hierarchy structure
- H decreasing with depth: Splits become less informative
- D jumping suddenly: Incoherent parent-child relationships

## J-Guided Tree Construction

Alternative to MST that directly optimizes J during construction:

```python
from scripts.mindmap.hierarchy_objective import build_j_guided_tree
import numpy as np

# Load your embeddings
embeddings = np.load("embeddings.npy")
titles = ["Machine Learning", "Deep Learning", "NLP", ...]

# Build tree optimizing J at each attachment
tree, stats, suggestions = build_j_guided_tree(
    embeddings,
    titles=titles,
    use_bert_entropy=False,  # Use Fisher (fast)
    intermediate_threshold=0.5,  # Suggest intermediate nodes
    verbose=True
)

# Results
print(f"Objective J: {stats.objective:.4f}")
print(f"Depth-surprisal correlation: {stats.depth_surprisal_correlation:.4f}")

# Suggestions for intermediate nodes (entropy residual too high)
for sugg in suggestions:
    print(f"Consider adding category between '{sugg['parent_text']}' and '{sugg['child_text']}'")
```

## Depth-Surprisal Correlation

A key diagnostic: does depth correlate with surprisal (-log probability)?

```python
from scripts.mindmap.hierarchy_objective import JGuidedTreeBuilder

builder = JGuidedTreeBuilder(embeddings, titles=titles)
tree = builder.build()

corr, slope = builder.get_depth_surprisal_correlation()
print(f"Correlation: {corr:.3f}")  # Should be positive
print(f"Slope: {slope:.3f}")       # Higher = better depth-probability alignment
```

**Interpretation:**
- `corr > 0.5`: Good - general concepts at root, specific at leaves
- `corr < 0.2`: Poor - hierarchy doesn't match concept generality
- `slope > 0`: Depth properly tracks information content

## Related

**Parent Skill:**
- `skill_ml_tools.md` - ML tools sub-master

**Sibling Skills:**
- `skill_train_model.md` - Training models with hierarchy-aware clustering
- `skill_embedding_models.md` - Model selection
- `skill_semantic_inference.md` - Running inference
- `skill_density_explorer.md` - Visualization

**Other Skills:**
- `skill_mst_folder_grouping.md` - MST-based hierarchy construction
- `skill_folder_suggestion.md` - Using hierarchies for folder suggestion

**Documentation:**
- `docs/design/MST_IMPROVEMENTS_PROPOSAL.md` - J-guided tree improvements
- `docs/design/SKILL_COVERAGE_GAPS.md` - Coverage analysis

**Education (in `education/` subfolder):**
- `book-14-ai-training/01_introduction.md` - Embedding and projection concepts
- `book-13-semantic-search/07_density_scoring.md` - Density and clustering foundations

**Code:**
- `scripts/mindmap/hierarchy_objective.py` - Main implementation
- `scripts/mindmap/build_mst_index.py` - MST tree construction
