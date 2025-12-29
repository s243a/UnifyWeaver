# Proposal: Hybrid Scoring for Bookmark Filing

## Status
Proposal (Nice-to-have, not yet needed)

## Summary

Add an optional hybrid scoring mode that combines **projected scores** (human-curated structure) with **raw embedding scores** (model-based semantics) and potentially **import/graph scores** (source structure) for bookmark filing recommendations.

## Background

The bookmark filing assistant uses a federated projection model to recommend folders for new bookmarks. During testing with the query "Hyphanet" (Wikipedia page about the anonymous P2P network), we discovered important characteristics of the two scoring approaches:

### Projected Score (Current Default)
- Based on Procrustes-transformed embeddings aligned to user's Pearltrees hierarchy
- Reflects **human-curated organization** and personal workflows
- For "Hyphanet": Ranked correct folder at **#2** (score: 0.176)

### Raw Embedding Score
- Direct cosine similarity from the embedding model (Nomic v1.5)
- Reflects **model's semantic understanding**
- For "Hyphanet": Ranked correct folder at **#38** (score: 0.498)

The projected score dramatically improved ranking (#38 → #2) by routing the query to the correct semantic domain (privacy/darknet tools) despite the lower absolute score.

## Why Absolute Scores Differ

The low projected score (0.176 vs 0.498 raw) is expected, not a bug:

1. **Cluster density**: The privacy/security cluster contains 120 semantically similar items
2. **Procrustes averaging**: The W matrix is a single rotation learned across all cluster items
3. **Score dilution**: In dense regions, cosine similarities are distributed across many neighbors

The ranking remains correct, which is what matters for filing.

## Three Complementary Signals

| Score Type | Source | Strength | Weakness |
|------------|--------|----------|----------|
| **Projected** | User's Pearltrees hierarchy | Consistency with existing organization | May perpetuate past misfiles |
| **Raw** | Embedding model | Objective semantic similarity | Ignores user's mental model |
| **Import/Graph** | Source structure | Corrects misfilings using natural structure | May not reflect user's custom logic |

### Import Scores (Graph Signal)
User feedback indicates that **import scores** (derived from the original graph structure) could be valuable for identifying **misfiled items**. While projected scores learn the user's current organization (including mistakes), import scores might reflect the *intended* or *natural* structure, helping to flag or correct items that drifted into the wrong folder.

### Use Cases

1. **Filing (projected dominates)**: User wants bookmarks where they would look for them, consistent with existing structure

2. **Quality check (raw as sanity check)**: If projected and raw strongly disagree, the original filing may have been a mistake

3. **Correction (Graph/Import)**: Using original graph connections to propose moves for misplaced items.

4. **Hybrid ranking**: Weight signals for edge cases

## Proposed Implementation

### Option A: Weighted Blend (Configurable)
```python
final_score = alpha * projected_score + (1 - alpha) * raw_score
# Default: alpha = 1.0 (current behavior)
# User can adjust for their needs
```

### Option B: Disagreement Flagging
```python
if abs(projected_rank - raw_rank) > threshold:
    flag_for_review(bookmark, projected_folder, raw_folder)
```

## Rationale for Default Behavior

**Projected score should remain the default** because:

1. Filing is about **user retrieval** - where would *you* look for this later?
2. Consistency with existing structure reduces cognitive load
3. User's hierarchy encodes domain knowledge the model lacks
4. The projection already incorporates the model's embeddings, transformed to align with user structure

## Recommendation

1. **No immediate changes needed** - current system works correctly
2. **Future enhancement**: Add `--compare-raw` flag for diagnostic purposes
3. **Consider later**: Weighted blend option if use cases emerge
4. **Research**: Investigate extracting "Import Scores" from the RDF graph for a new correctness signal.

## Test Case Reference

Query: "Hyphanet" (Wikipedia page for anonymous P2P network, formerly Freenet)

| Method | Rank | Score | Notes |
|--------|------|-------|-------|
| Raw embedding | #38 | 0.498 | Confused by "Hyp-" prefix matches |
| Projected | #2 | 0.176 | Correctly routed to privacy/darknet domain |

Top raw matches (incorrect):
- #1 Hysteresis (0.597)
- #2 Hardy Space (0.561)
- #3 holomorphic functions (0.544)

These are unrelated math/physics topics that share surface-level embedding similarity.

## Files Involved

- `scripts/infer_pearltrees_federated.py` - Inference engine
- `scripts/bookmark_filing_assistant.py` - User-facing assistant
- `src/unifyweaver/targets/python_runtime/minimal_transform.py` - Procrustes implementation
- `models/pearltrees_federated_single.pkl` - Trained model (51 clusters)
