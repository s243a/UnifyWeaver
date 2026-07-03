# Mindmap LINEAGE → filing training data (design + methodology)

Status: lineage sampler built (`gen_mindmap_lineage.py`); negative-generation + judge-scoring specified here,
prototype next. Non-enwiki data from SimpleMind `.smmx` maps; prep for fusion with Pearltrees once harvested.

## 0. Why mindmaps, why lineage first

The SimpleMind maps are the user's own systems-theory domain — structurally and topically the closest proxy we
have to the Pearltrees filing target, and **not** present in enwiki. We build **LINEAGE first, then the 70/30
split on top of it** (directional/superposition — see `feedback_enwiki_sampling_strategy`). Lineage is
well-defined here because a mindmap node hangs off exactly one structural parent — the same property that made us
choose single-path LINEAGE for Pearltrees (`project_mu_cosine_path_multipath_result`).

## 1. Positives — complete, and graded by distance

The dataset is small (491 lineage chains over 6 maps), so we take **complete coverage of positives**: every
(descendant, ancestor) pair along each materialized path. They are **not binary** — μ is **graded by lineage
distance**: direct parent = strong, grandparent = weaker, great-grandparent = weaker still. The depth gives
calibrated positives for free.

Materialized paths are cleaned (`gen_mindmap_lineage.py`): the shared sentinel `Root Node` (in folder `root`,
a generic placeholder to which all maps link up) is dropped, `via link` nav nodes filtered, cross-map join
echoes (consecutive duplicate titles) collapsed. Privacy is scrubbed upstream by `parse_smmx.py`
(`privacy.py::is_private_title`: a "private" node drops itself + subtree, a private root drops the whole map).

## 2. Negatives — sampled, typed, graded (not enumerated)

The negative complement is ~O(N²) and would swamp the positives, so we **sample** (enwiki uses 3:1). Three types
in increasing value:

1. **Directional-reverse** (ancestor | descendant) — free; drives the directional margin μ(desc|anc) ≫ μ(anc|desc).
2. **Cross-branch** (different subtrees) — the easy "unrelated" tail.
3. **Hard negatives — siblings / cousins** (same parent/grandparent, *not* ancestor-descendant). The valuable
   ones: they teach μ that *same branch ≠ lineage*, the confusion a hierarchy model must resolve.

**Key subtlety — no pair is truly unrelated.** Because every node shares the global `Root Node`, even
cross-domain pairs share distant ancestry. So a "negative" means *"not a direct lineage link,"* and μ should
**fall off gradually with tree-distance**, not snap to 0. This graded fall-off is the signal we want, and it is
why hard sibling/cousin negatives matter more than random cross-map ones (the shared root already makes those
mildly positive).

### 2a. Candidate generation — substitution, e5-stratified

Generate hard negatives by **substituting** a node in a true path with an alternative (an alternate parent/path
— exactly the filing question "could it go here instead?"). **Permutation** (scrambling path order) mostly
yields obviously-broken hierarchies — kept only as a cheap easy-negative floor.

Control hardness with **e5 distance of the substituted node vs the original**: e5-near = sibling/cousin = hard
negative; e5-far = easy. **e5 distance is used to *stratify candidates*, not as μ** — semantic distance ≠ filing
plausibility (a sibling can be e5-close yet filing-wrong; a valid alternate parent can be e5-far). The μ comes
from a judge (§3).

## 3. Judge scoring

Remove the last hop of a path → (path-minus-leaf = folder, leaf = item to file). Each example is thus **actual
filing training data** — more target-aligned than raw lineage pairs, and buildable now without the harvester.
Present the judge ≥2 candidate filings (2 = cheapest; 3–4 = better-shaped curve), scored **pointwise and
independently — μ does NOT sum to 1** (matches the §14 prompt and the model's per-pair μ).

### 3a. LLM judge (codex `gpt-5.5-low`, or Haiku)

- **Max-normalize to the model's top pick = 1**, then a scoring curve: **linear first** (preserves the graded
  middle); a **temperature-controlled sigmoid** only if raw scores come back too bunched to separate hard-from-easy
  (a *steep* sigmoid discards the grading — avoid).
- **Multi-parent guard:** anchoring to the model's top (not the ground-truth) honors genuine second parents
  (Pearltrees supercategories, cross-map links) — a valid substituted path *should* be allowed near 1. But keep
  the **ground-truth as a sanity floor**: if the true filing scores *low*, flag it (judge error or a genuinely
  ambiguous node), don't silently train on it.
- Tag `judge=gpt-5.5-low` (its own calibration row; see `DESIGN_calibrated_judges.md`, `JUDGES` in
  `mu_attention.py`). Cost/latency: `gpt-5.5-low` ≈ 5 s/pair, non-Anthropic.

### 3b. Graph judge (nearly free) — `μ = decay( distance-to-truth )`

Score a candidate by its **graph distance to the true filing**, decayed to μ (truth at distance 0 → **1**;
1 hop = sibling/uncle → high; far → low). This is the **structural** analogue of the e5 idea, and unlike e5 it
maps to μ **directly and honestly** — distance-to-truth *is* a plausibility. Tag `judge=graph`
(`JUDGES["graph"] = 1`, already defined).

**Whole-lineage, not just the immediate parent.** A bare parent-to-parent hop is myopic: two candidates can be
equidistant from the true parent yet sit in very different lineages (`Math/Calculus/X` vs `Math/Analysis/X` — one
leaf-hop apart but nearly the same folder path — versus one hop into an unrelated subtree). So the graph-μ
accounts for the **full materialized-path prefix**, via two complementary whole-lineage factors:

- **LCA depth (structural):** how deep the *shared* prefix runs (shares-only-Root ⇒ ~0) — a topological measure
  of common ancestry.
- **LCA-depth carries the whole-lineage *structural* weight** in `μ_graph` (below). e5-of-prefix does **not**
  enter `μ_graph` — multiplying it against LCA-depth double-counts, since a deep shared prefix is *already*
  e5-similar (review, critical #3). Instead **e5 is routed to a separate `lineage-rank` operator** (§3c), where
  it is a *ranking* signal (cross-entropy), not a regression factor.

So `μ_graph = decay(hops) · lca_depth_frac` (pure structural; decay variant per the table below) — the regression
target of the **`lineage`** operator. The semantic e5 signal goes to **`lineage-rank`** (§3c).

**Decay choice — alternatives (we pick a default; these are the project's existing metrics).** The decay is
exactly a flux/cost-function over the tree; see `docs/design/COST_FUNCTION_PHILOSOPHY.md` and
`docs/design/SCAN_STRATEGY_SPECIFICATION.md`:

| variant | form | note |
|---|---|---|
| **hop / shortest-path** (our default) | `μ = γ^d`, or `1/(1+d)` | simplest; no flux interpretation |
| exponential-decay flux | `Σ_paths (decay/branching)^|p|` | our `exp(-λd)` is the single-path case |
| power-law / inverse-radial | `1/r^N` (log-flux `= −N·log r`) | N-dim source model; heavier tail |
| PPR / Markov walk | stationary dist. biased by the node seed | probabilistic dual; handles multi-path |
| LCA depth | depth of lowest common ancestor | tree-native; "shares-only-Root ⇒ ~0" |

Default: **hop-distance with geometric/exp decay + LCA depth** (cheap, tree-native). The power-law and PPR
variants are the natural upgrades if the graded fall-off needs a heavier tail or multi-path mass — already
specified in `COST_FUNCTION_PHILOSOPHY.md` and left as alternatives.

### 3c. Two operators — `lineage` (MSE) and `lineage-rank` (CE)

How e5 enters determines the loss: as a **scoring factor** it implies regression (error/MSE); as a **ranking**
signal it implies cross-entropy. Rather than choose, split into two operators (two `op_emb` entries, routed
per-operator like SYM/HIER/ELEM), each with its own loss — which also keeps e5 out of the regressed magnitude:

- **`lineage` → MSE / regression.** Target = the pure structural `μ_graph = decay(hops) · lca_depth_frac`.
  Learns the graded *magnitude* (parent > grandparent > far). No e5 → no LCA double-count.
- **`lineage-rank` → cross-entropy / ranking.** Target = an *ordering* over a node's candidate parents, from a
  **teacher** score (NOT the model's own output — self-referential bootstrapping drifts/collapses):

  ```
  rank_score(c) = μ_graph(c)  +  β · e5_sim_prefix(c)         # additive soft margin (dense e5 signal)
  ```

  Candidates are ordered by `rank_score`; `lineage-rank` is trained by softmax-CE to reproduce that order.
  Because CE is **scale-invariant**, `lineage-rank` carries order information the MSE head can't — notably in
  the flat graded-middle where the regression targets bunch. **This is the filing-primary operator** (filing =
  rank the candidate folders); `lineage` supplies graded confidence / fall-off.

**e5 is on the GRANDPARENT lineage (the prefix), not the immediate parent.** The immediate parent is the
decision the structural `decay(hops)` already makes; e5 compares the *ancestral context above it* — so
`e5_sim_prefix(c)` = e5 similarity of the `root … grandparent` prefix of candidate `c` vs the truth's. (`Math/
Calculus/…` stays close to `Math/Analysis/…` because their grandparent prefixes match.)

**β scales e5 by how much prefix there is to compare** (user):

  ```
  β = ( G / (L + G) ) · k
      G = grandparent-prefix length (root … grandparent),   L = total path length (root … node),   k = tuned blend const
  ```

  So a shallow node with no grandparent (`G=0`) → **β=0**, e5 contributes nothing (nothing to compare); deeper
  nodes → more ancestral context → larger β, up to `~0.5·k`. `k` is tuned on the pilot.

**Losses:** `lineage` → the existing graded-MSE head; `lineage-rank` → a candidate-softmax CE (same family as
the model's transitive/directional margins). Both are weighted like the other operators.

## 4. Two judges, complementary — and a cost-smart split

Graph judge = cheap, structural, but **blind to a semantically-valid parent that is structurally distant**
(a cross-domain-but-correct filing). LLM judge = semantic, catches those, but costs. Their **disagreement is the
useful part**: agree → confident μ; diverge → the ambiguous, probably-multi-parent cases.

**Cost-smart pipeline:** run the **graph judge over everything (free, `judge=graph`)**, then spend LLM
(`gpt-5.5-low`) calls **only where graph-μ is uncertain** — near-ties and structurally-distant candidates. The
model's judge axis learns both calibrations side-by-side (`corpus_emb + judge_emb` provenance token).

## 5. Integration + status

- Feeds the graded round (`build_graded_round.py` → `load_graded`) as `corpus=mindmap` rows with per-row operator
  + calibrated μ + judge tag.
- SimpleMind is **most useful once fused with Pearltrees** (`fuse_corpus.py` needs the PT cache) — this is prep;
  full value lands with the harvester.
- **Built:** `gen_mindmap_lineage.py` (491 chains); `prototype_graph_judge.py` — graph-μ validated on Chaos
  theory: true parent = 1.0, ancestors graded up the chain (2 hops ≈ 0.36, 3 ≈ 0.22 at γ=0.6), far → floor 0.02.
  **Finding:** a single-map parse is **fragmented** (many candidate `hops=99`), so the graph judge needs the
  **connected cross-map / fused graph** for real candidate connectivity — another reason SimpleMind pairs with
  Pearltrees (`fuse_corpus.py`). Nav (`via link`) nodes are now bypassed in the *structure* (not just display).
  **Next:** run on the fused graph → substitution candidate gen → targeted LLM judge on uncertain cases →
  graded-round rows.

## 6. Review responses (PR #3426)

**Positive decay (D1):** graded-positive μ uses the SAME family as the graph judge — `μ = γ^d` (geometric in
lineage distance `d`), γ tuned so a direct parent ≈ 0.9, a depth-5 ancestor ≈ 0.35. Harmonic `1/(1+d)` is the
one alternative (heavier tail). Pinned to one family, not two.

**Depth bias (methodology-2):** complete enumeration gives C(8,2)=28 pairs for a depth-8 chain vs 1 for depth-2,
so deep systems-theory nodes would dominate. Mitigation: **per-chain pair weighting** (down-weight each pair by
`1/(#pairs in its chain)` so every chain contributes equal mass), OR cap transitive pairs (keep all adjacent +
sample K transitive). The pair generator will emit **pair-count-by-depth** and apply the weight.

**Negative μ-floor (methodology-3):** since all nodes share the global root, **negatives are NOT μ=0.**
Directional-reverse (ancestor|descendant) → `μ_rev ≈ 0.05–0.15` (residual membership); cross-branch →
`decay(tree-distance)` down to `μ_floor ≈ 0.02` at shares-only-Root; only truly out-of-forest pairs get 0.

**Negative ratio by type (D2):** the 3:1 budget is split, weighted to hard — per positive ≈ 1.5 hard
(sibling/cousin, e5-near) + 1.0 cross-branch (medium) + 0.5 easy (permutation/far), logged by type.

**70/30 semantics on a lineage (methodology-1):** unlike enwiki's distinct downward-vs-bidirectional EDGE types,
on a lineage the 70/30 is a **query role**: 70% directional = μ(descendant|ancestor) [downward membership]; 30%
"superposition" = **ancestor-as-query** / symmetric μ(anc,desc) fed both orders (SYM operator). Same operators,
different query construction — will state this in §0.

**Multi-hop vs remove-last-hop (methodology-4):** **complementary, not competing.** The multi-hop
ancestor-descendant pairs (graded by distance) train the μ operators directly (LINEAGE positives). Remove-last-hop
generates the **filing-candidate** examples (true parent + substituted alternates) the judges score — it *adds*
hard negatives + judge-calibrated μ *on top of* the lineage positives, it does not replace them.

**Multi-parent, operationalized (D3):** graded-round schema adds `gt_rank` (rank of the ground-truth filing
among candidates by judge μ) + `gt_mu`. Disposition: `gt_rank==1` → normal; `gt_rank==2 & spread<τ_tie` → **keep
both** (multi-parent); `gt_rank≥3` OR `gt_mu<τ_low` → **flag + exclude from training**. `τ_tie≈0.1`, `τ_low≈0.3`
(μ scale), tuned on the pilot.

**Cost-split + judge_agreement (D4, methodology-7):** output stores `mu_graph`, `mu_llm`, and
`judge_agreement = abs(mu_graph − mu_llm)`. LLM spent only where graph-μ is uncertain: **near-tie** = top-two
graph-μ within `τ_tie`; **structurally-distant** = candidate hop-distance `≥3` OR shares-only-Root. Else free
graph-μ alone.

**Judge fusion — DON'T (methodology-6):** graph-μ is truth=1 *by construction*; LLM-μ is *discovered* and can
prefer a better parent — **not the same scale.** Stored as **separate judge-tagged rows** (`judge=graph`,
`judge=gpt-5.5-low`), never averaged; the model's `judge_emb` is the only place they reconcile. `judge_agreement`
is diagnostic, not a fused target.

**Judge confidence spread (methodology-5):** the LLM pass logs `raw_top`, `raw_runner_up`, `spread` alongside the
normalized μ, so the training loop can down-weight low-spread (uncertain) calls.

**e5 hard/easy boundary (nit):** TBD from the pilot — set at the **median e5 distance of in-map sibling pairs**
(empirical, per-map), not a fixed constant.

See also: `DESIGN_path_operator.md`, `DESIGN_calibrated_judges.md`, `feedback_enwiki_sampling_strategy`.
