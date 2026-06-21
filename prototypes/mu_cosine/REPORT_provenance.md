# Part B — provenance / source-conditioning token (infrastructure)

Generalizes the deferred **judge axis** of `DESIGN_directional_attention.md` into a **provenance token**
that records *where each label came from* — both the **corpus** (the text the graph/embeddings derive
from) and the **judge** (who produced the label). It is built as **one** maskable token with a
**factored** embedding, so it presents to the attention set as a single input while still spanning the
`corpus ⊗ judge` product.

## What was built

- **Codebooks** (`mu_attention.py`): `CORPORA = {simplewiki, enwiki}`, `JUDGES = {haiku, graph}`.
  `haiku` = a bought LLM judgment (the `SYM` positives, the `LLM` fixture); `graph` = the Wikipedia
  graph itself (the `WIKI` edges, and the free μ=0 `SYM` negatives = non-edges). `enwiki` is reserved
  for a later corpus.
- **Factored token**: `provenance = corpus_emb[corpus] + judge_emb[judge] + prov_tag`. `prov_tag` marks
  the *slot* in both states so the model can always locate the provenance input; the additive factors
  keep any corpus×judge combination expressible without adding a second token.
- **Maskable, masked-by-default**: reusing the off-manifold-noise scheme of the absent-ancestor slots,
  the token is replaced by a unit random vector (factors dropped, `prov_tag` kept) with prob
  `--prov-mask` during training and **always** on the default inference path. A masked provenance token
  = **provenance-agnostic μ** (marginalize over sources). Every existing call site stays a 3-tuple
  `(node, root, op)` and automatically gets the agnostic answer; a 5-tuple `(node, root, op, corpus,
  judge)` reveals the source.
- **Tagging**: every training label is threaded with its `(corpus, judge)` — `WIKI`→(simplewiki,
  graph), `SYM` positives→(simplewiki, haiku), `SYM` negatives→(simplewiki, graph), `LLM`→(simplewiki,
  haiku).

## Honest scope

The data is **single-corpus** (`simplewiki`) and the judge is **operator-correlated** (Haiku⇄SYM/LLM,
graph⇄WIKI), so the token carries little *new* signal yet. It is therefore validated **structurally**,
not by accuracy.

## Structural validation (the `[PROV]` probe + the ablation)

For SYM queries on clean domain nodes, μ with provenance **masked** (default) vs **revealed**:

| reveal | mean \|Δμ\| (prov ON) | mean \|Δμ\| (prov OFF control) |
|---|---|---|
| judge = haiku | 0.027 | 0.008 |
| judge = graph | **0.656** | 0.009 |

- **The slot is wired and read.** Revealing `judge=graph` collapses μ by 0.66 — the token has learned
  that graph-judged `SYM` labels are exactly the μ=0 non-edges. Revealing `judge=haiku` (the default
  positives' own source) barely moves μ (0.027) — the honest "near-constant, little new signal"
  result for single-corpus data. Masking marginalizes both → the agnostic default path.
- **Those shifts are genuinely the trained token.** In the always-masked control (`--prov-mask 1.0`,
  `corpus_emb`/`judge_emb` never receive gradients) revealing either source does nothing (Δ ≈ 0.008).

## Ablation — provenance does NOT regress Part A (same seed 1)

| metric (default = masked) | prov ON (`--prov-mask 0.5`) | prov OFF (`--prov-mask 1.0`) | Part A (no token) |
|---|---|---|---|
| WIKI held-out order-acc | 97.8% | 98.5% | 99.2% |
| SYM held-out corr | +0.722 | +0.751 | +0.756 |
| `pos_phys` corr | **+0.831** | — | +0.838 |
| `pos_math` corr | +0.853 | — | +0.818 |
| `cross_MP` corr | +0.870 | — | +0.780 |
| 5-way discrimination | 16/25 (64%) | 17/25 (68%) | seed-range 56–84% |
| gate-leak (every op) | 0/5 | 0/5 | 0/5 |

ON vs OFF differ only within the seed-noise band already established in `REPORT_matheng.md`
(discrimination 56–84% across seeds; SYM ±0.03). **Adding the provenance token leaves the Part-A
results intact** — `pos_phys` stays recovered (+0.831), Math/cross_MP hold. The provenance machinery is
in place and ready to carry real signal the moment a second corpus (`enwiki`) or a second judge arrives
— no architecture change, just new codebook entries.

Reproduce: `train_mu_attention.py --pairs mu_pairs_scored_matheng.tsv --llm --steps 900 --bs 64
--lr 5e-4 --wiki-weight 0.5 --margin-weight 1.0 --wiki-abs 0.5 --prov-mask {0.5|1.0}` — the `[PROV]`
probe and the ablation print in the per-operator validation block.
