# Training on the multi-corpus graded round — node-type finally pays off

Wiring the fused graded round (`build_graded_round.py`, `REPORT_graded_round.md`) into the trainer and
running it. Headline: with the cross-corpus type diversity the bridges supply, the **node-type token now
helps** — the result `REPORT_nodetype.md` was waiting for.

## Trainer integration
- `--graded <out>_pairs.tsv` adds a mixed-operator graded path: per-row operator (WIKI/ELEM/SYM), weighted
  MSE to the calibrated μ, **bridges down-weighted** (`--bridge-weight 0.2`) so the μ≈0.9 "same concept"
  bulk doesn't swamp the directional signal. Endpoints carry corpus/judge (maskable) + node-type tags.
- The fused nodes are unioned into the e5 build, each embedding its **title** (`embed_text`), not its
  `mm:/pt:/wiki:` key — `build_e5_tables` gained a `texts=` override for this.
- **GPU support** (`--device auto|cuda|cpu`): e5 embedding + training run on CUDA; batches tokenized on CPU
  and moved per-step. On a 4 GB GTX 1660 SUPER, `--bs 128` fits. (The e5 cache now stores the embedded
  `human` strings, so pre-existing caches rebuild once.)

## Recipe matters: warm-start, not from-scratch
Training the graded round **from scratch** on the harsh cumulative config collapsed SYM into the sigmoid
dead-zone (μ→0 everywhere, SYM corr +0.001; WIKI L stuck at 7.42). The round is meant to **fine-tune** an
already-trained checkpoint:

    python3 train_mu_attention.py --init-from model_nodetype.pt --graded /tmp/graded_pairs.tsv \
        --use-nodetype --pairs mu_pairs_scored_cumulative.tsv --steps 700 --bs 128 --lr 3e-4 --device cuda

Warm-starting from `model_nodetype.pt` (L3) preserves the healthy SYM/WIKI/ELEM and *adds* the multi-corpus
round. Warm-start also grows `corpus_emb`/`judge_emb` for the new `pearltrees`/`mindmap`/`human` codebook
entries.

## Results — node-type A/B (identical fine-tune, ±`--use-nodetype`)

| metric | **+nodetype** | −nodetype | control / prior |
|---|---|---|---|
| multi-domain discrimination (argmax) | **94%** (34/36), top-2 100% | 86% (31/36) | — |
| graded held-out **SYM** fit (r) | **+0.988** (MSE 0.002) | +0.577 (MSE 0.047) | — |
| graded held-out **WIKI** fit (r) | +0.757 | +0.706 | — |
| **SYM** held-out μ corr | **+0.838** (MSE 0.034) | +0.822 | control +0.726 |
| **WIKI** held-out edge order-acc | 99.8% | 99.8% | — |
| **ELEM** centrality corr / direction | **+0.702 / 100%** | +0.603 / 98% | — |
| gate-leak (SYM / WIKI, ≥0.3) | 0/5 · 0/5 | — | control 1.1% |

**Node-type helps across the board**, biggest on graded fit (+0.988 vs +0.577) and discrimination (94% vs
86%); SYM even edges past the +0.726 control. `REPORT_nodetype.md` found node-type *collinear with the
operator* on single-corpus data and banked it behind `--use-nodetype` (default off); now that bridges give
the same relation distinct endpoint types (mindmap_node ⟷ pearltrees_collection ⟷ category/page), the token
carries real signal. The gate-and-bank strategy paid off.

## Notes
- The graded `_pairs.tsv`/`_nodes.tsv`, the `model_graded.pt` checkpoint, and the e5 caches are
  local/regenerable (derive from gitignored fused/`.pt_cache` data) and **not** committed.
- Bridge dominance handled by `--bridge-weight 0.2`; the thin ELEM in this round (6 targets) rides the
  existing wiki-page ELEM training — ELEM stayed healthy (+0.702).

## Re-measured on the corrected round (bridge = same-concept; references → element_of)

The numbers above were on the *pre-correction* (bridge-dominated) round. After the same-concept fix —
`bridge` 1654→122 (identity only) and cross-dataset references → their specific relation (`element_of`
1538, see_also fallback) — the same warm-start fine-tune (`--use-nodetype`, GPU) gives:

| metric | corrected round |
|---|---|
| graded held-out **ELEM** fit (r) | **+0.999** (MSE 0.002) |
| graded held-out **SYM** fit (r) | +0.865 |
| graded held-out **WIKI** fit (r) | +0.837 |
| **SYM** held-out corr | **+0.836** (> control +0.726) |
| **WIKI** order-acc | 99.8% |
| **ELEM** centrality corr / direction | +0.662 / 100% |
| discrimination | 94% (34/36) |

The fix isn't just semantically honest — it *helps*: the references' correct `element_of` relation hands the
previously-starved ELEM operator (6 → 1538 targets) rich cross-corpus, node-type-diverse signal, and it fits
at **r +0.999** while every other operator stays healthy (SYM still beats the control).

## Next
- Bring the **account** token + the `Team <name> <id>` e5-text online once the `s243a_groups` account is
  harvested (the plumbing is in; `--use-account` will gate it).
- Sweep more mindmaps into the fused corpus for a larger graded round; re-measure node-type's margin as the
  within-operator type diversity grows.
