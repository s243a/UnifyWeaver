# Channel campaign: the S channel exists — data was the bottleneck, as diagnosed

*2026-07-09. The B1 disposition ("blocked on data") tested: 2,000 stratified pairs scored (gpt-5.5-low,
1,770 recovered after an ingest guard; ~2h, within approved spend), then the B1b anchored recipe re-run on the
campaign data. Branch: `claude/channel-campaign`.*

## 1. The stratified sample confirmed its design hypothesis

Strata per corpus (100k_cats + Behavior slice): transitive h1–5, siblings, cousins, random.

| stratum | D mean±sd | S mean±sd |
|---|---|---|
| transitive h1→h5 | 0.85→0.29 (the familiar decay) | ~0.25–0.37, sd 0.14 |
| **siblings** | 0.23 | **0.480 ± 0.153** |
| cousins | 0.12 | 0.276 ± 0.175 |
| random | 0.007 | 0.018 ± 0.041 (clean negatives) |

**sd(S): lateral 0.193 vs transitive 0.137 — the variance the S channel needed, exactly where B1's failure
analysis said it would be.** (All prior S labels came from ancestor pairs where assoc barely varies.)

## 2. B1-retry: the missing channel establishes on BOTH corpora

Same anchored recipe (judge_emb 5→9 + last layer + readout, agnostic-anchor, lr 5e-4 + clip), ~1,900 campaign
training rows/corpus, descendant-disjoint held sets:

| channel | old data (B1b) | campaign data |
|---|---|---|
| **llm-S** | ~0, unstable | **+0.74 exploratory / +0.61 fresh, stable & climbing** |
| llm-D | ~0.54 | +0.75 / +0.64 |
| graph-d | +0.75 expl / 0.18 fresh (no transfer) | +0.72 / **+0.47–0.57 — transfers now** |

Anchor drift 0.08–0.12 (small). The graph channel's new transfer comes from the lateral/random strata giving
it real negatives (hit_prob=0 rows) instead of ancestor-only structure.

**Caveat to quantify next:** part of the S correlation is BETWEEN-strata (step 1 already showed +0.6 — e5
similarity separates siblings from random at the stratum level; the old ancestor-only eval had no S variance
to detect anything with). The within-stratum correlation is the sharper number; training clearly adds on top
(+0.62 → +0.74), and stratum-level discrimination is itself the deployable skill.

## 3. Judge-format guard

Both the campaign and a model-probe run hit the same crash: judges occasionally emit stray non-dict tokens
(bare ints) in the response stream, and one crashed the whole ingest (responses safe on disk; re-ingested).
`score_inferred_tail.ingest` now skips non-dict objects with a count. At campaign scale, format discipline is
a judge-quality dimension alongside correlation and bias.

## 4. Judge economics (user)

gpt-5.6-tera and plain gpt-5.6 are NOT available via ChatGPT-account codex (400). **gpt-5.6-luna IS** — it
required a codex CLI upgrade (0.142.0 → 0.144.1). A luna validation run (fresh 250, three-way vs the two
gpt-5.5 runs, decision rule pre-registered: corr vs the 0.954 self-consistency ceiling + bias vs half a
quantization step) determines whether cheap-luna can carry future campaign scoring.

## Repro

```
python3 sample_channel_campaign.py --per-corpus 1000
python3 score_with_codex.py --pairs /tmp/mu_data/campaign_pairs.tsv --batch 10 --out ... --responses ...
python3 check_campaign_strata.py
python3 fine_tune_channel_heads.py --data campaign --steps 800 --lr 5e-4 --unfreeze-last
```

## 5. Judge identity as an e5-name function — measured (user, 2026-07-09)

Current scheme: distinct learned `judge_emb` rows (a new judge = zero-init, no prior — luna's onboarding
problem). The designed alternative (`DESIGN_amortized_fusion_heads`: `cond(f) = W·e5(name) + residual`) predicts
family/version structure in the names. MEASURED (e5 cosine of judge-name embeddings):

| | within GPT family | claude family | GPT vs other LLMs | GPT vs non-LLM |
|---|---|---|---|---|
| cosine | **0.969** | 0.897 | 0.792 | 0.809 |

e5 reads vendor+version exactly as predicted (user): `5.5-low ↔ 5.6-luna` = 0.968. Under name-conditioning a
new judge onboards with a family-graded prior (luna ≈ full transfer from gpt-5.5-low; a new Claude judge
mostly-from-haiku/opus; a new vendor from the LLM centroid) instead of a zero vector — borrowing strength falls
out of the name geometry. Caveat: raw e5 cosines are baseline-inflated (~0.8); the learned translation W's job
is to amplify the relative block structure. Strengthens the case for migrating judge conditioning to the
name-function when B2 lands.

**Luna disposition criterion (user, superseding the plain three-way rule):** the indexed judge_emb scheme only
offers binary choices — reuse 5.5's row (contaminates its calibration if luna differs) or a new zero-init row
(discards the 0.97 family prior; the probe showed zero-init rows barely learn). A similar-but-not-identical
judge falls in the gap. So: **interchangeable** ⇒ map luna to the gpt-5.5-low row, take the cheap data, defer
the architecture; **middle case** ⇒ do NOT add luna under the indexed scheme — either migrate to
name-conditioning first (B2, now with a forcing use-case) or use the pragmatic bridge: a new index WARM-STARTED
as a copy of 5.5's row (hand-made name-prior; inherit family calibration, learn only the deviation);
**degenerate** ⇒ stay on 5.5. General principle: the indexed scheme is adequate while the judge set is static;
the first similar-but-not-identical judge is the forcing function for the general judge/source architecture.
