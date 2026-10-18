# Embedding section-categorisation layer — built, calibrated, and an honest negative on current data

The semantic escalation after exact + fuzzy (`pt_sections.py` ladder: `exact_phrase` → `fuzzy` → `embedding`).
Lexical fuzzy catches **typos** (`Subtoipcs`); embedding is meant to catch **synonyms / paraphrases** that
share no edit-distance with a keyword (`Members`, `Narrower areas`, `Parent concepts`). Each category has a
few canonical **exemplars**; a section label is e5-encoded (`query:`) and cosine-matched against the
exemplars (`passage:`), best-per-category. Encoder: the same frozen `intfloat/e5-small-v2` as the model,
loaded **offline** from the HF cache (`section_embed.py`).

## Why it needs a conservative gate (live calibration on the real harvest)
Running e5 over the **28 residual labels** (the ones exact + fuzzy both miss, out of 73 distinct) showed the
naive "nearest exemplar, absolute threshold" categoriser would **inject more noise than signal**:

1. **The residual is overwhelmingly TOPICAL, not relational.** `Electromagnetism`, `Control Theory`,
   `Transfer Functions`, `Vector Calculs`, `dimensional analysis`, `The Equations` are *content* section
   names — not relation-type headers. They should stay inferred / structural-default, **not** be forced into
   a relation.
2. **Everything sits in a narrow e5 band (cos 0.80–0.89)** where `see also` / `related` is a generic
   attractor; 2nd-place margins are tiny (0.01–0.02).
3. **Genuine signal doesn't separate from junk.** The strongest real-ish hit `Links → reference (0.886)` is
   barely above the junk `My Groups → element_of (0.873)` and `Friends Pages → see_also (0.865)` — which the
   tests explicitly want rejected. A low absolute threshold sweeps in the topical pack, including the
   *harmful directional* commitments (`The Equations` / `My Groups` → element_of).

So `embed_mode` uses a **conservative threshold (0.88) + a 1st-vs-2nd margin gate (0.02)**, and an embedding
hit carries **confidence = the cosine** — a *graded, <1.0* tier (a soft relation prior that feeds the
operator-noise model, **not** a hard label), so even an accepted hit lands in the inferred tier rather than
polluting the tagged set.

## Result at the safe gate (threshold 0.88, margin 0.02)

| | count | examples |
|---|---|---|
| **rescued** | **1 / 28** | `Links → reference` (cos 0.886) |
| left inferred (correct) | 27 / 28 | the topical/content names above |
| known junk rejected | all | `Meta`, `My Groups`, `Friends Pages`, `Papers`, `Transient response…`, `Nonlinear control` |

**One real, non-directional rescue; zero false positives; zero junk admitted.**

## Honest verdict
On the **current** harvest the embedding layer is **near-neutral** — the lexical layers (exact + fuzzy +
tag/qualifier + parent-signal guard) already capture essentially all the *relation-header* signal, and what
remains is genuinely topical content or Pearltrees nav junk that *correctly* stays inferred. This directly
explains the "surprised how much fuzzy tagged" observation: there were almost no paraphrase-style headers
left for embedding to catch.

What the layer buys is therefore **not a number now** but:
- **Safety + readiness** — calibrated to reject the topical pack, so it can be enabled (`fuse_corpus.py
  --section-method embedding`) without injecting noise, and it will earn its keep on **paraphrase-rich
  corpora** (other users' Pearltrees, where headers like `Members` / `Narrower areas` actually occur).
- **A reusable e5 encoder** (`section_embed.py`) for any other semantic-match need (e.g. feeding section→
  exemplar cosine as a *feature* to the posterior rather than a hard label).

## Future directions (deferred)
- **Topical-section ⇒ membership.** A section literally named `Electromagnetism` under a Physics tree means
  its pearls are *members of the Electromagnetism subtopic* — i.e. it implies `element_of` of a *new
  subtopic node*, not a relation-type. Modelling that (synthesising the subtopic node) is a different
  feature than label→relation-type categorisation; noted here as the higher-value way to use these topical
  names.
- **Section→exemplar cosine as a posterior feature** (graded confidence input) instead of a thresholded
  label.
- **LLM layer** for the genuinely-ambiguous residual + the anomaly/quarantine queue (budget-gated).
