# Ensemble bridge judge — architecture + a calibration finding

`bridge_ensemble.py` scores each cross-corpus bridge with several **models** and combines them with a
pluggable **judge** into a keep/quarantine verdict. Built to the spec: models are named scorers; the judge
receives `{model_name: score}` (so it *knows* the models but need not use the names); the default judge is a
factory-built closure `geomean_judge(threshold=0.7)` (geometric mean ≥ threshold, caller-overridable — 0.7
is by-feel, not a law). A second factory, `confirm_judge(model_name, t)`, demonstrates a **name-aware**
judge.

## Models
- **e5** — cosine of frozen e5 title vectors (independent of all labels).
- **model:CKPT** — symmetric μ(a|b) from a trained MuAttention, provenance MASKED (agnostic). Use a
  checkpoint that **predates** the bridges (`model_nodetype.pt`) ⇒ unbiased *and* richer than e5.
- **llm** — optional, Haiku, budget-gated (same interface; not wired by default).

## Calibration finding (the useful part)

Running `e5` + the pre-bridge `model_nodetype.pt`, geomean ≥ 0.7, on 827 bridges → **quarantined 549 (66%)**.
That is **too aggressive**, and *why* is the finding:

| bridge | e5 | model | truth |
|---|---|---|---|
| `mm:cybernetics ↔ pt:cybernetics` | 1.00 | **0.91** | same concept ✓ |
| `pt:cybernetics ↔ wiki:centrifugal-governor` | 0.83 | **0.047** | **legit** (textbook cybernetics) |
| `pt:control-system-engineering ↔ wiki:norbit` | 0.78 | **0.047** | suspect |
| `pt:cybernetics ↔ wiki:ladybird-of-szeged` | 0.80 | 0.018 | suspect |

The pre-bridge model **confirms concepts it knows** (same-slug cross-corpus: 0.91) but **abstains** —
≈0.04, *undifferentiated* — on the cross-corpus **page** pairs it never trained on. The legit textbook
bridge (`centrifugal-governor`) and the bad one (`norbit`) get the **same** 0.047: the near-zero is "I don't
know," not "wrong." So a geomean that lets abstention **veto** over-quarantines, sweeping in legit bridges.

And **e5 alone can't reject** these either — its cosines are compressed (0.78–0.86 across the whole set), so
it separates text-dissimilar links weakly and never abstains.

### Conclusions
1. **Pluggability vindicated.** The geomean default isn't right for an *abstaining* model — exactly the case
   the configurable judge was designed for. `confirm_judge("model:model_nodetype.pt", 0.5)` trusts only the
   model's positive votes and defers the rest, instead of letting abstention veto.
2. **The cheap judges rank, they don't reject.** For cross-corpus *page* bridges, neither e5 (saturated) nor
   the pre-bridge model (abstains) can reliably *reject* a legit-looking link. They are good at **ranking
   suspicion** (the model's low-confidence set surfaces what to look at), not at the final call.
3. **The real rejecter is the LLM** (or a model trained *with* the bridges — informed but circular). The
   ensemble's job is to **bound the LLM bill**: rank by geomean, send the top-N most-suspicious to the LLM.

## Resolution: the over-quarantine was a DATA BUG, not just calibration

The 66% quarantine had a root cause: **most "bridges" weren't bridges.** `fuse_corpus.py` /
`parse_pearltrees.py` labelled *every* pt→wiki link `bridge`, but a bridge is the **same concept** across
corpora (identity, μ≈0.9). A wiki page in a collection that names a *different* thing (`Cybernetics`
collection → `Centrifugal governor` page) is the collection's cross-dataset **reference**, not identity.
The model's abstention (~0.047) was the honest signal that these *aren't* the same concept — it was right.

Fixed with a **same-concept gate** (`bridge` iff the endpoints' normalised titles match, else `see_also`).
Result: 827 → **61** true bridges; the ensemble (e5 + pre-bridge model, geomean≥0.7) now **keeps 60/61**, and
the single "quarantine" (`mm:courses-dynamical-systems ↔ pt:courses-dynamical-systems`, same slug) is itself
legit (e5 0.911; the model is merely under-confident at 0.327). So once the bridges are *actually* identity
links, the ensemble agrees with them — exactly as it should.

The architecture finding still stands for the genuinely-unknown cross-corpus pairs (the model abstains, can't
reject) — but those are now correctly typed `see_also`, so they no longer masquerade as bridges needing a
verdict.

## Next (item 2: review the current bridges)
Use the ensemble **ranking** to feed a **bounded LLM review** — the top ~20–30 lowest-geomean bridges
(model-abstained + any genuinely text-dissimilar), not all 549. That keeps the Haiku spend small while
catching the real bad links; LLM verdicts then re-include legit non-obvious bridges (BELBIC,
`centrifugal-governor`, `towards-a-new-socialism`) and drop the noise. Budget-gated — confirm before the run.
