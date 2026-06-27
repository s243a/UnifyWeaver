# 4-layer / 0.7-confidence base model — first training run (and an overtraining flag)

The first real training of the chosen **4L / `--blend-tagged-conf 0.7`** config (per
`REPORT_capacity_conf_tradeoff.md` + the design dialogue) — the base we intend to **build up by
fine-tuning** as new Pearltrees harvests + mindmap bridges arrive.

## Run
```
train_mu_attention.py --graded context/graded_cx_pairs.tsv \
  --init-from model_nodetype.pt --use-nodetype --layers 4 \
  --infer-blend --blend-tagged-conf 0.7 --steps 1500 --bs 128 --seed 1 \
  --save model_4l_blend07.pt
```
- Warm-started from the 3-layer `model_nodetype.pt` (3 shared layers + heads loaded; **4th layer
  random-init**, fine-tuned).
- Graded round: 2865 train / 505 held-out; **115 inferred** rows (the fuzzy-unlabelled residual) trained via
  the posterior blend, tagged rows regularised at `c=0.7`.
- Artifact: `model_4l_blend07.pt` (gitignored; regenerable).

## Result

| metric | this run (4L/0.7, 1500 steps) | 4L/0.7 A/B reference (700 steps) |
|---|---|---|
| multi-domain discrimination | **100% (25/25)** | 97.3% (mean) |
| WIKI held-out order-acc | 99.9% | 99.8% |
| **SYM held-out corr** | **+0.554** | **+0.689** (range 0.63–0.77) |

## The flag: 1500 steps OVERTRAINED relative to 700
Discrimination saturated at 100% while **SYM held-out dropped to +0.554 — below every 700-step A/B seed.**
That is the in-distribution-up / held-out-down signature of mild overfitting: the extra 800 steps bought a
perfect discrimination probe at the cost of SYM generalisation. (SYM is a noisy 40-positive held-out set, so
some of the gap is noise — but 0.554 is *outside* the A/B range, so it is partly real.)

**Recommendation for the production base:** train **fewer steps (~700–900)** — or add early-stopping on the
SYM held-out corr — so the base starts from a better-generalising point. The current 1500-step model is
usable (disc/WIKI are strong) and SYM should partly recover under fine-tuning, but a shorter base is the
cleaner starting point for the build-up-by-fine-tuning plan.

## Next
- (Recommended) re-save the base at ~700–900 steps for better SYM generalisation.
- Begin the **fine-tuning arc**: fold each new harvested Pearltrees tree + bridged mindmap into the graded
  round and continue-train from this base — the 4th layer trains up gradually across rounds (dissolving the
  random-init handicap), and `c` can be tightened from 0.7 toward 0.85 as data grows.
- The ~30 fuzzy-unlabelled section labels are mostly **topical content sections** (correctly inferred, not a
  parser failure) + a little Pearltrees nav junk + ~4 genuinely ambiguous (`Courses`/`Links`/`Papers`/
  `course material`) — the only candidates worth an LLM pass, and only if a cheap source-side triage doesn't
  resolve them first.
