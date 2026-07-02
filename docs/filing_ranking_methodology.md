# Filing & ranking methodology

How we rank, how we combine a learned model with an LLM, how we measure what it *costs* to do a task, and how we
decide the results are *good*. Scope: the Pearltrees bookmark filer (`scripts/infer_pearltrees_federated.py`,
`scripts/mu_filer_ranker.py`, `scripts/bookmark_filing_assistant.py`, `scripts/llm_cli.py`,
`scripts/llm_reranker.py`, `scripts/eval_rerank_agreement.py`) and the µ-attention model in
`prototypes/mu_cosine/`.

Status: methodology + tooling are in place; empirical numbers are being filled in (see §7). Not much hinges on
this being perfect — it's a design record for a light review, not a proof.

## 1. Two dual ranking problems

µ(bookmark | folder) is a **directional** membership score (µ(a|b) ≠ µ(b|a)). Lay all (bookmark × folder) scores
in a matrix and the two tasks are just two axes:

- **Filing** — fix a *bookmark*, rank the *folders*: "where does this go?" (read a **row**). The production task.
- **Curation** — fix a *folder*, rank the *bookmarks*: "which items fit here?" (read a **column**). Useful for
  populating a thin folder, finding misfiled items (low-fit items currently inside), or suggesting additions.

Same model, same scores — only the aggregation axis differs. Filing packs ~15 folder *titles* per decision (small);
curation packs many *bookmarks* per folder (large) — which drives different model choices (§6).

## 2. The pipeline — division of labour

`e5 coarse-rank ALL folders → µ rerank the top-K shortlist → (optional) LLM rerank / agent decision`

- **e5 (coarse):** one matmul over all folders (milliseconds). A *general* embedder — sane topical relevance for
  any query, including regions µ never trained on (catch-all for untrained space). Not skipped, because µ-over-all
  is both slow (~15 s over 8,800 folders) and *noisier* off-distribution.
- **µ (rerank the shortlist):** directional membership, the tuned combiner `0.1·e5 + 0.9·max(µ-elem, µ-hier, µ-sym)`
  (`eval_filing.py`, `DESIGN_model_applications.md`). µ is **pointwise** — one score per item — so it ranks 100 or
  10,000 items with no joint-reasoning burden. It carries the *width*.
- **LLM (final):** reasons over the *small* shortlist only. An LLM is weak at *jointly* ranking many items (see §4
  working-set), so we keep its set inside its comfort zone (~15) and let µ do the width.

For routine filing the LLM stage is often unnecessary — the filing **agent** (`bookmark_filing_assistant.py`)
already takes the µ-shortlist and reranks-and-decides in one call. A *separate* reranker (`llm_reranker.py`) is
opt-in, justified only (a) at service scale as a cheap pre-filter before an expensive agent, or (b) for big/rich
candidate sets that exceed one good agent decision. It also has an offline role: **reranking-as-eval** (§5).

## 3. Adaptive shortlist — three parameters

The shortlist size should track query *difficulty*, not be fixed:

```
shortlist_size = clamp( |{ candidates with score ≥ top − δ }|, K, M )
```

- **K (lower, ≈15)** — the one-cycle **working set**. Guarantees the cheap single-pass path for easy queries.
- **δ (semantic threshold)** — the **difficulty sensor**. Sharp top ⇒ band collapses to K; a cluster of near-ties
  ⇒ band grows to include the real contenders. Must be **relative** (a fraction of the top-to-bottom spread or a
  normalized-margin cut), not an absolute score cut — raw µ scores aren't calibrated across queries.
- **M (upper)** — an **iteration budget expressed as a list size**: since K ≈ one cycle, M/K ≈ how many
  chunk-and-merge cycles you'll spend on the hardest query. M's true cost depends on *which model* handles the
  >K case (a big band on Haiku = several cycles; on a long-context model = ~one), so a large band is itself the
  signal to route to the stronger/long-context model.

This makes the shortlist a **difficulty router**: easy → cheap one-shot on a short list; hard (research, many
dynamically-close folders) → bigger list handled by costlier machinery. You pay for rigour only when the query
earns it.

## 4. Measuring the cost to do the task

The unit is **tokens (and iterations) per *correctly-completed* task**, not per call.

- **Token accounting** — `llm_cli.call_llm` accumulates prompt/completion tokens; `reset_usage()`/`get_usage()`
  give per-task totals (chars/4 estimate; `claude -p --output-format json` gives real `usage`/`total_cost_usd`).
- **Working-set / the "knee".** An LLM's limit on jointly ranking N items is *not* context length (100 items is
  nowhere near the window) — it's attention dilution (softmax over N spreads ~1/N) + finite depth for global
  consistency, loosely analogous to human working memory (~3–7 chunks). Below the working set → one pass; past it
  → it must chunk-and-merge (roughly **linear** extra cycles). The **knee** = where one-shot quality falls off
  (accuracy knee) or self-refinement rounds climb (cycle knee).
- **Measuring the knee (`claude -p`, agentic).** Because `claude -p` is Claude Code, we run a **refine loop**
  ("propose an order, critique, refine, repeat until confident") and count **rounds-to-stabilize** (the model's own
  sense of "good enough" = its self-assessed working set), plus `num_turns`/tokens/cost. Sweep N ∈ {10,15,20,30,
  50,75,100}. Haiku is the **smallest** Anthropic model, so its knee is a **conservative lower bound** for the
  family, and its `total_cost_usd` is a **baseline** to compare non-Anthropic models against — subtracting the
  **fixed overhead** (system prompt / framing tokens) to compare the *marginal* cost of the list itself.
- **Economics of a separate reranker.** `total = iterations × tokens/iteration × price`. A pre-ordering can cut the
  agent's own spend even on a short list (position anchoring), but a cheap *wrong* pre-order can *cost* tokens via
  corrections — so the honest metric is tokens-per-*correct* task, and it can flip with N and with model. Only
  measurement decides.

## 5. Evaluating whether the results are good — two methods

- **(A) Ground truth — the actual filing.** From `load_filing`, each bookmark has its true folder (tree-id).
  Metrics (`eval_rerank_agreement.py`), matched by **exact tree-id**:
  - **shortlist recall@K** — is the true folder even in the top-K µ shortlist? Top-1 accuracy is *bounded* by this,
    so it's reported separately (don't conflate recall with ranking).
  - **µ-top1 vs LLM-top1 accuracy**, *conditioned on the true folder being in the shortlist* — did the rerank move
    it to #1?
  - **Coverage conditioning:** only ~53% of filing folders exist in the federated model, so we condition the sample
    on the true folder being in the model's folder set — otherwise the 47% coverage gap shows up as fake ranking
    misses.
- **(B) LLM judge.** When there is *no* ground truth (open-ended / semantic sorts, or ranking quality on arbitrary
  lists), a **stronger** model (Sonnet/Opus) grades the ranking — Haiku can't reliably grade itself. This is the
  more expensive path; used only where ground truth is unavailable.
- **Reranking-as-eval (judge-free µ-quality signal).** Run the LLM reranker over µ's shortlist and measure
  **displacement** (mean |Δrank| after the LLM reorders) and **top1_changed**. Low ⇒ µ already agrees with a strong
  judge ⇒ µ ranks well — on *any* bookmark, no ground truth needed. It also **predicts economics**: low displacement
  ⇒ the agent has little to fix ⇒ fewer agent tokens. (Validate the judge itself against ground truth before trusting
  its displacement — hence method A.)

## 6. Model selection, by stage

The wrapper (`llm_cli`) makes the backend a flag, so match the model to the stage:

- **Filing rerank/agent (~15 folder titles):** **Haiku** (`claude -p`) — strong format adherence (clean parseable
  rankings), cheap on subscription, and the small set sits in its working set. Tool-calling edge matters for the
  *agent* (it decides/acts).
- **Curation over many bookmarks (long list in one prompt):** **Gemini / `agy`** — long context is the one place it
  genuinely wins, and it spends otherwise-idle Gemini quota.
- **Judge for open-ended quality:** **Sonnet/Opus** (§5B).
- The δ-band (§3) routes automatically: a big band ⇒ hand it to the long-context model rather than burn Haiku cycles.

## 7. Empirical results

Runs use the s243a federated model (8,800 folders) + `model_prod.pt`, Haiku via `claude -p`.

- **N=10, built-in probe set (no ground truth):** mean displacement **0.39**, top1_changed **50%**, ~198 tokens/
  rerank. Qualitatively the LLM *refines* µ's topic-right #1 to a more specific folder (relativity, bash control
  structures), not random reshuffling.
- **Coverage / recall:** id spaces match; **~53%** of filing folders exist in the federated model (a coverage gap,
  addressed by conditioning the sample on the true folder being in-model).
- **N=100 (real bookmarks + true folders, coverage-conditioned, Haiku via `claude -p`, seed 7):** 100/100 parsed,
  0 timeouts.
  - **shortlist recall@15 = 28%** — µ surfaces the true folder in the top-15 (of 8,800) ~¼ of the time. **This is
    the bottleneck**: for 72% of bookmarks the true folder isn't even a candidate, so no rerank can help.
  - of the 28 in-shortlist cases: **µ-top1 35.7% → LLM-top1 64.3%** — the LLM rerank nearly **doubles** top-1.
  - **mean displacement 0.491, top1_changed 69%** — µ's order is far from the LLM's, so the rerank is doing real,
    productive work (not a "µ already good, skip it" case).
  - **tokens/rerank ≈ 184** (18.5k total for 100) — cheap.
  - **Read:** the LLM rerank earns its keep *where the true folder is a candidate* (36%→64%); the lever for the
    other 72% is **recall, not reranking** → widen the shortlist (bigger K / the adaptive δ-band, §3).
  - _Caveats:_ "true folder" = where the user filed it (other folders are often equally valid), so strict tree-id
    match **understates** quality — 28%/64% are conservative. Single model, single 100-sample.
- **Recall@K curve (N=100, μ+e5 over all 8,800 folders, judge-free):** true folder within top-K —
  | cutoff | @15 | @30 | @50 | @100 | @200 | @500 | | e5 | 41 | 44 | 48 | 53 | 55 | **64** | | μ | 22 | 24 | 26 | 34 | 37 | 43 | | max(μ,e5) | 40 | 44 | 47 | 50 | 52 | 63 |  (%).
  Findings: (1) **e5 climbs** with K (41→64%) — widening helps recall. (2) **µ recall < e5** at every K — µ is a
  *rerank* signal, not a recall signal. (3) **max(µ,e5) ≈ e5** — the union adds nothing, so the µ-tower / wider-µ
  cutoff is NOT worth building for recall. (4) **µ set-selection loses recall**: the filing eval's µ shortlist
  recall@15 was 28% vs e5's own 41% — µ reranking e5-top-64 → top-15 *demotes* true folders.
  **Revised design:** e5 *selects* the shortlist set (bigger K); µ only *reorders within* it (never prunes below
  e5's recall — set µ's `shortlist_k` = the LLM's target K); LLM does final precision. `max(µ,e5)` dropped;
  µ-tower shelved for recall. Roles: **e5 = recall (wide net), µ = precision (reorder), LLM = final precision.**
- **Knee sweep (Haiku one-shot vs refine-loop, N=10…100):** _pending — rounds-to-stabilize, num_turns, cost vs N;
  the accuracy knee via ground-truth-orderable lists._
- **e5-direct vs µ-select (pending):** e5-top-15 → LLM vs µ-select — since µ costs ~13 pts of recall, e5-direct may
  win end-to-end; tests whether µ should touch the *set* at all or only the *order*.

## 8. Open items / caveats

- Confidence-adaptive α (defer to e5 when µ is uncertain) needs margin **calibration** to work as a live
  single-query signal — deferred; OOD is currently handled by the LLM/agent as final arbiter.
- `codex` backend flag syntax is unverified (needs node ≥ 22).
- Per-call `claude -p` startup (~15–20 s) dominates large-N runs; for a service, use the API, not the CLI.
- Displacement believes the judge; validate the judge against ground truth (method A) before trusting it.
- **µ recall < e5 is likely undertraining, not architectural.** µ is a small head on *frozen* e5, trained on a
  modest filing set and applied here to the federated model's 8,800 folders (partly out-of-distribution). **Primary
  lever: retrain µ on more data** (the RDF-assembly + harvest in this branch) → re-run the recall@K curve. Nuances:
  µ rides on e5 so it's *bounded* by e5's information (can approach, not exceed, e5's recall), and its directional
  objective leans precision, so µ may stay precision>recall by design — but the gap should close with data.
- **Bi-encoder / µ-tower (FUTURE TO-DO, currently NOT helpful).** Distill the µ cross-encoder into a precomputable
  per-folder × per-operator embedding via a learned MLP head on µ's hidden layer (operator/provenance as inputs —
  the `MLP(spec)` head from `DESIGN_path_operator.md`), so µ scoring becomes a dot-product (fast as e5) and
  `max(µ,e5)` over all folders is trivial. Trained separately by distillation (freeze the cross-encoder; ranking loss
  on generated (bookmark,folder,µ) triples). **Not worth building now** — µ recall < e5, so a fast µ-net adds nothing.
  **Its value is coupled to µ's recall becoming competitive** (i.e. build it *after* the retrain lifts µ recall). Keep
  the cross-encoder for the final small-set rerank regardless (two-stage µ: fast approximate tower for recall,
  accurate cross-encoder for rerank).
