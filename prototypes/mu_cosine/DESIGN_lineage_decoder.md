# Lineage Decoder — design sketch (FUTURE WORK, not an immediate priority)

Sketched during a training run, per the "would be cool to design for future work" note. This is the generative
counterpart to the **retrieval** LINEAGE operator (`train_lineage.py`): instead of *picking* the best existing
folder/path, a decoder *generates* the placement path — and, crucially, can **propose a folder that doesn't exist
yet**. Everything below is design, not built.

## 0. Core framing: the decoder is a *constrained optimizer around the encoder* (no separate decoder network)

The cleanest conception (supersedes the architecture-first framing below): the encoder defines a score/energy
landscape `μ_lineage(bookmark | path)`, and **decoding = optimize the path to maximize that score** — an
energy-based / inference-as-optimization view. There is *no decoder network to train*; the "decoder" is a test-time
optimization loop over the encoder we already have.

**The one thing that makes it work: it must be a PROJECTED optimizer.** Free optimization of the path embeddings
finds off-manifold degenerate maxima (high score, not a real path). The **snap-to-existing-children step is the
projection** — it keeps each iterate on the manifold of real, tree-connected folders (or marks NEW). So precisely:
*decoding = projected iterative optimization of the encoder's score; projection = snap to actual folders + respect
tree connectivity.*

This unifies everything already built:
- **Retrieval = the 1-step greedy version** (single argmax). The decoder is just *more iterations of the same
  objective* — retrieval and decoding are one system at different iteration counts.
- **Margin gate = the stop/novelty rule** — converged with sharp margin → snap; flat → propose NEW; confidence
  plateau → stop deepening (depth placement).
- **Anytime/partial for free** — stop early → a coarse general-branch solution, which is actionable.
- **Diffusion/recursive (§4b) = the update rule** — a refinement step moves toward the encoder's high-score region;
  warm-init from the retrieval top-1. So §4b's "architectures" are really *choices of optimizer/update rule*, not
  separate models.

Net: the decoder collapses to **init from retrieval → iterate the encoder score under the snap-projection → stop by
margin.** The sections below (why, novelty, training, naming, eval) all still apply — read them as details of this
optimizer.

## 0b. Training the optimizer = meta-learning (learning-to-optimize) — staged, with a safety net

If the decoder is an optimizer around the encoder (§0), then *training* it is **learning-to-optimize / meta-learning
/ amortized inference**: encoder = learned *scoring* (good path), meta-learned optimizer = learned *search* (find it
fast). Same "model carries its own meta-signal" through-line as the confidence-adaptive blend (there: *when* to trust
itself; here: *how* to search itself).

**Optimizer state / IO (the L2O template):** the learned optimizer's inputs are (1) the **bookmark embedding** — the
task/target descriptor, fixed across steps — and (2) the **last encoder output on the current partial path** — the
current-solution feedback (analogue of "current params + gradient"). It outputs the next path-embedding estimate,
which snap-projects to identities; re-encode; repeat. Two design points:
- **Markovian-sufficient by re-encoding.** Because the encoder re-reads the *whole* partial path each step, its
  output summarizes the entire partial solution — so the optimizer can **revise an earlier level (backtrack)** with
  no separate memory. That's precisely what greedy can't do. (A recurrent trajectory memory is an L2 add-on for
  sharper backtracking; L1 stays Markovian on these two inputs.)
- **Feedback must carry DIRECTION, not just magnitude.** The scalar μ says "how good," not "which way" — you can't
  step from one number. Feed the encoder's **hidden representation** of the partial path (attention output pre-readout)
  and/or the **per-frontier-child μ distribution** (says which child to move toward). The optimizer then learns a
  vector field: "given where I am (hidden state) and where I want to be (bookmark), here's the next move."

Do it **staged**, cheapest first:
- **L0 — hand-designed optimizer** (greedy projected walk, retrieval-init). No meta-training; likely most of the value.
- **L1 — small learned update rule** (the recursive net). Add ONLY if L0 leaves a measurable gap (greedy stuck / no
  backtracking).
- **L2 — full learned optimizer** (recurrent state, learned step size/schedule). Only if L1 shows headroom.

Why the meta-learning is realistic here (not a research gamble):
- **Learn only the RESIDUAL.** μ_lineage was trained on prefixes (coarse-to-fine), so a greedy walk already converges
  decently — the optimizer only needs to learn the *revision/lookahead* greedy lacks (fix an early level after seeing
  the leaf). Small correction ≫ easier than learning search from scratch.
- **Freeze the encoder** during optimizer meta-training (fixed energy; learned search over it — no moving target).
- **Dense loss** — supervise every step to reduce path-error / raise μ, plus a convergence-speed term. Sparse
  final-only reward is where learned optimizers go unstable / fail past the training horizon.
- **Safety net (why it's safe to try here):** the snap projection means even a bad optimizer step lands on a *real
  folder* — worst case the decoder degrades to retrieval, never to garbage. Anytime + monotonic-ish by construction.

## 1. Why a decoder (what it adds beyond retrieval)

Retrieval (current) scores `μ_lineage(bookmark | candidate-path)` over existing folders + their prefixes, and takes
the top hit. It can place into any *existing* node (leaf or intermediate). It **cannot**:
- propose a **new leaf** under a good branch ("create `Wave momentum` under `Physics ▸ Waves`"),
- propose a **new intermediate branch** when the topic isn't covered,
- express placement as a *path the tree doesn't yet contain*.

A decoder emits the path level-by-level and, at each level, may **create** rather than select. That's the
"this belongs somewhere new" capability.

## 2. Core idea: decoding = autoregressive prefix-extension in lineage space

The retrieval work already gives us the decoder's backbone. We trained on **prefixes** (path-prefix dropout), so the
model already scores `μ_lineage(bookmark | prefix)` at every depth — that *is* a per-level SELECT distribution.
Decoding is just walking it:

```
path = [root]
loop:
    children = existing children of path[-1]                      # candidates at this level
    s = μ_lineage(bookmark | path + [c])  for c in children      # per-child fit (reuse the operator)
    if max(s) margin is CONFIDENT:  path.append(argmax c)         # SELECT an existing child
    else:                           path.append(NEW(bookmark, path))   # PROPOSE a new node
    stop when depth-confidence drops (the depth-placement signal) or a leaf is chosen
```

So the decoder is autoregressive prefix-extension, and **selection at each step is the LINEAGE operator we already
have.** The only genuinely new pieces are the *stop/novelty* decision and *naming*.

## 3. Novelty detection = the margin gate we already built

The decision "select existing vs propose new" is exactly the **confidence/margin gate** from #3391 and the filing
composition: if the best existing child's `μ_lineage` **margin is sharp**, an existing folder fits → SELECT; if the
margin is **flat/low** (no existing child is a good home), that's the signal to **propose a new node**. So novelty
detection is not new machinery — it's the margin signal, reused at each decode step. (This also reuses the "gate
self-adjusts to how much μ can be trusted" property: confident → commit to existing structure; uncertain → create.)

Likewise **stop depth** = where per-level margin stops being confident (the depth-placement signal): decode down the
tree until the model is no longer sure a deeper child is warranted, then stop (or propose a leaf).

## 4. Three architecture options (increasing ambition)

- **A. Embedding-space snap-or-new (smallest, fits our stack).** Decode the path as a sequence of *target embeddings*
  in lineage space. At each level, snap the target to the nearest existing child if within a margin; else mark a
  **NEW node** (a point in embedding space) to be named later. No text generation inside the model. Most consistent
  with the frozen-e5 + μ-attention design; the "new folder" is an embedding + a naming step (§6).
- **B. Retrieval-augmented select-or-new head.** At each level, retrieve existing children via `μ_lineage`, plus a
  learned `[PROPOSE-NEW]` option; a small classifier head picks among {children, propose-new}. If propose-new, hand
  off to a naming step. Modest new head on top of the existing encoder.
- **C. Full text-autoregressive path decoder.** Generate folder-title tokens level by level, conditioned on the
  bookmark + path-so-far (seq2seq). Most flexible (free-form new folders) but the heaviest — needs a real decoder
  stack and the most data, and drifts from the embedding-centric approach. Likely overkill vs (A/B).

**Lean: A first** (or B) — it reuses μ_lineage as the selector, adds only novelty (margin) + a naming step, and
stays in embedding space. C only if free-form generation is later shown to be needed.

## 4b. Decoding paradigm: iterative refinement (diffusion / recursive) — PREFERRED over autoregressive

The §2/§4 framing was left-to-right autoregressive. A better fit is **non-autoregressive iterative refinement**
(diffusion or recursive): represent the whole path as a fixed-width sequence of node slots, start from a rough/partial
estimate, and **refine all slots jointly over several steps** — "start with a partial solution and optimize." Why it
suits filing:
- **Coarse-to-fine = general→specific.** Early steps set the general (root-ward) slots at high confidence; later
  steps sharpen the specific (leaf-ward) slots — the exact structure the retrieval results showed (general levels
  reliable, leaf hard).
- **Anytime / partial-is-valid.** Any intermediate state is a usable placement — a correct general branch is
  actionable (searchable by name, resolvable by id). Stop refining when per-slot confidence plateaus.
- **Revisability (the big one over AR).** If sharpening the leaf reveals the branch was wrong, refinement can *fix an
  earlier slot*; autoregressive commits left-to-right and cannot backtrack.
- **Compute allocation.** More refinement steps where uncertain, fewer where confident.

**Output sizing:** fixed width = **2× the longest existing path (13 → 26 slots)**. The 2× headroom is for
*propose-new* (a generated path with new intermediate nodes can be deeper than any existing path) and for
overshoot-then-prune during refinement. Unused slots decode to EOS/empty.

**Embedding-space diffusion (fits the frozen-e5 stack):** the sequence is 26 path-node *embeddings*. Condition on the
bookmark embedding; iteratively denoise the sequence toward the true-path embeddings (from noise, or from a
bookmark-broadcast init). **Readout per slot = snap-or-new** (§3): snap to the nearest existing child within a
confidence margin, else mark NEW; the **margin gate is the per-slot novelty detector**, and confidence plateau across
refinement steps is the **stop-depth** signal. Recursive variant: a net that maps (bookmark, current-path-estimate)
→ improved estimate, iterated — same properties, simpler than a full diffusion schedule; a reasonable first cut.

## 4c. Numbers / IDs — converge in embedding space, resolve IDs by snapping (never generate them)

The hierarchical list starts with a materialized `/id/id` line of **tree IDs (arbitrary numbers)**. A decoder must
NOT try to *generate* these — there's no per-number embedding and number tokenization is a known headache. The
resolution makes the problem vanish: **the decoder converges in EMBEDDING space and resolves IDs by lookup, not
generation.**
- The refinement target is the **semantic (title) content** of the path — "converge to the target hierarchical list"
  means converge the node-embedding sequence toward the target *titles' embeddings*, not reproduce the `/id` string.
- **Readout resolves identity by SNAP:** each converged node-embedding → nearest existing folder → its ID comes from
  the **lookup** (we already store folder→id). No number is ever emitted.
- **New nodes** (embedding far from any existing → margin = NEW): no ID needed — Pearltrees assigns one on creation,
  and the name comes from a Haiku call (§6). Again, nothing numeric is generated.
- The `/id` line stays an **encoder-side anchor** for μ_lineage matching (with ID-dropout); it is deliberately not a
  decoder output. (Even encoding needs no number vocabulary — e5's subword/digit tokenizer already yields *some*
  embedding per id-string, enough to act as a unique-ish anchor.)

So iteration converges *embeddings*; identity/IDs are a nearest-neighbor + lookup step at readout. This keeps the
whole decoder in the frozen-e5 embedding space and avoids number-tokenization entirely.

## 5. Training (teacher-forcing on paths; held-out-leaf for propose-new)

- **Selection:** teacher-force on the true path — same jsonL, same prefixes we already cache. At each level the
  target is the true child; the operator already learns this.
- **Propose-new supervision:** hold out some leaf (or intermediate) folders during training so the true path's
  final node is *absent from the candidate set* — the model must learn to emit NEW there instead of forcing a wrong
  existing child. This is the generative analogue of bookmark-holdout, and it's the only new data recipe needed.
- The **prefix-dropout / ID-dropout** masking we built already trains robustness to partial paths — directly useful
  for a decoder that must extend truncated/partial trees.

## 6. Naming a proposed new folder — a spectrum (retrieval-first; learned compact vocab; LLM only for the novel tail)

Embedding-space + snap needs *no vocabulary for placement*; a vocabulary only resurfaces here, for naming a proposed
node (a slot = embedding + parent/sibling context). Mirror the placement philosophy — retrieval-first, generate only
the residual:
1. **Retrieval / template naming (no generation, no vocab).** Most new-folder names are *recombinations of known
   topics* (the folder is new; its words aren't). Try nearest existing titles / nearest inner-bookmark titles to the
   slot embedding, or a template compose ("X and Y" from the two nearest concepts). Handles a large fraction, zero OOV.
2. **Two-tier domain vocabulary + a tiny AR namer** (for the residual) — word-level with subword backoff:
   - **Tier-1: whole-word tokens from the titles.** Measured on the real harvest: **~1,804 unique *folder*-title
     words** (the natural naming pool — folder names are generic recombinations), or ~5,361 words if you include
     recurring (≥2×) bookmark words. No training — just collect them. Frequency-thresholded (≥2×) is the natural
     Tier-1/Tier-2 boundary.
   - **Tier-2: a small learned subword vocab + character floor** for the **~6,864 singleton/tail** words (don't spend
     a token each; compose from subwords, chars as the no-hard-OOV floor).
   - Optional: recurring multi-word folder n-grams as atomic units for fluent phrases.
   - **Punctuation = structure, not characters** (measured: parens in 8% of titles, hyphens 2%). *Parentheticals are
     disambiguators/glosses* ("Satire (Social forces)") — split "X (Y)" into base `X` + qualifier `Y` (Y is a
     human-readable disambiguator, same role as the id-anchor/lineage; for gloss/acronym cases Y also enriches the e5
     embedding). Naming: generate base, add "(Y)" only on a name collision (Wikipedia-style), Y from the parent
     branch. *Hyphens* split two ways: semantic compounds ("Counter-activism") stay whole (one concept), slug
     artifacts ("s243a-wikispaces") are rare singletons → Tier-2 subword backoff. For *matching*, e5 absorbs all of it
     ("20th-century" ≈ "20th century"); punctuation is a vocab/naming concern only, not a placement one.
   The full vocab is **~2k whole words + a few-hundred subword pieces** — on-device-trivial (fits the PWA ethos, no
   mandatory LLM). And Tier-1 is mostly **SELECTION not generation** (mirror placement): pick the folder-words nearest
   the slot embedding, order them; the tiny AR namer composes selected words + generates only tail subwords.
3. **LLM (Haiku) — only the genuinely-novel tail**, where even the learned vocab lacks coverage; or a one-tap user
   confirmation. Small, because placement already succeeded and naming is recombination most of the time.

Honest caveat: the compact vocab reintroduces OOV *only at the naming tail* (truly novel topics); the char floor
prevents hard failure but can be awkward there — which is exactly where step 3 earns its place. Older option:
**cluster naming** — accumulate unfiled bookmarks whose slots land near the same new-node embedding and name the
cluster (any of steps 1–3), better when proposing folders in batch.

## 7. Evaluation (the genuinely hard part — scope before building)

No ground truth for "propose new," so:
- **Held-out-folder recovery:** hide a real leaf; does the decoder (a) get its parent path right and (b) propose a
  new-node embedding close to the hidden folder's embedding, with a sensible name? Measurable via path-overlap on the
  prefix + embedding distance on the proposed leaf.
- **Path-overlap with partial credit** (already have): correct prefix + reasonable new leaf scores well.
- **Human/LLM judgement** for name quality (last resort; costs budget).
Define this eval *before* building — it's the riskiest part.

## 8. Integration with the existing agent

Today: `infer_pearltrees_federated.py` retrieves folders, then **parent-walks** for the breadcrumb, then the LLM
assistant picks. A decoder would **replace the parent-walk** with a learned path *and* feed the assistant a
`[create new folder here]` candidate — turning "file into existing" into "file, creating structure as needed."
Intermediate proposed nodes remain actionable exactly as the retrieval ones (searchable by name / resolvable by id).

## 9. Smallest first step (when we pick this up)

Not now. When we do, the smallest viable cut is the **recursive** variant of §4b (simpler than a full diffusion
schedule): a small net mapping `(bookmark, current-26-slot-path-estimate) → improved estimate`, iterated a few steps,
on top of the trained LINEAGE operator. Reuse `μ_lineage` for per-slot **snap-or-new** (margin = novelty), start the
estimate from the retrieval top-1's path (already a decent partial solution), and refine. Eval = **held-out-leaf
recovery** (scope this first — riskiest part). Naming via a single Haiku call per proposed node. Diffusion is the
richer version if the recursive first cut shows the paradigm works. Net new cost: one small refinement head + one
eval recipe — everything else (operator, margin gate, prefix training, embeddings) already exists.
