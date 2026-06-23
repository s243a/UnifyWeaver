# Calibrated judges, semantic drift, and range-valued cross-entropy targets

*Design note for the `mu_cosine` MuAttention prototype. Theory + how it maps onto this codebase +
implementation notes. Prototype-only; no WAM-Rust core changes implied.*

## 0. TL;DR

The model is supervised by several **judges**, each answering a different question about a pair `(child,
parent)`:

- **graph** judge (enwiki / mindmap structure) — *is there an edge, and which way does membership point?*
- **haiku** judge (bought LLM grades) — *how much does this actually mean, semantically?*
- **human** judge (SimpleMind mindmaps — proposed) — a hand-curated semantic edge, typed.

Today the graph judge uses a **hard-coded point target** (`μ(child|parent) → 0.9`). That `0.9` is an
**uncalibrated assumption of 0.1 semantic drift** — nobody measured it. This note argues for two changes:

1. **Calibrate the target empirically** — fit `μ̂ = f(structure)` from the Haiku grades we already own;
   the *residual variance* of that fit **is the semantic drift, quantified**.
2. **Make the target a range, not a point** — supervise with inequalities / bands (cross-entropy or
   hinge), which is the honest encoding of label uncertainty *and* makes multi-judge fusion
   non-adversarial.

Both reuse existing labels (no new Haiku spend for the enwiki arm) and stay inside the prototype.

---

## 1. The problem: structure is not semantics

A Wikipedia category edge `child → parent` is a **structural** fact. It tells you the two categories are
linked and (with direction) which is the parent. It does **not** tell you the edge is *semantically* a
membership: enwiki's broad parents connect genuinely unrelated nodes. From the cybernetics round, all of
these are real graph edges (reachable through shared parents) yet are semantic junk:

- `Organizational_cybernetics → … → Galaxies`
- `Feudalism → … → Recursion`
- `Tragedy_of_the_commons → … → Symbolic_interactionism`

The graph judge, trained to push every real edge toward `0.9`, would score these ~0.9. Haiku scored them
**0.0**. **That gap is the semantic drift** — the difference between graph-adjacency and semantic
membership. It is exactly what a semantic judge (Haiku, or a human map) buys over structure alone, and it
is why we cannot fully eliminate Haiku: the graph cannot self-supply the correction for its own noise.

Framing the two roles crisply:

> **The graph answers "is there an edge, and which direction." The semantic judge answers "how much does
> it actually mean, and is the edge real."**

---

## 2. What this codebase already has

Three operators (`mu_attention.py:41`): `OPS = {"SYM": 0, "WIKI": 1, "LLM": 2}`. Two judges
(`mu_attention.py:49`): `JUDGES = {"haiku": 0, "graph": 1}`. Two corpora (`mu_attention.py:48`):
`CORPORA = {"simplewiki": 0, "enwiki": 1}`. Provenance is a factored, maskable token
(`corpus_emb + judge_emb + prov_tag`), masked-by-default.

| operator | judge | supervision today | source |
|---|---|---|---|
| `WIKI` | graph | `bce(μ(p\|c),0) + bce(μ(c\|neg),0) + wiki_abs·bce(μ(c\|p),0.9)` **+ directional margins** | free graph edges |
| `SYM` | haiku (pos) / graph (neg) | order-invariant `MSE(μ, μ_haiku)`; negatives are free non-edges at μ=0 | `mu_pairs_scored_*.tsv` |
| `LLM` | haiku | frozen boundary fixture | `tests/fixtures/wikipedia_physics_boundary_haiku.tsv` |

Two things to notice in the **exact** `WIKI` loss (`train_mu_attention.py:219–223`):

```python
L_wiki = (bce(mu_pc, 0) + bce(mu_cpn, 0)                       # pin wrong-order + random-neg to 0
          + args.wiki_abs * bce(mu_cp, 0.9)                    # <-- the hard-coded point target
          + args.margin_weight * (F.relu(m - (mu_cp - mu_pc)).mean()      # directional inequality
                                  + F.relu(m - (mu_cp - mu_cpn)).mean()))  # vs random-neg inequality
```

- The `0.9` is the assumed-drift constant. Everything else in the block is already **inequality /
  hinge** supervision (`F.relu(m - …)`), i.e. *the codebase already supervises ranges* — just for
  *ordering* (direction), not for *magnitude*. Extending bands to magnitude is the same machinery.
- The `wl` column (graph walk-length / depth) is written into every pair file by `gen_within`, but
  `load_pairs` (`train_mu_attention.py`) reads only `(a, b, μ)` — **`wl` is currently unused metadata.**
  The structural feature we want to calibrate against is already on disk, just discarded.

---

## 3. Calibration: turn the assumed drift into a measured one

We already own the joint data needed to fit it: every Haiku-scored **downward** pair carries, on the same
row, both its **graph features** and its **Haiku μ**. So fit

```
μ̂ = f( depth(wl),  log child-degree,  #distinct-parents,  e5-cos(child,parent) )
```

over the ~5k graded positives, and use `μ̂` as the target for the millions of un-graded free edges.

- Start trivially: empirical **mean + variance of Haiku μ binned by depth**. Then add features.
- The high-value cheap feature is **`e5-cos(child,parent)`** (from the cached e5 tables): it's "graph +
  embedding" calibration with **zero LLM**, and it is precisely the axis that separates a tight edge from
  a junk edge *at the same depth* (it's what would catch `→Galaxies`).
- `μ ∈ [0,1]` graded ⇒ the natural likelihood is **Beta**; fit a Beta regression (or quantile
  regression) so each `(structure)` maps to a full conditional **distribution**, not just a mean.

**The residual is the drift.** `μ_haiku − μ̂` is the part of membership the structure cannot predict — the
per-edge semantic drift. Its **variance per stratum quantifies how much Haiku is irreplaceable** and
*where*:

- low residual variance (expected at depth-1 direct children: reliably high μ) ⇒ the graph rule is
  trustworthy ⇒ **skip Haiku there**, let the calibrated graph judge carry the label;
- high residual variance (the cross-field boundary strata) ⇒ the fit cannot substitute ⇒ **Haiku budget
  is irreducible there.**

This is the same drift measurement as "train `WIKI`-only, correlate against held-out Haiku grades," reached
more cheaply and with per-stratum resolution.

---

## 4. Range-valued targets (the key refinement)

Cross-entropy does **not** require a point target. You can supervise a **band or an inequality** and leave
everything inside it free. This is strictly better than a point whenever you are uncertain of the exact
value — which, for a calibrated judge, is *always* (the fit has variance). Three forms, increasing
elegance:

1. **One-sided hinge** — penalize only outside the band: `relu(lo − μ)` and/or `relu(μ − hi)`; zero loss
   inside. (Same `F.relu` the `WIKI` margins already use.)
   - direct child → `μ ≥ 0.7` (lower-bounded, **not** pinned to 0.9)
   - non-edge negative → `μ ≤ ε`
   - "See Also" associative → two-sided `μ ∈ [0.2, 0.7]`
2. **Distributional CE** — fit a **Beta(α,β)** per `(edge-type, depth, e5-cos bin)` and minimize
   cross-entropy / KL toward it. **A high-variance Beta *is* a range** — it penalizes the tails, not a
   point. "Calibrate it" and "make it a range" become the *same operation*: the fitted variance is the
   band width, automatically.
3. **Quantile bounds** — use a fitted quantile as a one-sided constraint, e.g. `μ ≥ q10(structure)` — the
   honest "at least this much" supervision.

Why ranges, beyond honesty:

- **Correct encoding of label uncertainty.** You supervise only what you actually know — an ordering, or a
  band — and let the model + other judges settle the interior. For "See Also," the true value genuinely
  *is* a range, not a number.
- **Non-adversarial fusion.** If the graph judge says `μ ≥ 0.7` and Haiku says `μ = 0.85`, there is **no
  conflict** — the point satisfies the inequality, both losses are zero. **Point targets fight even when
  compatible** (`0.9` vs `0.85` both pull); bands only penalize *genuine* disagreement — which is exactly
  the signal worth flagging for a Haiku tiebreak.
- It is the same instinct the code already had in choosing `0.9`-not-`1.0` to dodge the predict-the-mean
  collapse, generalized: **bound what you know; don't invent precision.**

---

## 5. The mindmap (SimpleMind) as a second, typed graph judge

The user's SimpleMind `.smmx` maps are a hand-curated semantic graph keyed by Pearltrees slug. They slot in
as `CORPORA["mindmap"]` + `JUDGES["human"]` with **no architecture change** (the provenance token already
supports new codebook entries). Two facts make the mindmap more than a copy of the enwiki arm:

**(a) Lower drift on hierarchy edges.** A human-curated `parent → child` is more reliably a true membership
than an enwiki category edge (which can be administrative noise). So mindmap hierarchy edges should
calibrate **higher and with tighter residual variance** than enwiki's `0.9` — less drift to correct.

**(b) Edges are typed — the target depends on edge type, and so does the *operator*.** From the parsed maps:

| edge type | what it means | operator | prior band (calibrate!) |
|---|---|---|---|
| `parent → child` (scaffolding skipped) | membership, directional | `WIKI` | `μ ≥ q_lo` (high) |
| `cloudmapref` + `element` GUID | node *is* a topic in another map → near-equivalence | `WIKI` | `μ ≥ q_hi` (~0.95) |
| `See Also` / `Via Link` / `Related` | **symmetric association**, not membership | `SYM` | `μ ∈ [≈0.2, ≈0.7]`, **wide σ** |

Note the third row is **symmetric** ⇒ it belongs in the order-invariant `SYM` operator, **not** the
directional `WIKI` margin. So `edge_type` doesn't merely scale a target — it selects which operator the
edge trains. That makes it the *dominant* feature.

For "See Also" specifically: `0.4` is a reasonable *prior mean*, but it is the edge type where guessing is
most dangerous, because (i) its true variance is wide (some are near-siblings, some distant), and the wide
`σ̂` is the right outcome — precision-weighting `1/σ̂²` then makes these low-confidence edges contribute
little; and (ii) it is the one type the cheap "bootstrap from enwiki overlap" route **cannot** calibrate,
because associative links are exactly the cross-cutting edges enwiki's tree *lacks*. So "See Also" is where
a **small one-time Haiku fixture** (≈40 edges per type, like the existing boundary fixture) earns its keep.

Scaffolding nodes (`See Also`, `Via Link`, `Super Categories`, `Related`, blanks) are **layout containers,
not topics** — the parser must skip them and pass the edge through (grandparent → grandchild).

**Calibration sources for the mindmap (no full Haiku grading):**

- **Bootstrap from overlap** — mindmap edges whose endpoints both resolve to enwiki/e5 nodes (~30% matched
  the cyber slice) inherit a target from the existing Haiku μ / e5-cos; extrapolate to the map-only edges.
- **Small Haiku fixture** — anchor each edge type's mean+variance once; never re-spent.

---

## 6. The unified picture

Three judges, each a **calibrated soft constraint** (band/inequality) with a **measured variance**:

| judge | corpus | answers | target form |
|---|---|---|---|
| `graph` | enwiki, simplewiki | structure + direction | `μ̂=f(struct)` band, fitted from Haiku |
| `human` | mindmap | curated semantic edge (typed) | per-edge-type band, fitted |
| `haiku` | bought | graded semantics + **disagreement tiebreaks** | point/narrow-band grades |

The variances route the work: trust the cheap judge where its residual is tight; fall back to Haiku where
it is wide or where two free judges' bands disagree. **Fusion** of a pair seen by multiple judges =
intersect their bands (precision-weighted); empty/conflicting intersection = the high-value pair to spend a
Haiku grade on.

---

## 7. Node types, relation operators & per-function templates

So far "membership" has been one undifferentiated thing. It isn't — and pulling it apart cleanly is what
lets one model carry enwiki categories, enwiki pages, mindmap nodes and Pearltrees collections without any
of them contaminating the others. Two **orthogonal** axes do the work:

- **Node-type token — what each *endpoint* is.** `NODETYPE = {category, page, mindmap_node,
  pearltrees_collection}`, a factored, maskable token exactly like `corpus_emb`/`judge_emb`. It tells the
  model the *structural role* of a node — a `category`/`mindmap_node` is an internal node that can have
  children; a `page` is a **leaf**. e5 still encodes the *title text* either way ("Hopf bifurcation" and
  "Bifurcation theory" both embed fine), so cold-start is unaffected; the node-type token only adds the
  role the title can't convey.
- **Relation operator — what the *edge* is.** Conditions the shared transformer trunk, exactly as the
  existing `OPS = {SYM, WIKI, LLM}` already do. Today's operators *conflate* membership: `WIKI` is
  structurally the subcategory-directional relation, and `SYM`'s graded template says "belong under /
  relate to" **without** distinguishing a subcategory from a page. The clean split is one operator per
  *relation function*:

| relation function | edge | node-types | shape |
|---|---|---|---|
| **subcategory-of** | `subcat ⊂ cat` | category→category | transitive, internal-node, directional |
| **element-of (page)** | `page ∈ cat` | page→category | terminal, **leaf**, directional |
| **association** | `A ↔ B` (See Also) | any | **symmetric**, order-invariant |

### Per-function Haiku templates ↔ operator embeddings

The key coupling: **the Haiku prompt template is the human-readable *definition* of a relation; the
operator embedding is the model's learned, compressed version of that same definition.** One template ↔ one
embedding. To add a relation you write a template that *defines* it — e.g.

- *subcategory* template: "B is a **subcategory** of A — a narrower field wholly contained in A's topic."
- *element/page* template: "B is a **direct member page** of A — an article that belongs in A's topic."

— and the labels each template produces train *its own* function token. The model then triangulates the
shared "membership" manifold **from the differences between the templates**.

### Why per-function embeddings (not one, not N models)

Giving each function its own token on a **shared trunk** is the sweet spot between two worse designs:

- **one merged operator** → loses the distinction (can't tell "is a subfield of" from "is an article
  about"), and
- **a separate model per relation** → loses all transfer; each relation relearns "membership" from
  scratch.

Shared body + per-function token = the trunk learns the common membership geometry **once** (the part
that's the same whether B is a subcat or a page), while each operator embedding only has to encode the
**delta** (subcat: transitive / has-its-own-children; page: terminal / leaf). That is multi-task learning,
and the function tokens are *where the differences live* so the weights can stay shared — **"let the model
find what generalizes between the functions."**

This composes with the rest:
- **Calibration is per function** (§3–4): each operator gets its *own* target band, because a direct
  subcat and a listed page genuinely sit in different μ-ranges. "Different embeddings per function" and
  "calibrated range targets per function" are the same axis.
- **Operator ⟂ node-type**: the operator says what the *relation* is; the node-type token says what each
  *endpoint* is. Together they express enwiki subcats, enwiki page-memberships, and mindmap edges in one
  model, each tagged so none bleeds into another.

### The page frontier (why this matters *now*)

Page membership isn't a new data source — it's the **same dump, filtered wider**: our ingest kept only
ns14→ns14 (category→category) edges; the page→category memberships are the *same* `categorylinks` table
with an ns0 (article) child. Bulk-ingesting all of it is tens of millions of rows ("a lot of data"), so the
pragmatic path is **scrape-on-demand**: pull the pages of *one* category only when filling a specific gap.

The gaps that need it are the **thin-or-absent categories**: `Bifurcation_theory` (0 subcats, 20 pages),
`Nonlinear_systems` (1 subcat), `Circuit_theory` (not an enwiki category at all). These are exactly the
nodes that are *unlearnable at the category level but rich at the page level* — so the page node-type and
the element-of operator are the **proof case** for this whole axis, not a someday-nicety.

The **Pearltrees harvester** (`.local/tools/browser-automation/`, `getTreeAndPearls?treeId=…`) is the page
on-ramp for nodes enwiki lacks. Its `contentType` maps straight onto the schema: `2`=Collection (a
`pearltrees_collection` node), `1`=PagePearl (a `page` node — **and it carries the pearl's `url`, usually
the Wikipedia article**), `5`=Shortcut (cross-tree alias = a `cloudmapref`-like edge), `7`=Section
(scaffolding — skip). So harvesting one collection (e.g. *Circuit Theory* `id13580844` →
{Mechanical-electric analogs, Phasors, Laplace Transform Circuits, 1st/2nd-Order Linear Circuits}) yields,
in one shot: **collection→page membership edges** (Haiku-judgable now under the element-of template) **and
each page's Wikipedia anchor** (the join key back into the enwiki page→category graph) — the exact tie-in
surface, for a topic enwiki has no category for.

### Current Haiku template (for reference)

The grading prompt has an **invariant skeleton** — directional framing ("grade B relative to A's topic"),
the six bands (1.0 nested → 0.0 unrelated/junk), a strictness paragraph naming the junk to zero, and a
no-tools / exact-TSV-output contract — and **per-round variables**: the region label, the in-band
examples, and the named junk. It runs as 2 parallel Haiku subagents (~17k tok each). A **per-function**
variant adds exactly one clause to the skeleton naming the relation ("B is a *subcategory* of A" vs "B is a
*member page* of A"); everything else stays, which is what keeps the function tokens comparable.

---

## 8. Implementation notes

**Files & hooks**

- `train_mu_attention.py` WIKI loss (`~:219`) — swap `args.wiki_abs * bce(mu_cp, 0.9)` for a
  band/distributional term against `μ̂(features)`; keep the existing directional margins untouched.
- `load_pairs` — start reading `p[3]` (`wl`/depth) instead of discarding it.
- e5-cos feature — read from the cached `e5_*_cos.pt` / e5 tables (already on disk per slice).
- New codebook entries when the mindmap lands: `CORPORA["mindmap"]`, `JUDGES["human"]`
  (`mu_attention.py:48–49`) — additive, no shape changes elsewhere thanks to the maskable token.

**Calibration script (first brick — enwiki only, reuses existing labels)**

1. Join every committed `mu_pairs_scored_*` row with its structural features: `depth` (`wl`), child/parent
   degree (from the slice graph), `#distinct-parents`, `e5-cos(child,parent)` (from cache).
2. Fit the **conditional distribution** `μ | features` — empirical quantiles per bin to start; then a Beta
   / quantile regression. **Output quantiles, not just a mean**, so targets come out as bands.
3. Report: the `depth → μ` calibration curve, the `e5-cos → μ` curve, and the **per-stratum residual
   variance** (the drift table). This single pass yields both the empirical rule *and* the drift number.
4. If the fit holds, wire the band target into the `WIKI` loss (hinge or Beta-CE), precision-weighted by
   `1/σ̂²`.

**Loss sketch (band form)**

```python
# μ̂_lo, μ̂_hi = fitted quantiles for this edge's (type, depth, e5cos) bin; w = 1/σ̂²
L = w * (F.relu(mu_hat_lo - mu_cp) + F.relu(mu_cp - mu_hat_hi)).mean()   # zero inside the band
# direct child: hi=1.0 (lower-bounded only)   negative: lo=0.0 (upper-bounded only)
```

**Mindmap arm (follow-on)**

- `parse_smmx.py` — walk the whole Dropbox subtree at once (cloudmaprefs are **relative filesystem
  paths**; canonicalize each map to its absolute path, resolve `cloudmapref` + `element` GUID against a
  global index). Emit `slug, enwiki_alias?, edge(src_slug, dst_slug, edge_type, depth)`. Skip scaffolding.
- Add `edge_type` as the dominant calibration feature; route symmetric types to `SYM`, directional to
  `WIKI`.
- Calibrate via overlap bootstrap + the small per-type Haiku fixture.

**Operators, node-types & the page frontier (follow-on)**

- `OPS` / `NODETYPE` codebooks (`mu_attention.py`) — additive, like `CORPORA`/`JUDGES`. Split membership:
  keep `WIKI` as **subcategory-of**, add an **element-of (page)** operator μ(page|category); add
  `NODETYPE = {category, page, mindmap_node, pearltrees_collection}` as a factored, maskable token. The
  e5 path is unchanged (title text still embeds); the new token only adds structural role.
- Per-function Haiku templates — one template *defines* each relation (the human spec); its labels train
  that operator's token. Keep the invariant skeleton (§7) and vary only the one relation clause + examples
  so the function tokens stay comparable.
- Page-membership ingest — same `categorylinks` dump, widen the filter to ns0 children
  (`scripts/ingest_enwiki_categories.py`); bulk = tens of millions of rows, so prefer **scrape-on-demand**
  for thin-category gaps (`Bifurcation_theory`, `Nonlinear_systems`).
- Pearltrees on-ramp — `getTreeAndPearls?treeId=…` via `.local/tools/browser-automation/`; map
  `contentType` 2→collection, 1→page (`url` = Wikipedia anchor / join key), 5→shortcut edge, 7→section
  (skip). One harvest yields collection→page membership edges (element-of, Haiku-judgable) **plus** each
  page's enwiki anchor (the corpus bridge) — for topics enwiki has no category for (e.g. *Circuit Theory*).

**Validation**

- WIKI-only (`--sym-weight 0`) vs held-out Haiku correlation, per stratum = drift sanity check.
- Calibration reliability (are the fitted bands actually covering the held-out Haiku μ at the stated
  rate?).
- Forgetting guards unchanged (gate-leak, WIKI order-accuracy, SYM corr).

**Sequencing**

Enwiki calibration first (zero new Haiku spend, produces the drift table). Mindmap arm second (needs the
parser + a small fixture). Everything stays in `prototypes/mu_cosine/`; nothing touches the WAM-Rust core.

---

## 9. Open questions

- Best feature set beyond `{depth, degree, #parents, e5-cos}`? (branching distribution, parent's own μ,
  cross-domain-argmax-boundary indicator.)
- Beta regression vs monotone quantile regression vs simple binning — where does the extra machinery start
  paying for itself?
- Does precision-weighting by `1/σ̂²` need a floor to stop ultra-confident depth-1 edges from dominating?
- For mindmap fusion: intersect bands, or treat each judge as an independent likelihood and multiply? The
  latter is cleaner probabilistically but assumes conditional independence the two graphs may violate.
- **Operator granularity** — how many relation operators before they stop helping? (subcat-of / element-of
  / association is the obvious split; does "near-equivalence" / "cross-map shortcut" deserve its own, or
  fold into subcat-of?)
- **Shared vs per-operator readout** — should each operator share the sigmoid head (max transfer) or get
  its own (max separation)? The shared-trunk argument says share the body; the head is the open part.
- **Does page↔subcat membership actually transfer?** The whole per-function bet assumes a common
  "membership" geometry. Measure it: train element-of with subcat-of frozen-in vs from scratch — does the
  shared trunk help or do the relations want different features?
- **Node-type as a token vs separate embedding tables** — one factored `NODETYPE` token (cheap, lets the
  model interpolate) vs a per-type encoder (more capacity, no cold-start sharing). Token first; revisit if
  pages and categories turn out to need different encoders.
