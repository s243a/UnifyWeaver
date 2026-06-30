# Wikipedia sampling strategy (μ-attention coverage)

How to extend the μ model's training coverage from Wikipedia — **what to sample, where to get it (DB vs API),
which tools to call, and how to build the DB if it doesn't exist.** Grounding result: coverage is the
*bitter-lesson* lever (`DESIGN_model_applications.md` — with enough in-domain data the transformer beats raw
e5-cosine; the win concentrates in trained regions). Prioritise **STEM-adjacent** domains (linguistics, political
science, social sciences, economics, philosophy — close to AI/science) over low-interest ones (music, cooking,
entertainment), which can be added later at lower budget for breadth.

> **Measured caveat (coverage round 1, linguistics/poli-sci/STS):** adding clean, well-separated domains gave
> **no μ gain** — μ stayed *below* e5-cosine there (new-domain mu-super MRR 0.39 vs e5-cos 0.53; recall@10 0.62
> vs 0.71), because **e5 already handles clean content well** (cf. "Music" generalising). The bitter-lesson
> "more data → μ wins" is **conditional on e5 being *weak* in that region.** So **prioritise coverage by
> e5-weakness** (the §5.2 disagreement signal / domains e5 *conflates*, e.g. Cooking→movies, or the dense STEM
> core where directional structure matters) — **not** "absent-but-clean" domains. Pick seeds where e5⇄μ disagree,
> not just where the graph is thin.

---

## 1. Data sources — DB first, API only for the gap

| source | what | when to use |
|---|---|---|
| **Category DB** (preferred) | `data/benchmark/enwiki_named/category_parent.tsv` — **title-based** `child<TAB>parent`, 9.9M edges, content-rooted (≈0 Tier-1 admin) | category hierarchy / subcategory edges — the default for downward sampling |
| **Page/article membership** | `cl_type=="page"` edges (article→category, `element_of`) — **in the same dump**, just not emitted by the current ingest | leaf categories (see §4); a graph join from the dump, *not* an API call |
| **Live MediaWiki API** | `fetch_category_pages.py` (en.wikipedia.org) | targeted leaves when the DB lacks pages, or freshness; rate-limited, polite |

**Key fact:** "pages aren't in the DB" is an *ingest-filter* artifact, not missing data. The category DB is built
from the `categorylinks` dump whose `cl_type` distinguishes `subcat` vs `page`; the ingest currently emits only
`subcat`. The page memberships are in the same dump (resolve article titles via the `page` table, namespace 0).

---

## 2. Building the DB (if it doesn't exist)

**Category graph** — `scripts/ingest_enwiki_categories.py` (correct 2024-schema 3-dump join):
```
python3 scripts/ingest_enwiki_categories.py \
    --page       <dumps>/enwiki-latest-page.sql.gz \
    --linktarget <dumps>/enwiki-latest-linktarget.sql.gz \
    --categorylinks <dumps>/enwiki-latest-categorylinks.sql.gz \
    --out data/benchmark/enwiki_named/category_parent.tsv
```
Joins `child_title = page[cl_from]` (ns 14) → `parent_title = linktarget[cl_target_id]` (ns 14), `cl_type==subcat`.
Dumps live under `context/gemini/UnifyWeaver/data/{enwiki,simplewiki}/` (page / linktarget / categorylinks
`.sql.gz`); both wikis present. **To add page `element_of`:** extend `load_titles` to also keep namespace-0
(article) titles and emit `cl_type=="page"` edges — scoped to the categories in your slice to bound size.

**Scoped slice** — `prototypes/mu_cosine/build_slice.py` (streaming BFS, the 644MB graph OOMs if loaded naively):
```
python3 build_slice.py --root Linguistics --graph data/benchmark/enwiki_named/category_parent.tsv \
    --depth 3 --apex-cap 30 --out context/slice_Linguistics.tsv
```
Downward closure (depth-bounded) + one level **up** + **apex-capped siblings** = a hub-safe **bidirectional**
slice. Depth 3 keeps drift low (depth 4 wandered: poli-sci → `1946 in aviation`); apex-cap bounds hub explosion.

---

## 3. Category sampling → training data

1. **Slice** the seed domains (`build_slice.py`, §2). Observed: clean — 0% Tier-1, ~0% Tier-2, ~97% new content.
2. **Convert to a graded round** (directional WIKI operator — what the membership eval measures), the
   graph-truth route (no LLM scoring needed):
   - slice is `child<TAB>parent`; the fused-edge convention is **`a_key=container/parent`, `b_key=member/child`**
     (`build_graded_round.py:17`), so emit `parent<TAB>child<TAB>subcategory<TAB>1.0`.
   - `build_graded_round.py --fused context/<prefix> --out <graded>` → `subcategory→WIKI, μ(member|container)=0.90`.
3. **Train** (warm-start, anti-forgetting via SYM replay; the established recipe):
   ```
   UW_MU_GRAPH=<merged_category_parent.tsv> python3 train_mu_attention.py \
       --init-from model_nodetype.pt --graded <graded>_pairs.tsv --use-nodetype \
       --pairs mu_pairs_scored_cumulative.tsv --steps 500 --bs 128 --lr 3e-4 --device cuda --save model_cov.pt
   ```
4. **Eval** on the **merged** graph (new domains as candidates): `eval_filing.py --source simplewiki
   --drop-admin junk` + `--holdout-nodes <slice_nodes.json>` to focus on the new domains; compare vs e5-cos and
   vs the pre-coverage checkpoint.

**SYM route (alternative):** `gen_mu_pairs.py --seeds … --bidir-frac 0` (downward) emits *unscored* candidate
pairs — needs a scoring pass (μ column). Prefer the graded route for tagged category structure (no scoring).

**Downward-vs-bidirectional ratio is the primary admin control** (`gen_mu_pairs --bidir-frac`): downward-from-
content stays in content (admin is reached going *up*); `0` = pure child-only. On a content-rooted/admin-flagged
DB, bidirectional becomes safe (reaches siblings/cousins for lateral coverage).

### Full training recipe (judge→loss routing) — ~70% directional + ≤30% lateral
The validated recipe that makes μ **beat** a trained e5-probe on direction/close-neg (REPORT §4.6):
- **~70% downward DIRECTIONAL (`graph` judge)** → **ranking** loss, *not* regression. 1-hop edges via
  `build_graded_round` (`subcategory`/`element_of`, `μ(member|container)=0.90/rev 0.10`); multi-hop via
  `transitive_closure.py` → `--transitive --transitive-hetero` (hop-aware variance, keeps hop distance). Train
  with `train_mu_attention.py --graded <dir> --dir-rank-weight 1.0 --sym-weight 0` (drops the order-invariant SYM
  pressure that competes with direction). **DONE + integrated.**
- **≤30% bidirectional LATERAL (`haiku` judge)** → **soft regression** to a Haiku operator **superposition**
  (siblings from the merged graph, scored by a Haiku judge on the §14 superposition prompt). The judge token
  routes the loss (graph→rank, haiku→regress) — `--dir-rank-weight` keys off `r[8]=="graph"`; `haiku` rows fall to
  the regression pool automatically. **DONE (proof-of-pipeline):** sampled sibling pairs → Haiku scored the
  superpositions → appended as `judge=haiku` `SYM` graded rows → `train_mu_attention.py --graded <dir+lateral>
  --dir-rank-weight 1.5 --graded-weight 0.3 --transitive … --sym-weight 0`. The graded round shows the routing
  (`51,632 WIKI → rank`, `SYM haiku → regress`); the full recipe trains end-to-end and **close-neg beats the
  e5-probe (0.80 vs 0.78)**, direction ~0.88 (vs the sole-objective standalone ceiling 0.982 — the multi-objective
  recipe trades some peak direction for lateral relatedness + transitive + calibration in one model). Scaling the
  lateral to a true ≤30% needs a larger Haiku batch (the pipeline is the same; this run used a 17-pair proof
  batch).

---

## 4. Page sampling — for leaf categories (`element_of ≈ subcategory`)

Thin domains are **leaf-heavy** (measured: 81% of a linguistics/poli-sci slice were leaves with no subcats). For
leaves the subcategory signal is exhausted; the rich signal is the **member pages** (`element_of`), which
correlates with subcategory membership. For the μ model, "page data" = the **page title** (e5-embedded) + the
`element_of` edge — no page *content* needed, so a fetch is cheap (≤500 titles/call) and the dump route is a join.

- **Preferred:** build scoped page `element_of` from the dump (§2 extension) — all memberships within a slice.
- **API:** `fetch_category_pages.py --cat <Cat> [--recurse-subcats N]` → `page<TAB>category<TAB>element_of`.

---

## 5. Which categories/leaves to mine for pages — the prioritisation funnel

Cheap → expensive. With the dump local, fetching is *not* the bottleneck, so most of this prioritises **which
edges to keep/weight**, not which to fetch. (When restricted to the live API, it prioritises the fetch.)

1. **Structural prefilter (≈free, local):**
   - **Article-count / richness** — rank by #articles (count rows per category in `article_category.tsv`); skip
     near-empty leaves, prioritise content-rich.
   - **Centrality / multi-parent** — leaves classified under many parents are bridging/important.
2. **Informativeness — disagreement (μ vs an independent judge), *refined by model confidence* (§6):**
   - Score each candidate's μ(node|root) vs an independent signal — **e5-cosine**, **Haiku**, and/or
     **ModernBERT** (an independent encoder = a useful ensemble member; reuses the dense∩greedy reliability idea).
   - **High disagreement + LOW model confidence (wide error bounds)** ⇒ **undertrained region → sample it.** This
     is the common case and the main yield (Haiku is right, we're thin). *Disagreement is mostly undertraining,
     not Haiku noise.*
   - **High disagreement + HIGH model confidence (judge outside the bounds)** ⇒ genuine conflict / possible tail →
     **escalate to a stronger model** (§7), don't blindly sample.
3. **Quality gate — category↔page coherence:** among informative picks, keep those whose member pages cohere with
   the category title (e5 sim) → informative *and* clean; drops genuinely-noisy categories. (Opposite objective to
   #2, so apply as a filter, not a selector — it needs the titles first.)
4. **Diversity:** cluster candidate e5-embeddings, take one per cluster → max breadth per budget, avoid redundancy.
5. **Bridge leaves:** high *cross-domain* μ (the bridge detector, PR #3322) — best for lateral/operator-
   superposition coverage.

---

## 6. Model confidence / error bounds

The μ readout is a **sigmoid point estimate** (no native variance), but per-prediction error bounds are derivable
cheaply — used as the disagreement discriminator in §5.2:
- **Operator-superposition spread** — `Var[μ]` across SYM/WIKI/ELEM (already computed per operator); high spread =
  relation-type ambiguity = low confidence.
- **Ancestor-sampling MC** — lineage is sampled stochastically; K forwards with different sampled ancestors → a μ
  distribution → σ (MC-dropout-style, no retrain).
- **Transitive product-propagated variance** — `V = Σ_links (1−μ)/μ` for multi-hop chains (built;
  `DESIGN_transitive_relations.md`).
Compare `|μ − μ_judge|` to ~2σ: within ⇒ consistent-given-uncertainty (undertrained, sample); outside ⇒ conflict
(escalate).

---

## 7. Judges & escalation (Haiku is trusted except in the tail)

- **Haiku = source of truth for high-μ values**; weak **only in the tail** (low-μ / no-relation / borderline,
  measured ~80% noise *there*). So trust Haiku for positive/membership signal; treat its tail judgments as noisy.
- **Disagreement ≠ tail.** A μ⇄Haiku gap is usually **undertraining** (sample), not Haiku error — use §6 confidence
  to tell them apart.
- **Escalation:** when a case *is* flagged tail (low μ + confident model + Haiku conflict, or high inter-judge
  entropy), escalate to a **stronger model** (Sonnet, then Opus). Budget: Sonnet ≈ ¼ Haiku, Opus less; the project
  has headroom (budgeted at 10% of the usage limit). Tag the judge (`JUDGES` in `mu_attention.py`).

---

## 8. Admin / maintenance categories (two tiers)

- **Tier-1 (meaningless):** maintenance/template/nav (`CatAutoTOC`, `Navseasoncats`, …) — procedural titles, zero
  topical content. **Drop** (eval + training). Trivially small (≈0.7% of training) and downward-from-content
  rarely reaches them; the content-rooted DB excludes most.
- **Tier-2 (loosely semantic):** structural/temporal `by-year/country/nationality`, `Establishments`. Real but
  thin meaning. **Keep in eval; down-sample (not drop) in training.** (~22% of training — down-sampling frees real
  capacity.) Filter helper: `eval_filing.is_admin(name, level="junk"|"all")`.

---

## 9. Tools index

| tool | purpose |
|---|---|
| `scripts/ingest_enwiki_categories.py` | build the title-based category DB from the 3 dumps (extend for `cl_type=page`) |
| `prototypes/mu_cosine/build_slice.py` | streaming BFS slice (downward + apex-capped bidirectional) from named seeds |
| `prototypes/mu_cosine/fetch_category_pages.py` | live-API page `element_of` for targeted/leaf categories |
| `prototypes/mu_cosine/build_graded_round.py` | fused subcategory/element_of edges → directional graded targets |
| `prototypes/mu_cosine/gen_mu_pairs.py` | walk-based SYM candidate pairs (needs a scoring pass) |
| `prototypes/mu_cosine/train_mu_attention.py` | train (`--init-from` warm-start, `--graded`, `--pairs`, `UW_MU_GRAPH`) |
| `prototypes/mu_cosine/eval_filing.py` | membership eval (`--source simplewiki`, `--drop-admin`, `--holdout-nodes`) |

*(Superseded: `prototypes/mu_cosine/sample_enwiki_downward.py` — a downward-only reimplementation of
`build_slice.py`; prefer `build_slice.py`, which also does the safe bidirectional context.)*
