# Session handoff — μ model · bridge detector · clustering (state + decisions)

*Cross-thread coordinator's note. The per-thread detail lives in `prototypes/mu_cosine/README.md` (the μ
status table + reports) and `docs/design/WAM_RUST_*` (theory + specs). This captures the **strategic
state, open PRs, standing decisions, and the gating move** — the things not in any single doc.*

## One-line state
**The method is proven end-to-end; the *data* (the category graph) is the one ceiling left.** Everything
converges on **graph widening** (`prototypes/mu_cosine/DESIGN_graph_widening.md`, #3313).

## Thread status
- **μ model (`prototypes/mu_cosine/`):** ✅ proven. Directional `MuAttention` (frozen e5 + learned
  operator/anchor/ancestry/provenance tags + 2-layer attention). WIKI directional order-acc ~99% (control
  50%); physics SYM **+0.838** (> +0.726 control); 4-domain discrimination strong (Phys/Chem/CS clean,
  Math thin); provenance token wired (maskable, masked-by-default). **Data-ceiling-limited** — see below.
- **Bridge detector (WAM-Rust core, `boundary_cache.rs.mustache`):** ✅ **complete + portable**. Three
  signals: fan-out → `n_eff` diversity (§5g) → in-domain-fraction purity (#3311) with **self-calibrating
  `τ_pure`** (#3315). Spec: `docs/design/WAM_RUST_BRIDGE_DETECTOR_SPECIFICATION.md`.
- **Bridge-informed clustering + declaration (`WAM_RUST_BRIDGE_CLUSTERING.md`):** ✅ increments **1–3
  done & e2e-verified** (swipl+cargo). `category_bridge_score`/`bridge`/`category_cluster`/`cluster_members`
  exposed to Prolog; `graph_analysis/4` collapses the per-predicate boilerplate (term+byte-identical to
  hand-wired). **Increment 4** (cross-check vs the Python `J=D/(1+H)` tool) is the only remainder — **defer
  to post-widening** (on the dense slice clustering gives one 8244-node blob; the cross-check needs a
  modular graph).
- **Theory (`WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §5f–§5g):** node-gated IC fixes the 96.7%→0.1% Lin
  saturation; `n_eff` = the structural twin of the hierarchy objective's `H`; parent-relative overlap +
  the bridge pipeline (MICA proposes → `n_eff` disposes → μ separates bridge vs leak).

## Open PRs to merge (check state; some may already be merged)
- **#3309** bidir walk — merge **after** fixing the false "95% cyclic" claim in its report (the graph is
  **0.6% directed-cyclic** — a clean DAG; the 95% was *undirected* cycle membership, irrelevant).
- **#3313** graph-widening spec · **#3316** bridge-clustering design · **#3322** `graph_analysis/4`.
- Various μ data PRs (#3310/#3312/#3314/#3318/#3320 and the engineering fine-tune-with-replay round) — most
  likely merged; the README table tracks what's landed.

## Standing decisions (don't re-derive these)
1. **The graph is a thin slice.** `data/benchmark/10k/` = **Simple English Wikipedia**, **Physics-rooted,
   10k *articles* → 8,247 categories** (of simplewiki's ~76,740). Physics-dense; **AI absent**; Math/CS/
   Engineering thin; `desc(Physics) ≈ the whole slice` (densely cross-linked). Source DB
   `data/simplewiki/simplewiki_categories.db` is **referenced but not in the repo** — widening needs it back
   (or escalate to enwiki for deep AI/CS).
2. **More data *in* a domain helps it; more *domains* dilute** (fixed 1.2M-param head). Confirmed both ways:
   physics SYM rose with more physics data (+0.479→+0.695→+0.838) and *regressed* under breadth
   (+0.695→+0.570 at 4 roots). **Multi-task is load-bearing** (single-task SYM collapses to +0.000).
3. **Training methodology:** **fine-tune with replay** (load checkpoint, mix new + a replay sample of the
   cumulative `mu_pairs_scored_*.tsv`, lower LR, forgetting-eval physics holds ~+0.838) — *not* full
   retrain, as training time grows. Ablate fine-tune+replay vs full-retrain once to confirm parity.
4. **Sampling:** μ-coherence pools = depth-≤3 downward closure ∩ calibrated-e5 μ (full closure explodes).
   Multi-root downward + cross-domain strata; bidir walk has a depth dial (`γ`-family,
   `DESIGN_bidirectional_walk.md`); `coinflip` (per-node `c/p`, exact zero-drift) beats the hub-dominated
   `global E[c²]/E[p²]`.
5. **Budget discipline:** ~82k Haiku tokens / 5h window (≈220k Sonnet-equiv × 3.75). ONE inline subagent,
   items in the prompt, scores in the reply, **no file I/O**, batch ≥40. Labels bought once, committed.
6. **Discrimination metric:** physics "brittleness" (seed-sensitive argmax) is mostly a **metric artifact**
   — central-domain nodes are genuinely high-μ-to-several-roots; **ranking is robust**. Report margin/rank,
   not just argmax. Missing modern subfields (Quantum/Stat-mech/Relativity) are a graph problem.
7. **#3281 is superseded/closed** (MiniLM-direct map). If it resurfaces in a branch, exclude it.

## The gating move + post-widening queue
**Graph widening** (#3313): re-extract a **multi-root, per-domain-balanced** slice from the simplewiki DB
(or enwiki). Drop-in output format. After it lands, in one sweep:
- μ model gains Math/CS/Engineering depth + AI; the provenance token starts carrying real signal (mix
  simplewiki+enwiki);
- bridge detector recalibrates `τ_pure` automatically and produces clean clusters (top-level leak-removal
  *will* separate on a modular graph);
- **increment 4** (the `n_eff ≈ H` cross-check vs the Python `J=D/(1+H)` clustering) becomes meaningful;
- bridge-guided sampler seeding (philosophy use case 6) and leak-pruning become worthwhile.

## Standing constraints (process)
Repo **s243a/UnifyWeaver** only. GitHub via `mcp__github__*`. Each agent: branch from `main`, own PR,
swipl+cargo e2e for WAM-Rust changes (render mustache at **edition 2021**; edition 2024 hits a pre-existing
match-ergonomics error). Commit trailers + PR footer per the session config. Do **not** put the model id in
commits/PRs.
