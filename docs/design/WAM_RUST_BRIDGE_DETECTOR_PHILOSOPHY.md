# WAM-Rust Bridge Detector — Philosophy

*Why a bridge detector exists, what it is for, and how it fits the μ / similarity work. The **theory**
lives in `WAM_RUST_GRAPH_FUNCTIONAL_SEMIRINGS.md` §5f–§5g; the precise definitions in
`WAM_RUST_BRIDGE_DETECTOR_SPECIFICATION.md`; the build order in
`WAM_RUST_BRIDGE_DETECTOR_IMPLEMENTATION_PLAN.md`. This doc is the **why**.*

## The question

On a DAG like the Wikipedia category graph, which nodes are *meaningful bridges* — articulation points
where genuinely distinct sub-domains connect through one organizing category — and which are *leak
conduits* — generic apex categories (`Main_topic_classifications`, `Categories`) that connect everything
indiscriminately and pollute the gated cones?

## The same problem, two faces

We already met the bad face in the real-data work (`WAM_RUST_CARET_REALDATA_MEASUREMENT_2026-06-18.md`,
cone-purity addendum): leak conduits — low-fan-in generic apexes with low cone purity — leak membership
across unrelated domains, which is the whole reason μ-gating exists. The bridge detector is the **positive
framing** of that same structure: instead of only flagging leaks to prune, it identifies the nodes that
*legitimately* organize a domain into distinct subfields. Diagnosing bridges and diagnosing leaks turn out
to be the **same computation read two ways** — see "the 2-D map" below.

## The insight: high branching is not high *bridging*

A node with many children is not automatically a bridge. There are three cases, and only one is a bridge:

- **Redundant hub** — many children, but their subtrees *overlap* (reconverge). The node isn't organizing
  distinct things; it's a pile of near-duplicates. *Low branch diversity.*
- **Leak conduit** — many children carving genuinely distinct regions, but the regions are *unrelated*
  (out-of-domain). The node connects everything; that is exactly a leak. *High diversity, low domain
  coherence.*
- **Meaningful bridge** — many children carving distinct regions that are *in-domain*. *High diversity,
  high domain coherence.*

So a bridge needs **three** signals, not one (each from a primitive that already exists):

1. **Fan-out** — is it a branch point at all? (degree / merge-frequency)
2. **Diversity of branching** — are the branches *distinct* or redundant? (`n_eff`, §5g)
3. **Domain coherence** — are the branches *in-domain* or junk? (cone purity / μ, the leak detector)

This is the synthesis from the theory note: **MICA proposes** (a high-fan-out merge point is a candidate),
**overlap disposes** (`n_eff` says whether the fan-out is diverse), and **μ judges** (purity says whether
it is a real bridge or a leak conduit).

## The 2-D map (bridge vs leak vs hub)

Diversity (`n_eff`) and domain coherence (cone purity / μ) are **orthogonal**, and together they classify
every high-fan-out node:

| | **high purity (in-domain)** | **low purity (out-of-domain)** |
|---|---|---|
| **high `n_eff` (diverse)** | **meaningful bridge** — organizes distinct subfields | **leak conduit** — connects distinct *unrelated* regions → prune / down-weight |
| **low `n_eff` (redundant)** | redundant hub — children overlap; candidate to collapse | weak generic node |

That single table is why the bridge detector and the earlier cone-purity leak detector are the same tool:
**leaks are one quadrant of the bridge map.**

## How we might use it

1. **Principled leak pruning / cone cleanup.** The high-`n_eff`-low-purity quadrant *is* the set of leak
   conduits. Down-weighting or cutting them cleans the gated cones — the original leak problem, now
   diagnosed positively rather than patched.
2. **Landmark / bridge selection for the caret precompute** (§5b–5c). Good bridges are the natural
   *designated bridges* / landmarks for between-node distance precompute; the detector picks them instead
   of hand-listing them.
3. **Domain-structure understanding & taxonomy QA.** Where does a domain genuinely branch into subfields?
   In-domain bridges are the structural joints; redundant hubs flag *mergeable* children; leak conduits
   flag *edges to cut*. A map of a category graph's real skeleton.
4. **Navigation / routing.** Bridges are the hub waypoints for moving between sub-domains — the high-value
   nodes to index for "how do I get from here to there."
5. **Feeding the directional-μ model.** The bridge structure marks the meaningful merge points (the MICA
   candidates) the directional model should care about — a structural prior for where order matters.
6. **Bridge-guided sampler seeding (a virtuous loop).** The training sampler (`gen_mu_pairs.py`) seeds
   random walks from a root and grows a *mesh* by adding walk endpoints — but raw endpoints **drift out of
   the domain** (the first real run wandered from `Physics` into `Tamils` / `Shinto_shrines`). A better
   frontier: take a couple of *short* walks from the root to nearby points, find the **good bridge** that
   joins their regions (their MICA, qualified by `n_eff` + purity), and seed the next walks from *that* — a
   structurally-important, in-domain node, close to the root when the walks are short (walk length is the
   locality knob). This replaces "add every endpoint" with "add good bridges," which (a) **keeps the mesh
   in-domain** — a leak conduit or redundant hub is rejected as a seed, the principled fix for the drift —
   and (b) covers the domain's real branch structure, yielding the *varied-anchor* pairs the directional
   model was starved for. It closes a loop: the μ map drives bridge detection, and the bridges seed the
   training data that sharpens the next μ map — a self-expanding, **drift-controlled** frontier. (Bound the
   loop: keep discovered seeds within a μ-coherent neighbourhood of the root so the frontier expands
   without wandering.)

## Where it sits

Pure **read-out / selection** on top of the shipped machinery — it introduces *no new math*. It reuses the
descendant sketches (`descendant_minhash_weighted`, `sketch_mu_mass`, `sketch_mu_overlap`), the node-gated
similarity (`gated_ic_node_filtered`, #3296), the membership cone (`descendant_mu_mass_gated`), and cone
purity. The detector is **assembly**, which is why it can be built and tested entirely in the Rust core
with no ML/HF dependency — the μ map it consumes is produced separately (the directional model, #3302, or
any dense μ map).
