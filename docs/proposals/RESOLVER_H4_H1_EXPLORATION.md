<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve: H4 (layered nontermination) + H1 (layered candidate backtracking) — solution-space exploration

> External adversarial-review exploration (the reviewer that produced the
> post-D59 pruning-design findings in `RESOLVER_PRUNING_DESIGN.md` §7).
> Prose exploration, no code — the input to a design round. Verdicts and
> counterexamples are the reviewer's; recorded here as the design brief's
> spec. Rechecked at branch head `4ad359b5`: resolver, pruning design, and
> probes byte-identical to the reviewed snapshot.

## The two hazards (must be treated jointly — same layered loop, one re-baseline)

- **H4 (nontermination):** in layered mode a dependency satisfied by a held
  package takes the `from_base` path — the held package is EXPANDED
  (`collect_deps`, requests pushed to the FRONT of pending) but never enters
  `Acc`. Cyclic dependencies among held packages never terminate (nothing
  records the cycle was visited). Classic mode is safe (selected packages
  enter `Acc`; `selected_ver/3` short-circuits). **Reachable from real data**
  — Debian Essential sets contain dependency cycles.
- **H1 (probable P3 regression):** layered mode COMMITS to the highest
  candidate version (candidate call in an if-then-else condition), so a
  downstream dead end never retries a lower version; pre-P3 backtracked.

## The load-bearing correction (compatibility criterion)

**CE2 is graph-cyclic but OPERATIONALLY TERMINATING.** Its path is
`base0 → a → base0`; selecting `a` makes the second `base0` expansion
productive. So the criterion is NOT "preserve syntactically acyclic
catalogs" — it is "preserve every operationally-terminating result." Any
fix that only distinguishes cyclic from acyclic *graphs* fails CE2. The
missing signal is **selection progress**, not ancestry bookkeeping.

## Approaches evaluated

| # | Approach | Verdict |
|---|---|---|
| 1 | Opaque holds / insert holds into Acc / unconditional visited set | **reject** — CE2 fails; putting holds in Acc also corrupts conflict/provider/`selected_ver` checks irreversibly |
| 2 | Active ancestor chain keyed on concrete Name-Ver only | **reject** — CE2 fails (base0 still an ancestor when newly-selected `a` re-requests it); a second CE shows finite failure→success |
| 3 | **Active same-state cycle closure** | **PREFERRED (reference semantics)** |
| 4 | Visited set reset after every catalog selection | **serious alternative** (needs broader proof) |
| 5 | Same-state detection that REJECTS the branch | reject — wrong default; real installed cycles must resolve to `[]`, not fail |
| 6 | Static SCC condensation / batch fixed-point | unsafe as an independent shortcut — providers/alternatives make the real expansion graph state-dependent; metadata may assist dynamic checks but must not authorize batching/reordering |
| 7 | Full `(Pending,Acc)` memoization / tabling | **termination claim falsified** — `a → [a,missing]` grows the state unboundedly, no whole state repeats |
| 8 | Fuel / depth limits / iterative deepening | instrumentation, not a fix — fixed limits reject deep finite catalogs (success→failure); iterative deepening changes first solutions |

### Approach 3 — active same-state cycle closure (the reference)
Order of operations matters: **first** run the existing selected/provided
checks and **validate the incoming request**; **then** suppress a
`from_base` expansion only when the actual held `Pkg-Ver` is already active
with the **same `Acc`**. Preserve all remaining pending obligations.

- Represent "unchanged `Acc`" by a **branch-local selection generation**
  advanced after every catalog insertion — including a second version of an
  existing name (the H2 shape). Key the active-expansion mark on
  `(Pkg, Ver, Generation)`. Backtracking must restore marks AND generation
  together.
- **CE2 passes**: selecting `a` advances the generation, so the second
  `base0` expansion is not suppressed. The H2 finite-failure CE keeps its
  terminating failure.
- **Termination** (abstract, not a runtime-resource bound): each forward
  branch adds ≤ M distinct catalog pairs (real picks can't repeat a selected
  requested name; an identical provider pair already in `Acc` is caught by
  `already_provided` first); at unchanged selection state, active held
  expansion depth is bounded by H (reachable held pairs). Finite dependency
  lists + alternatives bound branching.
- **Cyclic first-solution semantics**: retain depth-first request order,
  omit only the repeated expansion. For held `a → [b,x]`, `b → [a,y]`: close
  the repeated `a`, then process `y` before `x`.
- **Cost**: full-`Acc`-snapshot compare would be O(H·|Acc|); generation-
  tagged membership is O(H). Long held chains still need explicit perf
  testing; a persistent membership tree is the fallback.
- **Unsound if closure precedes request validation**: held
  `a-1 → b-1 → a=2` must FAIL, not close.

### Approach 4 — visited set reset per catalog selection (alternative)
Remember held pairs expanded since the latest `Acc` insertion; clear the set
whenever ANY catalog pair is added (a second version of an existing name
must also clear it — counting only newly-selected *names* is insufficient);
validate each incoming request before suppressing. O(H) membership, O(1)
reset, no ancestry frames. Extra proof obligation over approach 3: a
*completed* repetition at unchanged `Acc` previously succeeded without adding
a selection, so replaying those pure ground tests is claimed to be a no-op —
this needs proof against alternatives, pending obligations, and backtracking.

### Evidence for 3 and 4
Agreed across **5,000 seeded model cases** under BOTH current-H1 and
restored-H1-backtracking; preserved all **4,631** baseline cases finishing
within the inference budget; the other **369** exceeded budget (budget-
exhausted, not proven divergences) and both models completed them.

## H1 repair — two materially different choices
- **Version-only restoration (recommended for a narrow repair):** preserve
  held-name and held-provider commitment; enumerate descending real versions
  in the branch body; use catalog providers only when no real candidate
  exists.
- **Classic-style fallback:** after exhausting real versions downstream, try
  catalog providers too.
If real `x` depends on `missing` and `p` provides `x`, version-only still
fails; classic-fallback selects `p`. Recommend version-only unless broader
fallback is explicitly chosen. Do NOT accidentally uncommit held ceilings or
change loaded-provider priority. H1 causes failure→success and
success→different-success (a lower version rescues an earlier alternative).
**H4 is required for H1**: a cyclic highest candidate must not prevent
reaching lower candidates.

## Sequencing
One coordinated H1/H4 round with separately reviewable stages: **H4 first,
then H1**, then one approved baseline update. If releases must be separate,
ship **H4 first** — H1 cannot backtrack past a nonterminating highest
candidate.

## explain_blocked (walks the same clauses; already has path-local `Seen`)
Two decisions required:
1. Held `a-1 → b-1 → a=2` fails resolution but currently yields an EMPTY
   explanation (the repeated name is suppressed before its new constraint is
   checked).
2. After H1, resolution can select `x-1` while explanations still traverse
   the highest `x-2` and report its ceiling.
Decide: do explanations describe the *preferred candidate's* blockage, or the
*absence of any solution*? (The existing traversal does not establish full
unsatisfiability.) Preserve unaffected enumeration order/multiplicity; never
treat a satisfied cycle as inherently blocked.

## Corpus the round must pin
- CE2's exact selection, and revised CE5/CE6 outcomes.
- Successful AND impossible cycle exits; incompatible closing-edge
  constraints.
- H2 progress incl. a second version of the same name.
- Virtual providers; distinct held versions under one name; sibling scope;
  branch rollback.
- A satisfiable cyclic highest version vs. a cyclic candidate requiring
  lower-version retry.
- Earlier alternatives rescued by lower versions.
- Explanation behavior and resource exhaustion as SEPARATE observables.
- **CE2 assertion split for the combined round**: the original catalog must
  retain `[a-1,b-2,d-1,x-2]`; the modified catalog's old NEGATIVE assertion
  becomes invalid — restored H1 finds `[a-1,b-1,d-2,x-2]`.

## Validation protocol
Run the **four-way matrix** — neither fix / H4 only / H1 only / both — on all
five runtimes (SWI, wamjs, Go, Rust, ClojureScript). Keep resource
exhaustion distinct from unsatisfiability throughout. **Reconsider G4 only
after** these cycle and choice semantics are established.

## Ranked recommendation
Approach 3 as reference semantics; approach 4 retained as a serious
alternative pending its broader proof; SCC metadata only if profiling
justifies it. One coordinated H4-then-H1 round.
