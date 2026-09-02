<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve P0.5: graduated freeze semantics + Pkg-mining adoptions

> Branch from `grok/pkg-resolver` as `grok/pkg-resolver-p05` and EXTEND the
> resolver. **HARD RULE THIS ROUND: do NOT edit the JS WAM runtime or emitter**
> (`templates/targets/javascript_wam/*`, `wam_javascript_lowered_emitter.pl`,
> `wam_javascript_target.pl`) — a concurrent hardening pass owns those files.
> If a resolver shape exposes a runtime bug, DOCUMENT it with a minimal
> reproduction in your handoff and work around it in the spec; do not fix it.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`).
Branch from **`grok/pkg-resolver`** as `grok/pkg-resolver-p05`. SWI-Prolog
(`swipl`); Node v18+. Everything this round lives in `examples/pkg_resolver/`.

First bring over the current design docs from the coordinator branch:
`git checkout origin/claude/peerhailer-exploratory-docs-aodas5 -- docs/proposals/PACKAGE_MANAGER_LOGIC_PROPOSAL.md docs/proposals/PKG_MINING_NOTES.md`
Read the proposal §2e (graduated freeze), §2f (named layers), and the mining
notes' top-10. That is this round's spec.

### The work (P0.5): holds get REASONS; the mining adoptions land

**1. Graduated freeze (§2e).** Extend the catalog term with reasons:
`Base` entries become `base(Name-Ver, Reason)`, Reason ∈
`layer_shadow | abi_anchor | modified | footprint | blanket`
(back-compat: accept bare `Name-Ver` as `blanket` so every existing P0 corpus
scenario and the P0 differential keep passing UNCHANGED — that is a gate).
New queries over the same clauses:
- `safe_upgrade(Catalog, Pkg, NewVer, Verdict)` — Verdict one of
  `safe(cost(Reason))` (footprint/blanket/layer_shadow), `coordinated(Set)`
  (abi_anchor: the minimal set that must move together — see below),
  `unsafe(modified)` (never without re-applying modifications),
  `no_candidate` (no such NewVer in the catalog).
- `upgrade_set(Catalog, Pkg, NewVer, Set)` — the minimal coordinated set for
  an abi_anchor upgrade: the reverse-dependency closure over BASE packages
  whose version constraints on Pkg break at NewVer, each with its own
  satisfying new version chosen (fail with an explanation term when a member
  has no satisfying version — reuse the `blocked/3` shape).
- `freeze_audit(Catalog, Audit)` — per base package: its reason, and for
  `blanket` entries whether a REAL reason is derivable from the catalog
  (tight-constraint reverse-deps ⇒ suggest abi_anchor; none ⇒
  `over_frozen(Pkg)` — the "TrixiePup freezes too much" diagnosis as a query).

**2. Named layers (§2f direction, minimal slice).** Generalize the base
partition: accept `layer(Name, [Pkg-Ver...])` entries in the catalog
alongside `base(...)` (base = the layer named `base`). `resolve_layered`,
`layer_closure`, and `explain_blocked` take the set of LOADED layers into
account (a dep satisfied by any loaded layer is not re-selected). Keep it
minimal: no per-file modeling, just package membership per named layer.

**3. Mining-notes adoptions (resolver-side items from the top-10):**
- `excluded/1` (blacklist) facts: filter CANDIDATE GENERATION only — an
  excluded package is never selected, but exclusion NEVER blocks removal or
  orphan computation (the Pkg bug: blacklisting an installed package
  accidentally protected it from removal — pin the non-conflation in a test).
- `dependents(Catalog, Pkg, Dependents)` — the `what-needs` query: which
  catalog packages (any version) depend directly on Pkg; and
  `dependents_installed/3` restricted to installed+base.
- `alias/2` name normalization facts, applied at the request edge only
  (requests resolve through aliases; catalog names stay canonical).

### Corpus + differential (the gates)
- Extend `test_resolver.pl`: every P0 scenario UNCHANGED and green, plus
  ~12–15 new scenarios: each safe_upgrade verdict incl. the coordinated set
  and its failure explanation; upgrade_set minimality (a base dep whose
  constraint tolerates NewVer stays OUT of the set); freeze_audit flagging an
  over-frozen blanket + suggesting abi_anchor for a tight one; a second
  loaded layer satisfying a dep; excluded never selected / excluded does not
  block removal; dependents + dependents_installed; alias at the request edge.
- Extend the JSONL corpus + `run_corpus_wamjs.sh` accordingly; extend
  `gen_catalogs.mjs` (new seed OK) to emit reasons/layers/excluded/aliases so
  the differential exercises the new queries. Rebuild via
  `examples/pkg_resolver/wamjs/build.sh` (allowed — it only RUNS the
  compiler). Differential: ≥2200 cases, **0 divergences** across ALL queries
  (old four + new ones).
- Update `examples/pkg_resolver/README.md` (reasons model, layers, the
  excluded-vs-removal rule, aliases; deferred list updated).

### Guardrails
- Edit ONLY under `examples/pkg_resolver/` (+ the two docs you checked out
  stay unmodified — they are reference). NO runtime/emitter/target/template
  edits (see the hard rule above); no edits to `examples/cli_args/`, shared
  files, harness, or tests outside pkg_resolver.
- Re-run and keep green: the three `tests/test_wam_javascript_*.pl` suites,
  `CONFORMANCE_TARGETS=javascript`, cli_args corpus 17/17 + differential
  5067/0/0 — unchanged from the lane tip, proving you touched nothing shared.

### Acceptance (must pass before handoff)
1. `swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl`
   → all P0 + new scenarios green (state the count).
2. `bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh` → all rows match SWI.
3. `bash examples/pkg_resolver/run_differential.sh` → ≥2200 cases,
   0 divergences (paste the summary + timings).
4. Pre-existing gates re-run green (list them).

### Handoff format
Return: the extended catalog shape (show one catalog with reasons + a second
layer + excluded + alias), the new-scenario list, corpus/differential numbers,
any runtime bug you had to work around (minimal repro — do NOT fix it), and
residuals (P2 GP-LMDB catalog backend, Provides/virtual, epoch/tilde, CLI
remain deferred).

## ↑↑↑ Copy to here
