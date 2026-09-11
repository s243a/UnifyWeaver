<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Grok prompt — uw-resolve P0+P1: frozen-base package resolver, Prolog spec → wam_javascript

> Branch from `grok/wamjs-lmdb` (latest JS-WAM) as `grok/pkg-resolver`. NEW
> program, argparser playbook: a frozen SWI reference + contract corpus first,
> then the wam_javascript build passing the same corpus + a seeded differential
> vs SWI. This one BACKTRACKS — it exercises the WAM tier the way cli_args
> exercised the pattern tier. Read `docs/proposals/PACKAGE_MANAGER_LOGIC_PROPOSAL.md`
> (on branch `claude/peerhailer-exploratory-docs-aodas5`) first — especially
> §2d, the frozen-base scenario this program exists to solve.

---

## Copy from here ↓↓↓

You are contributing to **UnifyWeaver** (`github.com/s243a/unifyweaver`), a compiler
that lowers Prolog to a shared WAM bytecode and runs it on per-language WAM
interpreters. Branch from **`grok/wamjs-lmdb`** as `grok/pkg-resolver`. SWI-Prolog
(`swipl`); Node v18+. First bring over the proposal + the argparser playbook for
reference: `git checkout origin/claude/peerhailer-exploratory-docs-aodas5 -- docs/proposals/PACKAGE_MANAGER_LOGIC_PROPOSAL.md examples/cli_args/wamjs/README.md examples/cli_args/README.md`

### The program (uw-resolve P0): a resolver that understands a frozen base
Layered/frugal distros (Puppy/Woof-CE style) run an **immutable curated base**
(held packages in read-only layers) under a writable layer. Stock apt cannot
reason about that boundary. Your resolver's whole point is that it can. One set
of relations, four queries:

1. **`resolve(Requests, Selection)`** — classic dependency closure with
   BACKTRACKING over candidate versions: every request satisfied, every
   dependency constraint met, no conflicts, one version per package name.
2. **`resolve_layered(Requests, Selection)`** — closure satisfiable **without
   touching the base**: a dependency already satisfied by a `base/1` package
   uses it (never re-selected, never upgraded); only non-base packages are
   chosen. When impossible, fail — and provide
   **`explain_blocked(Request, Blocked)`** yielding terms like
   `blocked(DepName, needs(Constraint), base_has(Version))` naming exactly
   which held package is the version ceiling (query the same clauses for the
   near-miss; do not build a second engine).
3. **`layer_closure(Request, Layer)`** — the request + its full non-base
   dependency closure as an ordered manifest (dependencies before dependents):
   the self-contained SFS-style layer that installs WITHOUT evolving the base.
4. **`removal_orphans(Pkg, Orphans)`** — given `installed/1` facts, the
   packages that were pulled in as dependencies and would be orphaned by
   removing Pkg (no other installed package or request needs them; base
   packages are never orphans). The PPM-style lifecycle trim, relationally.

### The model (keep P0 minimal and clean)
- Catalog facts: `package(Name, Ver)`, `depends(Name, Ver, DepName, Constraint)`,
  `conflicts(Name, Ver, OtherName)`; environment facts: `base(Name-Ver)`,
  `installed(Name-Ver)`, `requested(Name)`.
- Versions: `v(Major, Minor, Patch)` compared lexicographically. Constraints:
  `any`, `eq(V)`, `gte(V)`, `lt(V)`, `range(Lo, Hi)` (Lo inclusive, Hi
  exclusive). NO Debian epoch/tilde semantics in P0 — note it as deferred.
- Determinism policy: `resolve*` return the FIRST solution under a documented
  candidate order (prefer base-satisfied, then highest version); `explain_blocked`
  and `removal_orphans` enumerate all answers (findall at the API edge).
  Selections/manifests are sorted deterministically for comparison.
- Style: pure relations, no assert/retract, no cuts you can replace with
  first-solution wrappers at the API edge; SWI is the oracle for every query.

### Deliverables — `examples/pkg_resolver/` (all new; SPDX headers)
1. **`resolver.pl`** — the spec, exporting the four queries (+ helpers).
2. **`test_resolver.pl`** — the CONTRACT CORPUS, written FIRST, ~15–20 plunit
   scenarios that pin the semantics (this is this program's "17 tests"). Must
   include at least: a resolution requiring genuine backtracking (a candidate
   version dead-ends on a conflict discovered deeper in the closure and an
   older version succeeds); a base-satisfied dependency NOT re-selected; a
   layered resolution that succeeds where naive `resolve` would have upgraded
   a base package; an unsatisfiable layered request with the exact
   `blocked(...)` explanation; a diamond dependency resolved to ONE version;
   a conflict exclusion; a multi-package layer manifest in dependency order;
   removal with orphans; removal where a would-be orphan is saved by another
   installed package; base packages never orphaned; an empty/trivial case.
3. **`wamjs/`** — the P1 build, argparser-playbook style: `build.sh` (swipl →
   wam_javascript, `mixed` emit mode; regenerable), the generated js/, a thin
   `resolver.mjs` shim (term↔JS conversion ONLY — catalogs/requests in as
   data, selections/explanations out as JSON; no resolver logic), and
   `run_corpus_wamjs.sh` — the SAME corpus scenarios driven through node,
   results compared to SWI including explanation terms.
4. **Differential**: `gen_catalogs.mjs` (seeded, mulberry32 like cli_args) —
   random catalogs (10–60 packages, 1–4 versions each, random deps/constraints/
   conflicts/base partitions/requests) — plus `run_differential.sh`: ≥2000
   generated cases through SWI and the wamjs build, **0 divergences** on all
   four queries (same selection, same explanation set, same orphan set; or
   both-fail).
5. **`README.md`** — the model, the four queries with examples (frozen-base
   framing), determinism policy, build/run instructions, deferred items
   (Debian version semantics, provides/virtual packages, GP-LMDB catalog
   backend = P2, `pkg`-style CLI via the cli_args machinery = later).

### Guardrails
- Everything lives under `examples/pkg_resolver/`. Runtime/emitter fixes are
  allowed if the resolver's shapes expose real wam_javascript bugs (the
  argparser precedent: each fix with a minimal probe in the appropriate
  `tests/test_wam_javascript_*.pl` suite, listed in the handoff) — but NO
  edits to `wam_target.pl`, `wam_text_parser.pl`, shared harness/registry/
  matrix/glue files, or anything under `examples/cli_args/`.
- Wrong code is worse than refusal; the differential is the semantics gate.
- Do NOT break: conformance (`CONFORMANCE_TARGETS=javascript`), the
  builtins/lowered/fact-sources suites, the cli_args corpus 17/17 +
  differential 0/0 (re-run them — you're branching from the lane tip).

### Acceptance (must pass before handoff)
1. `swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl`
   → all contract scenarios green.
2. `bash examples/pkg_resolver/wamjs/build.sh` regenerates; corpus-under-node
   matches SWI on every scenario.
3. `bash examples/pkg_resolver/run_differential.sh` → ≥2000 cases,
   **0 divergences** (paste the summary block + wall times for both legs).
4. All pre-existing suites/gates re-run green (list them).

### Handoff format
Return: the model + determinism policy as implemented, the corpus scenario
list, any wam_javascript fixes the program forced (symptom → cause → fix →
probe), the differential summary + timings, honest notes on where backtracking
depth/performance stood (profile with UW_PROFILE=1 if slow), and residuals.

## ↑↑↑ Copy to here
