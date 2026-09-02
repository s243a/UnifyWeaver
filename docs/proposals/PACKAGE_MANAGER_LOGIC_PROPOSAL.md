<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Proposal: package-management logic as a transpilable Prolog spec

**Status:** proposal (no code). Follows from the argparser maturity demonstration
(`examples/cli_args/`, A1–A4: one Prolog spec → four verified builds) and the
Babashka-as-shell-host discussion. Candidate for incubation here and extraction
to a submodule/repo once it grows past a demonstration.

---

## 1. The thesis

Dependency resolution is a logic-programming problem wearing an imperative
costume. "Find versions of these packages such that every dependency range is
satisfied, no two selections conflict, and pins are honored" is a relation —
version constraints, transitive closure, conflict exclusion, search with
backtracking. Every package manager reimplements this by hand (apt grew a
solver; aptitude and 0install embedded SAT solvers; Nix/Guix went functional
to escape the problem). Written as Prolog relations, the resolver **is** its
own specification: `resolve(Requests, Catalog, Selection)` either holds or
doesn't, and SWI-Prolog executes the spec directly.

What UnifyWeaver adds — and what nobody else's resolver has — is that the spec
is **transpilable and oracle-verifiable**. The argparser exercise proved the
pipeline end-to-end: one frozen Prolog reference, projected into TypeScript /
vanilla JS / annotated JS / ClojureScript / the JS WAM interpreter, each build
gated by the production corpus and a 5,067-line seeded differential at zero
divergences. The same discipline applies verbatim here: SWI is the executable
oracle; the differential harness is the correctness gate; the transpiled
artifact ships wherever the shell lives.

## 2. The three use cases, in increasing ambition

### 2a. Declarative Linux environment setup (the near one)

A machine spec as facts:

```prolog
want(git).  want(nodejs, '>=18').  want(swipl).
prefer(source(nodejs), nodesource).
```

The resolver produces a **plan** — an ordered list of actions
(`install(apt, git)`, `add_repo(nodesource)`, `install(apt, nodejs)`) — and a
thin host-side executor runs it. Crucially, we do **not** replace apt/dnf/npm:
we sit above them as the planning layer, the way `bb.edn` tasks or Ansible sit
above shells, but with the plan derived relationally instead of scripted. The
plan itself is inspectable data (print it, diff it, dry-run it) — the same
"debuggability first" ethos that motivated the annotated-JS target for
peerhailer in the original exploratory doc.

### 2b. The shell host (where Babashka earns its place)

The executor and CLI want to live in a shell-friendly host. Two projections of
the same spec:

- **node** — the proven path today (`wam_javascript`, mixed mode, 2–2.5× tier;
  or the pattern lane where the logic is deterministic).
- **Babashka** — a single static binary, ~10 ms startup, `babashka.fs` /
  `babashka.process` built in: the natural body for a `setup-env` command on a
  fresh machine that has no node yet. The CLJS lane already carries whole
  programs; bb specifically needs one bounded verification pass (the A3-ported
  lowering has only been exercised under nbb — its bb/JVM spellings exist but
  are unproven; see the CLJS port report).

Because both are projections of one spec, the plan a bb script computes on a
bare VM and the plan the node daemon computes are provably the same plans —
that is the transpilable-spec advantage, identical in shape to the argparser's
"one contract, every host" result.

### 2c. Remote package-management logic (the peerhailer connection)

This is the part that turns "another project" into a peerhailer feature. Once
the resolver is a host-neutral artifact, **where it runs becomes a deployment
choice**:

- A hail daemon can compute a plan **for a remote peer**: the peer ships its
  state (installed-package facts) as data; resolution happens wherever it's
  cheapest/trusted; the signed plan travels back over the routed channel
  peerhailer already provides. Logic moves, or facts move — both are just
  terms.
- Fleet setup becomes: one spec, N peers, each peer's executor applying its
  own derived plan — with peerhailer's identity/approval machinery (the
  `route approve` / sealed-sender work) providing exactly the trust layer a
  "remote machine will now run installs" story needs.
- The catalog problem maps onto work already landed/in flight: package
  metadata is a large fact base queried by bound name — precisely the
  **GP-LMDB indexed fact store** shape (seek-indexed lookups without loading
  the catalog; the dependency-free backend for bare machines, the LMDB backend
  where the native dep is acceptable).

## 2d. The concretizing case: frozen-base (Puppy-style) resolution

Post-proposal context (TrixiePup64 / Woof-CE discussion) sharpened P0 into a
specific unsolved problem. Frugal/layered distros assemble an **immutable
curated base** (SFS layers, `apt-mark hold` on its packages) under a writable
save layer. Stock apt then structurally cannot answer the questions users
actually have:

1. *Is X installable **without touching the base**?* (apt proposes upgrading
   held foundations, or refuses unhelpfully)
2. *If not — which held package is the ceiling, and what would have to move?*
   (the version-ceiling **explanation**; apt reports "held broken packages")
3. *Then give me X + its non-base dependency closure as a **self-contained
   layer** (SFS-style) instead* — install without evolving the base at all.
4. *If I remove X, which of its separately-installed deps become orphans?*
   (PPM's lifecycle-aware trim, done relationally)

All four are the SAME clauses queried differently — `base/1` facts partition
the package universe, resolution closes over non-base candidates, a failure
branch names the blocking held package, and the closure-minus-base IS the
layer manifest. This is the P0 contract-corpus scenario set, and it is why
this resolver is not redundant with apt: it reasons about a boundary apt is
built to ignore. (Historical echo: Puppy's abandoned `Pkg` project and PPM's
dependency heuristics were circling exactly this in bash.)

## 3. Why Prolog is the right spec language here (concretely)

| resolver need | Prolog form |
|---|---|
| dependency ranges | facts + arithmetic guards (`depends(a-'1.2', b, '>=2.0,<3')`) |
| transitive closure | the textbook relation; already a compiled pattern fleet-wide |
| conflict exclusion | negation/constraints (`\+ conflicts(Sel)`) |
| version search / backtracking | native WAM execution — the thing the hybrid tier exists for |
| "explain why not" | failure branches ARE the explanation; enumerate near-misses |
| pins, priorities, preferences | clause order + first-solution semantics, visible in source |

The "explain" row deserves emphasis: imperative solvers bolt explanation on;
a relational resolver gets "what would have to change for this to succeed" by
querying the spec differently. That is a user-facing feature, not an
implementation nicety.

## 4. Honest constraints (what must be true / built first)

1. **Backtracking lives in the WAM tier.** The pattern lane we hardened is
   det/semidet (the argparser never backtracks). A resolver searches. So the
   resolver core targets `wam_javascript` (proven, profiled, D41/D42 mixed
   mode) — while plan *emission* and the CLI can use the pattern lane.
2. **A bb-hosted resolver needs one of:** (a) the pattern-lane CLJS build, if
   the resolver is written in a det style with explicit candidate enumeration
   (possible but fights the grain), or (b) `wam_clojure` — which the D39 fleet
   census found has **no conformance arm at all**. Standing up that arm is the
   prerequisite, and it's on the fleet-gaps board already (CONF-CLOJURE).
3. **Version-ordering builtins.** Debian/semver comparison is fiddly and
   host-visible; it should be a small, oracle-tested builtin family (the
   `sub_string`/string-tag playbook: add to runtime + probe vs SWI), not
   ad-hoc string math in the spec.
4. **Executor security.** A plan that installs software, applied remotely, is
   an attack surface. The proposal's line: plans are data, signed and reviewed
   through peerhailer's existing approval machinery; executors are dumb
   (apply-only, no logic); the resolver never shells out.
5. **Scope discipline.** This is a *planner over existing package managers*,
   not a new package format, store, or Nix competitor. Guix proves the
   "package logic in a real language" idea works; our differentiator is the
   spec that transpiles and verifies, not a new ecosystem.

## 5. Placement: incubate here, extract when it grows

Same lifecycle as `examples/cli_args/`:

- **Incubate as `examples/pkg_resolver/`** in UnifyWeaver — it exercises the
  compiler (that's why it belongs here first), reuses the differential
  methodology, and stays honest via the SWI oracle.
- **Extract to a submodule/repo** when it acquires users or a release cadence
  — the examples directory's frozen-reference + generated-builds layout maps
  cleanly onto a standalone repo with UnifyWeaver as a build dependency.
- **peerhailer integration last**: peerhailer consumes the artifact (a
  resolver module + plan schema), never the Prolog toolchain.

## 6. Phased plan (each phase independently valuable, argparser-style)

| phase | deliverable | gate |
|---|---|---|
| **P0** | `resolver.pl`: minimal relation (packages, ranges, depends, conflicts, pins) + a toy catalog + plan emission, running under SWI | a plunit contract corpus (the "17 tests" of this project) written FIRST, from real-world resolution scenarios incl. at least one requiring backtracking and one unsatisfiable-with-explanation |
| **P1** | `wam_javascript` build via the A2 playbook (build.sh, shim, JSONL runner) | corpus green + a seeded differential vs SWI over generated request/catalog pairs, 0 divergences |
| **P2** | catalog as a GP-LMDB indexed fact store; scale test (a real distro's package index imported to facts) | bound-lookup bytes-read proof; resolution wall-time on the full catalog |
| **P3** | bb host (via CONF-CLOJURE + wam_clojure arm, or det-style CLJS) for the bare-machine `setup-env` story | same corpus + differential under bb |
| **P4** | peerhailer remote flow: facts-over-the-wire, plan-over-the-wire, signed via existing approval machinery | end-to-end demo: spec on host A, state from host B, plan applied on B |

P0+P1 alone would be a compelling second demonstration program for the
compiler — one that *backtracks*, which `cli_args` deliberately never did, and
would therefore exercise the WAM tier the way the argparser exercised the
pattern tier.

## 7. Decision requested

1. Green-light P0/P1 as the next demonstration program (Opus lane, argparser
   playbook)?
2. Priority of the bb path — i.e., does CONF-CLOJURE + the wam_clojure arm get
   pulled forward, or does bb wait for P3?
3. Name. Working name: **`uw-resolve`** (the artifact peerhailer would consume
   could keep its own name later).
