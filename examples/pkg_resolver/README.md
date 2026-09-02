<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve P0 — a frozen-base package resolver

Layered / frugal distros (Puppy, Woof-CE) run an **immutable curated base**
under a writable layer. Stock apt cannot reason about that boundary: it will
happily "upgrade" a held package that lives in a read-only SFS. This program
is a Prolog spec of a resolver that *can*. One set of relations, four queries;
SWI-Prolog is the oracle; the same spec is compiled through `wam_javascript`
(`emit_mode(mixed)`) and gated by the contract corpus plus a seeded
differential.

Debian epoch/tilde version semantics, `Provides:`/virtual packages, a GP-LMDB
catalog backend, and a `pkg`-style CLI (via `examples/cli_args`) are deferred.

## The model

The catalog is **data**, not the Prolog database. Every query takes

```
catalog(Packages, Depends, Conflicts, Base, Installed, Requested)
```

as its first argument. There is no `assert`/`retract`.

| Field | Shape |
| --- | --- |
| `Packages` | `[package(Name, v(M,I,P)), …]` |
| `Depends` | `[depends(Name, Ver, DepName, Constraint), …]` |
| `Conflicts` | `[conflicts(Name, Ver, OtherName), …]` (checked in both directions) |
| `Base` / `Installed` | `[Name-Ver, …]` |
| `Requested` | `[Name, …]` — manual/root installs, used only by `removal_orphans` |

**Versions** are `v(Major, Minor, Patch)`, compared lexicographically.

**Constraints:** `any` · `eq(V)` · `gte(V)` · `lt(V)` · `range(Lo, Hi)`
(Lo inclusive, Hi exclusive). No Debian epoch/tilde in P0.

**Requests** are an atom `Name` (constraint `any`) or `req(Name, Constraint)`.

## The four queries

All four are exported from `resolver.pl`. `Cat` is the catalog term.

### 1. `resolve(+Cat, +Requests, -Selection)`

Classic dependency closure with **backtracking** over candidate versions:
every request satisfied, every dependency constraint met, no conflicts, one
version per package name. May "upgrade" a package that happens to sit in the
base — that is the naive solver stock apt would run.

```prolog
?- scenario_catalog(upgradeable_base, Cat), resolve(Cat, [app], Sel).
Sel = [app-v(1,0,0), lib-v(2,0,0)].   % lib 1.0 is in the base; classic picks 2.0
```

### 2. `resolve_layered(+Cat, +Requests, -Selection)`

The same closure, **without touching the base**. A dependency already
satisfied by `base(Name-Ver)` is used in place (never re-selected, never
upgraded). Only non-base packages appear in `Selection`. When a held package
is a version ceiling, the query **fails**.

```prolog
?- scenario_catalog(upgradeable_base, Cat), resolve_layered(Cat, [app], Sel).
Sel = [app-v(1,0,0)].                 % lib 1.0 stays in the base; not in Sel
```

`explain_blocked(+Cat, +Request, -Blocked)` (nondet) names the ceiling.
Same clauses as the layered walk; not a second engine. Terms look like

```
blocked(lib, needs(gte(v(2,0,0))), base_has(v(1,0,0)))
```

`explain_blocked_list/3` is `findall` + `sort` at the API edge.

### 3. `layer_closure(+Cat, +Request, -Layer)`

The request plus its full **non-base** dependency closure as an ordered
manifest (dependencies before dependents): the self-contained SFS-style
layer that installs without evolving the base.

```prolog
?- scenario_catalog(layer_tree, Cat), layer_closure(Cat, a, Layer).
Layer = [c-v(1,0,0), b-v(1,0,0), d-v(1,0,0), a-v(1,0,0)].
```

### 4. `removal_orphans(+Cat, +Pkg, -Orphans)`

Given `installed/1` and `requested/1` facts, the packages that were pulled
in as dependencies and would be orphaned by removing `Pkg` (no remaining
requested root needs them). **Base packages are never orphans.** The
PPM-style lifecycle trim, relationally.

## Determinism policy

- `resolve/3` and `resolve_layered/3` commit to the **first** search
  solution (`!` at the API edge only). Candidate order: layered prefers a
  **base-satisfied** version; otherwise **highest version**.
- Selections and layer manifests are sorted (by `Name-Ver` for
  `resolve*`; topological with sorted sibling names for `layer_closure`)
  so corpus/differential comparison is stable.
- `explain_blocked/3` enumerates; callers `findall`+`sort`.
- `removal_orphans/3` returns one sorted list.

## Files

| path | what |
| --- | --- |
| `resolver.pl` | the spec (pure relations) |
| `test_resolver.pl` | contract corpus (plunit + `corpus_case/4`) |
| `dump_corpus.pl` | SWI → JSONL for the node runner |
| `wamjs/` | P1 build, argparser-playbook style |
| `wamjs/build.sh` | `swipl` → `wam_javascript`, `emit_mode(mixed)` |
| `wamjs/resolver.mjs` | term ↔ JSON **only** — no resolver logic |
| `wamjs/run_corpus_wamjs.sh` | the same corpus through node vs SWI |
| `gen_catalogs.mjs` | mulberry32, 2200 catalogs |
| `run_differential.sh` | SWI vs wamjs, 0-divergence gate |

## Build / run

From the repo root (SWI-Prolog 9.x, Node v18+, `mkdir -p output/advanced`):

```bash
# contract corpus (SWI oracle)
swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl

# regenerate the JS WAM project, then drive the same corpus through node
bash examples/pkg_resolver/wamjs/build.sh
bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh

# ≥2000 seeded catalogs, four queries, 0 divergences
bash examples/pkg_resolver/run_differential.sh
```

## Deferred (not P0)

- Debian epoch / tilde / letter version semantics
- `Provides:` / virtual packages / alternatives
- GP-LMDB catalog backend (P2 — the `indexed` / `lmdb` fact-source path)
- `pkg`-style CLI on top of `examples/cli_args`
- write paths (install / remove / commit a layer)
