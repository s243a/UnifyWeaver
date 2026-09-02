<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve P2 — store-backed resolution

P0.5 (term catalog) is unchanged and still the API. P2 adds **store-backed**
resolution: packages / depends / conflicts / reverse-deps live in D43
indexed fact stores; the machine-local environment stays a term. Bytes
read on a bound query are proportional to the query, not the catalog.

Debian epoch/tilde, `Provides:`/virtual packages, write paths, incremental
store updates, and a `pkg`-style CLI remain deferred.


Layered / frugal distros (Puppy, Woof-CE) run an **immutable curated base**
under a writable layer. Stock apt cannot reason about that boundary: it will
happily "upgrade" a held package that lives in a read-only SFS. This program
is a Prolog spec of a resolver that *can*. One set of relations; SWI-Prolog
is the oracle; the same spec is compiled through `wam_javascript`
(`emit_mode(mixed)`) and gated by the contract corpus plus a seeded
differential.

Debian epoch/tilde version semantics, `Provides:`/virtual packages, write
paths, incremental store updates, and a `pkg`-style CLI remain deferred.
The GP-LMDB / D43 indexed catalog path is P2 (this document).

## The model

The catalog is **data**, not the Prolog database. Every query takes a
`catalog/6` (P0) or `catalog/9` (P0.5 extras) term as its first argument.
There is no `assert`/`retract`. Bare `catalog/6` is still accepted: missing
layers / excluded / aliases are `[]`, and a bare `Name-Ver` hold is
`blanket`. Every P0 scenario is unchanged.

```
catalog(Packages, Depends, Conflicts, Base, Installed, Requested)
catalog(Packages, Depends, Conflicts, Base, Installed, Requested,
        Layers, Excluded, Aliases)
```

| Field | Shape |
| --- | --- |
| `Packages` | `[package(Name, v(M,I,P)), …]` |
| `Depends` | `[depends(Name, Ver, DepName, Constraint), …]` |
| `Conflicts` | `[conflicts(Name, Ver, OtherName), …]` (checked in both directions) |
| `Base` | `[Hold, …]` — see **Freeze reasons** |
| `Installed` | `[Name-Ver, …]` |
| `Requested` | `[Name, …]` — manual/root installs, used only by `removal_orphans` |
| `Layers` | `[layer(Name, [Hold, …]), …]` — named loaded layers (§2f, minimal) |
| `Excluded` | `[Name, …]` — candidate-generation blacklist only |
| `Aliases` | `[alias(Alias, Canonical), …]` — request edge only |

**Versions** are `v(Major, Minor, Patch)`, compared lexicographically.

**Constraints:** `any` · `eq(V)` · `gte(V)` · `lt(V)` · `range(Lo, Hi)`
(Lo inclusive, Hi exclusive). No Debian epoch/tilde in P0.5.

**Requests** are an atom `Name` (constraint `any`) or `req(Name, Constraint)`.
Alias rewriting happens here only; catalog names stay canonical.

## Freeze reasons (§2e)

A base hold is either the P0 pair `Name-Ver` (read as `blanket`) or
`base(Name-Ver, Reason)` with

```
Reason ∈ { layer_shadow, abi_anchor, modified, footprint, blanket }
```

| Reason | `safe_upgrade` verdict |
| --- | --- |
| `footprint` / `blanket` / `layer_shadow` | `safe(cost(Reason))` |
| `abi_anchor` | `coordinated(Set)` — minimal reverse-dep closure that must move with `Pkg` |
| `modified` | `unsafe(modified)` — never without re-applying modifications |
| (no such `NewVer`, or `Pkg` is not a base hold) | `no_candidate` |

`upgrade_set(Catalog, Pkg, NewVer, Set)` is that coordinated set: reverse
dependencies **among BASE packages** whose *current* version's constraint on a
moving package breaks at the new version. Each broken member is repaired with
the highest catalog version that satisfies the moving pins. A base dependent
whose constraint still holds at `NewVer` stays **out**. When a member has no
satisfying version the query fails; `upgrade_set_result/4` binds the same
`blocked(Name, needs(C), base_has(V))` shape as layered resolve.

`freeze_audit(Catalog, Audit)` walks **base-layer** holds only (other named
layers are not freeze-audited):

- non-blanket → `audit(Name, held(Reason))`
- blanket + a tight (`\== any`) reverse-dep from another base package →
  `audit(Name, suggest(abi_anchor))`
- blanket + no such reverse-dep → `audit(Name, over_frozen)` — the
  "TrixiePup freezes too much" diagnosis as a query

## Named layers (§2f, minimal slice)

`layer(Name, [Hold, …])` entries live in the 7th catalog argument (and are
also accepted inside `Base`). The layer named `base` is the frozen base;
every layer present in the catalog is treated as **loaded**. A dependency
already satisfied by any loaded layer is used in place (`from_base`) and is
not re-selected — the same rule P0 applied to `Base` alone. No per-file
modeling; membership is package-name + version per named layer.

## Excluded vs removal (Pkg bug we refuse to copy)

`excluded/1` names filter **candidate generation only**. An excluded package
is never selected by `resolve` / `resolve_layered`. Exclusion is **not**
consulted by `removal_orphans`: blacklisting an installed package must not
protect it from being orphaned. (Stock Pkg conflated the two.)

## Aliases

`alias(Alias, Canonical)` is applied at the **request edge**
(`resolve*`, `explain_blocked`, `layer_closure`, `removal_orphans`,
`safe_upgrade`, `upgrade_set`, `dependents*`). Catalog package names stay
canonical: requesting `urxvt` selects `rxvt`.

## Queries

All are exported from `resolver.pl`. `Cat` is the catalog term.

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

The same closure, **without touching loaded layers**. A dependency already
satisfied by a loaded-layer hold is used in place (never re-selected, never
upgraded). Only packages not held in a loaded layer appear in `Selection`.
When a held package is a version ceiling, the query **fails**.

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

The request plus its full **non-loaded-layer** dependency closure as an
ordered manifest (dependencies before dependents): the self-contained
SFS-style layer that installs without evolving the base / loaded layers.

```prolog
?- scenario_catalog(layer_tree, Cat), layer_closure(Cat, a, Layer).
Layer = [c-v(1,0,0), b-v(1,0,0), d-v(1,0,0), a-v(1,0,0)].
```

### 4. `removal_orphans(+Cat, +Pkg, -Orphans)`

Given `installed` and `requested` fields, the packages that were pulled in
as dependencies and would be orphaned by removing `Pkg` (no remaining
requested root needs them). **Base / loaded-layer packages are never
orphans.** Exclusion does not apply. The PPM-style lifecycle trim,
relationally.

### 5. `safe_upgrade(+Cat, +Pkg, +NewVer, -Verdict)` / `upgrade_set/4`

See **Freeze reasons**. `upgrade_set_result/4` is the always-binding form
used by the JS-WAM shim (`ok(Set)` | `blocked(...)` | `no_candidate`).

### 6. `freeze_audit(+Cat, -Audit)`

Per base-layer hold: held reason, or a derived `suggest(abi_anchor)` /
`over_frozen` diagnosis for blankets.

### 7. `dependents(+Cat, +Pkg, -Dependents)` / `dependents_installed/3`

`what-needs`: catalog packages (any version) that depend **directly** on
`Pkg`. `dependents_installed/3` keeps only those currently installed at
that version or held in a loaded layer at that version.

## Determinism policy

- `resolve/3` and `resolve_layered/3` commit to the **first** search
  solution (`!` at the API edge only). Candidate order: layered prefers a
  **loaded-layer-satisfied** version; otherwise **highest version**.
- Selections and layer manifests are sorted (by `Name-Ver` for
  `resolve*`; topological with sorted sibling names for `layer_closure`)
  so corpus/differential comparison is stable.
- `explain_blocked/3` enumerates; callers `findall`+`sort`.
- `removal_orphans/3`, `safe_upgrade/4`, `upgrade_set/4`, `freeze_audit/2`,
  `dependents/3`, `dependents_installed/3` each return one sorted /
  ground answer.

## P2: store-backed resolution

Real catalogs cannot be an in-memory term per query. The **big three**
(packages, depends, conflicts) plus a **precomputed reverse-deps** index
are D43 P/2 fact stores. The **environment** (base / layers / installed /
requested / excluded / aliases) is machine-local and tiny — it stays a
term, passed per query:

```
env(CatId, Base, Installed, Requested, Layers, Excluded, Aliases)
```

`resolver.pl`'s catalog-as-term API is untouched. `resolver_store.pl`
exposes the same 10 queries as `*_store` predicates. Lookups always bind
`CatId|Name` (seek, not scan).

### JSONL dump schema

One JSON object per fact (`kind` discriminates). `catalog` is the store
key prefix so many catalogs can share one index.

```jsonl
{"kind":"package","catalog":"linear","name":"a","ver":[1,0,0]}
{"kind":"depends","catalog":"linear","name":"a","ver":[1,0,0],"dep":"b","constraint":"any"}
{"kind":"depends","catalog":"blocked_base","name":"app","ver":[1,0,0],"dep":"lib","constraint":{"op":"gte","v":[2,0,0]}}
{"kind":"conflicts","catalog":"conflict_pair","name":"foo","ver":[1,0,0],"other":"bar"}
{"kind":"base","catalog":"upgradeable_base","name":"lib","ver":[1,0,0],"reason":"blanket"}
{"kind":"layer","catalog":"named_devx","layer":"devx","name":"gcc","ver":[1,0,0]}
{"kind":"installed","catalog":"removal_basic","name":"app","ver":[1,0,0]}
{"kind":"requested","catalog":"removal_basic","name":"app"}
{"kind":"excluded","catalog":"excluded_select","name":"bad"}
{"kind":"alias","catalog":"alias_rxvt","alias":"urxvt","canonical":"rxvt"}
```

Only `package` / `depends` / `conflicts` are indexed. Env rows stay
term-side. `depends` also emits a reverse-dep posting at build time.

### Store key layout (D43 P/2)

D43 stores are scalar P/2 (atom/int/float/string). Compounds are packed:

| Store prefix | a1 (index key) | a2 (packed atom) |
| --- | --- | --- |
| `pkg` | `CatId\|Name` | `Major.Minor.Patch` |
| `dep` | `CatId\|Name` | `Ver#Dep#Constraint` |
| `conflict` | `CatId\|Name` | `Ver#Other` |
| `revdep` | `CatId\|DepName` | `Name#Ver#Constraint` |

Constraint packing: `any` · `eq:1.0.0` · `gte:1.0.0` · `lt:2.0.0` ·
`range:1.0.0:2.0.0`. Builder: `store/rich_to_p2.mjs` then
`store/build_stores.sh` (`scripts/js_wam/uw_fact_index.js`). `lmdb(Dir)`
uses the same P/2 JSONL via `uw_fact_lmdb.js` (opt-in; no repo
`package.json` dependency).

### SWI-side data

SWI (the oracle) reads the **same P/2 JSONL files** the indexer consumes
(`load_p2_jsonl/1`), not the binary `.data`/`.idx`. SWI has no UWFI
reader; the JSONL is the canonical row encoding, so both engines see
identical keys and packed cells. WAM seeks `indexed(Prefix)` built from
those rows.

## Files

| path | what |
| --- | --- |
| `resolver.pl` | the spec (pure relations); API unchanged in P2 |
| `resolver_store.pl` | store adapter (same 10 queries; env term + P/2 facts) |
| `test_resolver.pl` | P0.5 contract corpus (append-only; unchanged) |
| `test_resolver_store.pl` | store vs term identity |
| `dump_corpus.pl` | SWI → JSONL for the term-catalog wamjs runner |
| `dump_store_data.pl` | rich JSONL + P/2 JSONL + store corpus cases |
| `wamjs/` | term-catalog JS WAM build |
| `wamjs_store/` | store-backed JS WAM build (D43 fact sources) |
| `store/` | dump schema helpers, indexer wrapper, 5k generator |
| `gen_catalogs.mjs` | mulberry32, 2400 term catalogs (seed `0xa5b6c7d8`) |
| `run_differential.sh` | term-catalog SWI vs wamjs, 0-divergence gate |
| `run_store_differential.sh` | store-backed 5k catalog, ≥500 cases, 0 divergences |
| `run_scale_demo.sh` | bytes-read proof + term-vs-store timings |

## Build / run

From the repo root (SWI-Prolog 9.x, Node v18+, `mkdir -p output/advanced`):

```bash
# P0.5 term-catalog contract
swipl -q -g test_resolver -t halt examples/pkg_resolver/test_resolver.pl
bash examples/pkg_resolver/wamjs/build.sh
bash examples/pkg_resolver/wamjs/run_corpus_wamjs.sh
bash examples/pkg_resolver/run_differential.sh

# P2 store adapter (identity + indexed corpus)
swipl -q -g test_resolver_store -t halt examples/pkg_resolver/test_resolver_store.pl
bash examples/pkg_resolver/wamjs_store/build.sh
bash examples/pkg_resolver/wamjs_store/run_corpus_wamjs.sh

# 5k catalog: bytes-read + timings + ≥500-case store differential
bash examples/pkg_resolver/run_scale_demo.sh
bash examples/pkg_resolver/run_store_differential.sh
```

## Deferred (not P2)

- Debian epoch / tilde / letter version semantics
- `Provides:` / virtual packages / alternatives
- `pkg`-style CLI on top of `examples/cli_args` (concurrent round owns `cli/`)
- write paths (install / remove / commit a layer)
- per-file / per-SFS modeling inside a named layer
- incremental store updates (rebuild the four indexes from a full dump)
