# Algorithm Optimization Manifest — Specification

## Naming note

The user-level directives are `decl_algorithm/2` and
`decl_algorithm_optimization/2`. The `decl_` prefix signals that
these are *declarative directives* (facts that describe metadata
about the workload), not callable predicates. The prefix also
avoids any collision with library predicates that might define a
plain `algorithm/2` — defensive even if no current standard
library uses that name.

## Terminology note

In this document and the scan-strategy docs that reference it,
**"algorithm" means the *logical specification* of the graph
problem** — the declarative *what*, not the operational *how*.
This is closer to the Datalog / SQL sense (a query is an
algorithm; its execution plan is a separate artefact) than the
algorithms-textbook sense (an algorithm is a step-by-step
procedure).

An algorithm in our sense may *hint* at how it's computed (e.g. a
recursive kernel suggests BFS-style expansion) but it doesn't
fully specify the operational behaviour. The **optimization
specification** completes the definition by saying which cache
mode, scan strategy, cost function, etc. to use. Together,
algorithm + optimization manifest = executable artefact.

This split matters because the two halves have different owners
(workload author vs perf engineer), different lifecycles (stable
vs measurement-driven), and different concerns (correctness vs
throughput). Mechanically the split is small; conceptually it's
the central point of this doc.

## What this is

A workload-author-level mechanism for separating two concerns:

1. **The algorithm** (logical specification): *what* is being
   computed (a kernel, its inputs, its output shape).
2. **The optimization manifest** (operational specification):
   *how* the codegen should compile the algorithm (cost-model
   knobs, cache mode, scan strategy, demand-filter spec, etc.).

The split exists because the two have different owners and lifecycles:

- Algorithm: stable, owned by whoever wrote the workload, rarely
  changes.
- Optimizations: tuning artefacts, owned by whoever cares about
  perf, change with measurements / hardware / data scale.

Without the split, every bench harness and every call site
re-declares the same option bundle. With it, an algorithm is
declared once with its current best-known optimization profile;
any caller that names the algorithm gets the optimizations for
free.

This document specifies the mechanism. It's deliberately small —
the abstraction is a thin layer over the existing option-list
plumbing the resolvers already consume.

## Connection to existing patterns

This generalises three patterns already present in the codebase:

| existing | how the manifest subsumes it |
|---|---|
| `user:demand_filter_spec(Strategy, Opts)` | an entry in the manifest's option list |
| `user:max_depth/1`, `user:dimension_n/1` | algorithm-level metadata in `decl_algorithm/2` |
| `statistics:declare_cache_hints/1` | `workload_locality/1` entry in the manifest |

The manifest is **additive**, not replacing existing mechanisms.
The codegen still reads `user:demand_filter_spec/2`, etc., for
backwards compatibility. New workloads use the manifest; old
workloads keep working unchanged.

## API

Two user-level predicates. Both declared in workload files via
`:- ...` directives (the standard Prolog idiom for facts about
the workload).

### `decl_algorithm(+Name, +AlgorithmOpts) is det.`

Declares an algorithm. `Name` is an atom identifying the
algorithm (e.g. `effective_distance`, `transitive_closure`,
`shortest_path`). `AlgorithmOpts` is an option list describing
the algorithm's structural metadata:

- `kernel(P/N)` — the recursive kernel predicate.
- `seeds(P/N)` — the seed predicate (input).
- `roots(P/N)` — the root predicate (target).
- `max_depth(D)` — bound on recursion depth.
- Other algorithm-specific keys (e.g. `dimension_n/1` for
  effective_distance).

Example:

```prolog
:- decl_algorithm(effective_distance, [
       kernel(category_ancestor/4),
       seeds(article/1),
       roots(root_category/1),
       max_depth(10),
       dimension_n(5)
   ]).
```

At most one `decl_algorithm/2` declaration per file. Multiple
declarations with the same name in different files are an error;
the codegen rejects compilation rather than silently picking one.

### `decl_algorithm_optimization(+Name, +OptList) is det.`

Declares optimizations for an algorithm. `Name` matches an
`decl_algorithm/2` declaration. `OptList` contains any options that
downstream resolvers consume.

Example:

```prolog
:- decl_algorithm_optimization(effective_distance, [
       cache_strategy(auto),
       working_set_fraction(0.001),
       expected_query_count(10),
       lmdb_cache_mode(auto),
       workload_locality(unknown),
       scan_strategy(auto),
       tree_cost_function(flux, [iterations(1), parent_decay(0.5), child_decay(0.3)]),
       tree_retention(snapshot_only),
       %% warm_budget_nodes_fraction(F) is a manifest-friendly alias
       %% that scan_strategy's resolver converts to an absolute
       %% warm_budget_nodes(N) before codegen. See the option list in
       %% SCAN_STRATEGY_SPECIFICATION.md for both forms.
       warm_budget_nodes_fraction(0.1),
       demand_filter_spec(hop_limit, [max_hops(10)])
   ]).
```

Multiple `decl_algorithm_optimization/2` facts for the same algorithm
are allowed — their option lists are concatenated in declaration
order using `append/2` (shallow concat — see the `load_algorithm_manifest/2`
sketch below for the trap with `flatten/2`). The manifest itself
does no de-duplication at concat time. When the merged list is
later read via SWI-Prolog's `option/3` accessor, the first
occurrence of any key wins — that's `option/3`'s documented
behaviour for repeated keys. This lets workload authors split
optimizations by concern:

```prolog
:- decl_algorithm_optimization(effective_distance, [
       cache_strategy(auto), working_set_fraction(0.001), ...
   ]).
:- decl_algorithm_optimization(effective_distance, [
       scan_strategy(auto), tree_cost_function(flux, [...]), ...
   ]).
```

If a key appears in multiple facts, the first occurrence wins
(matches `option/3` semantics). To override, place the overriding
fact earlier or pass the option directly to the codegen entry
point.

## Merge semantics

When the codegen enters `write_wam_haskell_project/3` (or
`compile_wam_runtime_to_haskell/3`), the option pipeline is:

```
1. Caller-provided options: Options0
2. Manifest options:        ManifestOpts
                              = concat of all decl_algorithm_optimization(Name, _)
                                facts for the declared algorithm
3. Merged options:          Options' = merge(Options0, ManifestOpts)
   - For each key in either:
       if Options0 has key  → take from Options0
       else                 → take from ManifestOpts
4. Resolvers run on Options' in their existing order.
```

**Caller wins on conflict.** This is the override path: a bench
that wants to experiment with `tree_retention(live)` passes that
option directly, and it beats the manifest's `snapshot_only`.

**Manifest is best-known-defaults, not policy.** The manifest is
a recommendation derived from measurement and cost-model logic.
Callers should feel free to override when they have a reason.

## Composition with resolvers

The manifest sits *before* the existing resolvers in the option
pipeline. By the time `resolve_auto_cache_strategy/2`,
`resolve_auto_lmdb_cache_mode/2`, etc., run, they see a merged
option list with manifest defaults filled in.

```
Options0 (caller)
  → merge with ManifestOpts
  → resolve_auto_use_lmdb
  → resolve_auto_cache_strategy
  → resolve_auto_lmdb_cache_mode
  → resolve_auto_scan_strategy   (new in P4 of the scan-strategy plan)
  → codegen
```

The resolvers themselves don't change. The manifest just provides
their inputs.

## Where the manifest is read

A small Prolog helper, conceptually:

```prolog
%% load_algorithm_manifest(+Options0, -OptionsWithManifest) is det.
%
%  Reads user:decl_algorithm/2 and user:decl_algorithm_optimization/2 (if
%  declared) and merges the optimization options into the codegen's
%  option list. Caller-provided options always win on conflict.
load_algorithm_manifest(Options0, OptionsWithManifest) :-
    (   user:decl_algorithm(Name, _AlgOpts)
    ->  findall(Opts, user:decl_algorithm_optimization(Name, Opts), AllOpts),
        %% IMPORTANT: shallow concat, not deep flatten. flatten/2 would
        %% destructure option terms whose values are themselves lists —
        %% e.g. `tree_cost_function(flux, [iterations(1), parent_decay(0.5)])`
        %% would have its inner parameter list pulled into the outer
        %% option list and corrupt the structure.
        append(AllOpts, ManifestOpts),
        merge_caller_and_manifest_options(Options0, ManifestOpts,
                                          OptionsWithManifest)
    ;   OptionsWithManifest = Options0
    ).
```

The merge helper is `merge_caller_and_manifest_options/3` — same
name used in `SCAN_STRATEGY_IMPLEMENTATION_PLAN.md` P0. We
deliberately avoid the bare name `merge_options/3` because
SWI-Prolog's `library(option)` exports a different
`merge_options/3` with reversed argument-order conventions;
sharing the name would invite confusion at use sites.

This helper runs once at the head of the codegen pipeline. There
are two public entry points:

- `write_wam_haskell_project/3` — the end-to-end path used by
  bench harnesses and users.
- `compile_wam_runtime_to_haskell/3` — the inner path used by
  codegen tests that don't want to write a project to disk.

`write_wam_haskell_project/3` calls `compile_wam_runtime_to_haskell/3`
internally, so if the manifest load lives only at the inner entry
point, both paths benefit. The current implementation puts it
explicitly at both for symmetry with the existing cost-model
resolver wiring (`resolve_auto_use_lmdb/2` already lives at both
sites); P0's deliverables include making `load_algorithm_manifest/2`
idempotent so the duplicate call in the outer path is harmless.

## Edge cases

### No algorithm declared

The codegen continues to work without a manifest — `Options0`
flows straight through. Existing workloads that don't use
manifests are unaffected.

### Algorithm declared but no optimization manifest

Same as caller-provided options only. The `decl_algorithm/2` declaration
itself is informational (the codegen reads its `kernel/1`,
`seeds/1`, etc. — same way it currently reads `user:max_depth/1`),
but no optimization options are merged.

### Multiple algorithms declared in one workload

Currently an error. A workload that genuinely computes two
algorithms in one binary is rare; the design defers the
multi-algorithm case to a future extension (probably named
manifests selected at call time). For now: one algorithm per
workload.

### Optimization for an undeclared algorithm

`algorithm_optimization(foo, _)` without a matching `algorithm(foo,
_)` is a warning. The optimization options are ignored; the
codegen logs that the manifest was orphaned.

### Manifest option that no resolver understands

Unknown keys are passed through. If a future resolver adds support,
existing manifests pick it up without changes. If a resolver
removes support for a key, the unknown entry is silently ignored
(or warned, depending on strict-mode setting).

### Conflicting `decl_algorithm_optimization/2` facts

Multiple facts concat with first-match semantics. If two facts
both set `cache_strategy/1` to different values, the first one
encountered wins. Workload authors who need explicit ordering
should consolidate into a single fact.

## Example: full workload file

```prolog
:- module(workload_effective_distance, []).

%% Algorithm declaration
:- decl_algorithm(effective_distance, [
       kernel(category_ancestor/4),
       seeds(article/1),
       roots(root_category/1),
       max_depth(10),
       dimension_n(5)
   ]).

%% Optimization manifest, split by concern
:- decl_algorithm_optimization(effective_distance, [
       %% Cache cost-model
       cache_strategy(auto),
       working_set_fraction(0.001),
       expected_query_count(10),
       mem_available_bytes(8000000000)  % 8 GB; overrides /proc/meminfo
                                        % (underscore-grouped literals
                                        %  also work in SWI-Prolog 8.x+,
                                        %  but the plain form is portable
                                        %  to older Prologs)
   ]).
:- decl_algorithm_optimization(effective_distance, [
       %% Cache tier
       lmdb_cache_mode(auto),
       workload_locality(unknown),
       cache_tier_floor_bytes(8000000)  % 8 MB
   ]).
:- decl_algorithm_optimization(effective_distance, [
       %% Scan strategy
       scan_strategy(auto),
       tree_cost_function(flux, [
           iterations(1),
           parent_decay(0.5),
           child_decay(0.3),
           flux_merge(sum)
       ]),
       tree_retention(snapshot_only),
       warm_budget_nodes_fraction(0.1)
   ]).
:- decl_algorithm_optimization(effective_distance, [
       %% Demand filter
       demand_filter_spec(hop_limit, [max_hops(10)])
   ]).

%% Kernel definition (unchanged)
%% Base case: a node reaches itself at hop 0.
category_ancestor(Root, Root, 0, _).
%% Recursive case: step to a parent, recurse, account for the hop.
category_ancestor(Cat, Root, Hops, Visited) :-
    category_parent(Cat, Mid),
    \+ member(Mid, Visited),
    category_ancestor(Mid, Root, H, [Mid|Visited]),
    Hops is H + 1.
```

A bench harness or end-user code that wants to run this workload
now just names the algorithm:

```prolog
:- use_module(workload_effective_distance).
:- write_wam_haskell_project(
       [category_ancestor/4, ...],
       [],  % no caller-level options — manifest provides everything
       'output/effective_distance_bench'
   ).
```

To experiment with a different optimization:

```prolog
:- write_wam_haskell_project(
       [category_ancestor/4, ...],
       [tree_retention(live)],  % overrides manifest's snapshot_only
       'output/effective_distance_bench_live'
   ).
```

## What this isn't

- **Not a DSL for algorithms.** `decl_algorithm/2` is a metadata
  declaration, not a language for *defining* algorithms. The
  kernel still lives in normal Prolog clauses.
- **Not a runtime mechanism.** Manifests are read at codegen time
  only. They don't affect runtime behaviour directly; their effect
  is mediated through the options they feed to the resolvers.
- **Not a replacement for explicit options.** Callers retain full
  control via the caller-wins merge rule.

## See also

- `docs/design/CACHE_COST_MODEL_PHILOSOPHY.md` — the cost-model
  resolvers that consume manifest entries.
- `docs/design/SCAN_STRATEGY_SPECIFICATION.md` — the scan-strategy
  options that go in a manifest.
- `docs/design/SCAN_STRATEGY_IMPLEMENTATION_PLAN.md` — Phase 0
  introduces the manifest abstraction; later phases assume it.
