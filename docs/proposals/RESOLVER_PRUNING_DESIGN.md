<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# uw-resolve: constraint insertion and branch trimming in `resolver.pl`

**Status:** design, not code. The only repository change in this round is this
document. Every number below was measured in a scratch copy of
`examples/pkg_resolver/resolver.pl` (commit `dadbed63`, P3); the scratch
prototypes are described precisely enough to be re-created, and the
implementation plan (§5) tells the implementing round what to build and how
to gate it.

**One-paragraph verdict.** The 5k-catalog `resolve_layered` is not a search
problem. Since P3 the layered walk commits to the highest candidate version
(§1.3, hazard H1), so the whole query runs with **zero search backtracks**; the
285k "backtracks" in the D54 census are the if-then-else choice points inside
two list walkers. What costs 7.56M WAM instructions is **24 linear scans of the
catalog lists** (14 over 15,001 `depends` rows, 10 over 7,520 `package` rows).
No guard that trims branches can touch that, because there are no branches
to trim. The one edit with leverage is a **per-call index over the catalog
lists built in pure Prolog at the API edge** (guard G1): on the wamjs leg it
takes the 5k query from "cannot finish under the runtime's 2,000,000-step
cap" to **675,217 instructions and a correct answer**, and cuts a
1,000-package query 8.8× (1,590,479 → 180,208 instructions, wall 3.4×).
It is provably answer-preserving (§2, G1). It is **not** free: on the
10-package catalogs of the 2,600-case differential it costs 11–14 % wall on
wamjs, and on SWI itself it makes the 5k query slower (9.0 → 13.1 ms) because
SWI's native scans are cheaper than building a tree. On the Rust leg the
index build as prototyped is **quadratic** (117 s at 5k versus 3.3 s
baseline) and must not ship there until the runtime cause is found (§5,
step 1c). Three further guards (G2–G4) are sound and cheap but small; three
candidate families are rejected with executable counterexamples (§3). Six
pre-existing semantic hazards found on the way, one of them a probable P3
regression, are recorded for the owner and are **preserved**, not fixed, by
this design (§3.2).

---

## 1. Cost model recap

### 1.1 The D54 census, re-derived on SWI

D54 counted the 5k `resolve_layered([p30])` (seed `0xc0ffee01`, base
`[p0-v(0,0,0)]`) at 7,559,597 WAM instructions on the ClojureScript leg,
with `matching_deps/4` entered 210,014 times and `matching_versions/4`
75,140 times. SWI's profiler on the same catalog and query gives the call
structure behind those two numbers:

| predicate | calls | why |
| --- | ---: | --- |
| `collect_deps/4` → `matching_deps/4` | 14 | 10 `from_catalog` picks + **4 `from_base` expansions of `p0`** (every dependency edge onto a held package re-walks that package's depends rows) |
| `candidates_high_first/4` → `matching_versions/4` | 10 | one full package-list scan per selected package |
| `pick_need/8` | 14 | |
| `no_acc_conflicts/4` / `conflicts_in/4` | 10 / 90 | |Acc| conflict-list scans per pick, over an empty list here |
| `already_provided/4` / `provides_sat/5` | 14 / 79 | |Acc| provides-list scans per request, over an empty list here |

210,014 = 14 × 15,001 and 75,140 = 10 × 7,514: every scan is a **full**
scan. 285,154 of the 302,910 SWI inferences (94 %) are the two list walkers.
On wamjs (`UW_PROFILE=json`, this box) each `matching_deps/4` iteration costs
28 interpreted instructions and each `matching_versions/4` iteration 26, so
the full query is ≈ 5.9M + 2.0M ≈ 7.9M instructions — the D54 figure.

### 1.2 Where the extra four scans come from

`resolve_pending/5` treats a `from_base` pick as "use in place": the held
package is **not** pushed onto `Acc`, so the next dependency edge onto it
fails `selected_ver/3`, goes through `pick_need/8` again, and calls
`collect_deps/4` again. `p0` is the most-depended-on package in the seeded
DAG (four edges inside the closure), so its (empty) depends list is scanned
four times. This is a spec-level shape, not a runtime defect, and it is
**order-sensitive** (§3.1, R2): the re-expansion pushes `p0`'s dependency
requests back onto the front of the pending list each time.

### 1.3 The search is deterministic in layered mode (hazard H1)

P0.5's `pick/7` enumerated candidate versions on backtracking in both modes.
P3's `pick_need/8` keeps that for `classic` (two clauses) but in `layered`
the candidate call sits in an if-then-else condition:

```prolog
pick_need(layered, Cat, Name, C, _Acc, Pkg, Ver, Origin) :-
    (   base_ver(Cat, Name, BV)     -> ...
    ;   layer_provider(...)          -> Origin = from_base
    ;   candidates_high_first(Cat, Name, C, Ver)   % <- committed by ->
    ->  Pkg = Name, Origin = from_catalog
    ;   provider_candidate(...), Origin = from_catalog
    ).
```

The `->` commits to the **highest** satisfying version; a downstream dead end
never retries a lower one. Probe CE6 (§6.1) shows it: on a catalog where
`b-2.0` needs `d = 1.0` and `x-2.0` needs `d = 2.0`, `resolve([b, x])`
answers `[b-2.0, d-1.0, x-1.0]` and `resolve_layered([b, x])` **fails**; the
pre-P3 file (`git show f802cf2:examples/pkg_resolver/resolver.pl`) answers
`[b-2.0, d-1.0, x-1.0]` in both modes. Whether this is intended is the
owner's call (it reads like a regression; the README still describes
layered as "the same closure" as classic). For this design it is an
**invariant**: the differential gate is SWI-oracled, so all five legs
reproduce the commit, and any guard must too.

Consequences for pruning: in layered mode the only backtracking surfaces
are alternatives groups (`member/2` over `Alts`) and providers of a virtual
with no real candidate (`member/2` over the provides list). A "doomed
subtree" in layered mode is therefore at most one linear pass, never an
exponential fan-out, which bounds what G4 can save.

### 1.4 The differential (B2) profile: where instructions go on small catalogs

The 2,600-case differential is the semantics gate but also the only
whole-program profile that exercises every query. wamjs, baseline,
`UW_PROFILE=json`, 11,903,199 instructions in 14.7 s:

| predicate | instructions | share | called from |
| --- | ---: | ---: | --- |
| `matching_deps/4` | 3,989,737 | 33.5 % | `collect_deps/4` (resolve, explain, upgrade_set) |
| `matching_versions/4` | 2,608,484 | 21.9 % | `candidates_high_first/4` |
| `lookup_held/3` | 950,464 | 8.0 % | `base_ver/3` — `append/3` + walk per call |
| `dep_breaks/5` | 909,083 | 7.6 % | `upgrade_set` only |
| `provides_sat/5` | 409,927 | 3.4 % | `already_provided/4`, `layer_provider/5` |
| `conflicts_in/4` | 320,729 | 2.7 % | `no_acc_conflicts/4` (|Acc| scans per pick) |
| `direct_on/4` + `dep_mentions/2` | 539,365 | 4.5 % | `dependents/3` (one scan; already optimal) |

### 1.5 Measured baseline and prototype numbers (this box)

Three scratch prototypes, all passing the 50-scenario corpus and
**byte-identical** to the baseline SWI output on all 2,600 differential
cases, and 0 divergences on the wamjs leg:

- **P1** = G1a + G1b with `sort/2` over position-tagged keys.
- **P2** = P1 with `keysort/2` instead (no position tags).
- **P3** = P2 + G2 + G4 + G1c.

| leg / workload | baseline | P1 | P2 | P3 |
| --- | ---: | ---: | ---: | ---: |
| SWI, 5k `resolve_layered(p30)`, 50 reps | 286,804 inf / **8.98 ms** | 142,472 / 15.94 ms | 142,472 / **13.09 ms** | — |
| wamjs, 250 pkgs (363 rows / 811 deps) | 421,879 instr / 462 ms | 54,383 / 191 ms | 44,773 / 178 ms | — |
| wamjs, 500 pkgs (736 / 1,549) | 808,155 / 768 ms | 96,189 / 271 ms | 77,699 / 263 ms | — |
| wamjs, 1,000 pkgs (1,503 / 3,036) | 1,590,479 / 1,476 ms | 180,208 / 436 ms | **143,848 / 395 ms** | 144,434 |
| wamjs, 2,000 pkgs (3,014 / 6,058) | **fails at the 2,000,000-step cap** | 350,022 / 706 ms | 277,442 / 663 ms | — |
| wamjs, 3,000 pkgs (`--stack-size=200000`) | cap | — | 409,362 / 1,034 ms | — |
| wamjs, 5,000 pkgs (`--stack-size=200000`) | cap (≈7.9M projected) | — | **675,217 / 1,695 ms**, correct selection | — |
| wamjs, B2 2,600 cases | 11,903,199 / 14.7 s | 8,872,767 / 16.4 s | 8,581,132 / 16.7 s | 8,626,924 / 18.7 s |
| Rust, 500 / 1,000 / 2,000 / 5,000 pkgs (resolve ms) | 471 / 848 / 1,465 / 3,333 | 1,115 / 4,206 / 15,079 / — | **1,041 / 4,017 / 13,718 / 117,509** | — |
| Rust, same, P2 with the tree replaced by a linear grouped list | | | 584 / 1,653 / 6,883 | |
| Rust, same, P2 building the keyed lists only (no sort, group, tree; scans used) | | | 401 / 788 / 1,531 | |
| Rust, same, keyed lists + `keysort/2` only (no group, tree; scans used) | | | 493 / 972 / 2,454 | |

Readings, each of which shapes a decision below:

1. **Instruction count scales as scans × rows at baseline and as rows at
   P2.** The wamjs per-instruction cost is what makes that matter; SWI does
   the baseline scans at ≈ 30 ns per row and is *faster without the index*.
2. **The index costs about 45 instructions per catalog row** on wamjs
   (`key_dep_rows/2` 27/row, `group_keyed/2` + `same_key/4` 17/row, tree
   build under the lowered tier). It pays for itself after roughly two
   scans. B2's catalogs have ≈ 20 rows and one to three scans per query, so
   B2 loses 11–14 % wall while dropping 25–28 % of its instructions: the
   sort builtins run natively and are not counted.
3. **The Rust leg is quadratic on the index build as prototyped.** Baseline
   is linear on this box (3.3 s at 5k); P2 is 35× slower at 5k and the ratio
   grows with size. Four bisection builds (table rows above) locate it:
   building the keyed lists alone is **linear** (401 / 788 / 1,531 ms,
   indistinguishable from baseline), so constructing long lists is fine;
   adding `keysort/2` alone is mildly super-linear (+0.1 / +0.2 / +0.9 s,
   and the `sort/2`-with-positions form P1 behaves the same, so both sort
   builtins share the cost); adding `group_keyed/2` / `same_key/4` on top
   is strongly super-linear (+0.1 / +0.7 / +4.4 s); and the tree build
   roughly doubles that again. What `same_key/4` and `build_tree/4` have
   in common, and the row builders do not, is **returning a suffix of a
   long list through an output argument** (`Rest1 = [K2-X|Rest]`,
   `Rest = Pairs`). The identical program is linear on wamjs and SWI, so
   this is a `WAM_RUST_STATUS.md` finding — a runtime shape to profile,
   not a spec property — and §5 step 1c says what to do with it before
   any Rust claim is made.
4. **The Go leg cannot be measured on the P3 source at all today** (571
   divergences against SWI at baseline and `fail` on the 5k probe): D57
   recorded that the Go/Rust/ClojureScript lanes were not P3-ported.
   Rust runs the P3 source (40/40 sampled differential cases, 0
   divergences) but see reading 3.
5. **The wamjs term leg has two hard ceilings unrelated to this design**:
   `Runtime.run` returns `false` after 2,000,000 steps (`wam_runtime.js`,
   `const limit = 2000000`), which is why baseline "fails" above ≈ 1,200
   packages; and T4-lowered list walkers recurse on the JS stack, so lists
   above ≈ 4,500 cells overflow the default stack (`RangeError` inside
   `lowered_key_pkg_rows_2`). Both are recorded as prerequisites in §5.

---

## 2. Guard catalog

Notation: `Cat` is the catalog term; `Acc` the partial selection (newest
first); a **test** is a goal that is called with all its arguments ground
(or whose bindings cannot escape) and whose only observable is
success/failure; "det" means exactly one solution, no choice point left.
The entailment arguments rely on two facts about the current code that the
reviewer should confirm first: (i) `resolve/3` and `resolve_layered/3` cut
at the API edge, so only the **first** solution of `resolve_pending/5` is
observable, and it is determined by the order in which choice points are
created and the order of solutions each one yields; (ii) every guard below
either replaces a goal by a logically equivalent one that yields the *same
solutions in the same order*, or inserts a test that can only fail on a
branch that would otherwise yield no solution. Either preserves the first
solution and all `explain_*` enumerations.

### G1 — per-call catalog index (accept; the load-bearing change)

**(a) Site.** `resolve/3` and `resolve_layered/3` (API edge), the catalog
accessors (`packages/2` … `provides_list/2`), `collect_deps/4`,
`candidates_high_first/4`, and optionally `base_ver/3`, `no_acc_conflicts/4`,
`already_provided/4` / `provides_sat/5`, `provider_candidate/5`.

**(b) Sketch.** Wrap the catalog in an internal term that carries balanced
lookup trees; every accessor gains one delegating clause so the rest of the
file is untouched; the two hot lookups take the tree when present and fall
back to the scan otherwise.

```prolog
resolve(Cat0, Requests, Selection) :-
    index_catalog(Cat0, Cat),              % new
    map_requests(Cat, Requests, Pending),
    resolve_pending(classic, Cat, Pending, [], Acc),
    !,
    sort(Acc, Selection).
% resolve_layered/3 identically.

packages(icat(Cat, _, _), Ps)     :- packages(Cat, Ps).      % one per accessor,
depends_list(icat(Cat, _, _), Ds) :- depends_list(Cat, Ds).  % placed next to it
% ... conflicts_list, base_list, installed_list, requested_list,
%     layers_list, excluded_list, alias_list, provides_list
dep_index(icat(_, DepT, _), DepT).
pkg_index(icat(_, _, PkgT), PkgT).

index_catalog(Cat, icat(Cat, DepT, PkgT)) :-
    depends_list(Cat, Ds),
    key_dep_rows(Ds, 0, KDs),  sort(KDs, SDs),  group_keyed(SDs, GDs),
    list_to_tree(GDs, DepT),
    packages(Cat, Ps),
    key_pkg_rows(Ps, 0, KPs),  sort(KPs, SPs),  group_keyed(SPs, GPs),
    list_to_tree(GPs, PkgT).

% (Name-Ver)-Pos-Req: sort/2 on this shape is a stable sort by key because
% Pos is unique, so no two elements compare equal and none is dropped.
key_dep_rows([], _I, []).
key_dep_rows([depends(N, V, D, C)|Rest], I, [(N-V)-I-Req|Ks]) :-
    dep_to_req(D, C, Req),
    I1 is I + 1,
    key_dep_rows(Rest, I1, Ks).

key_pkg_rows([], _I, []).
key_pkg_rows([package(N, V)|Rest], I, [N-I-V|Ks]) :-
    I1 is I + 1,
    key_pkg_rows(Rest, I1, Ks).

group_keyed([], []).
group_keyed([K-_-X|Rest], [K-[X|Xs]|Gs]) :-
    same_key(Rest, K, Xs, Rest1),
    group_keyed(Rest1, Gs).

same_key([], _K, [], []).
same_key([K2-I-X|Rest], K, Xs, Rest1) :-
    (   K2 == K
    ->  Xs = [X|Xs1], same_key(Rest, K, Xs1, Rest1)
    ;   Xs = [], Rest1 = [K2-I-X|Rest]
    ).

list_to_tree(Pairs, Tree) :-
    length(Pairs, N),
    build_tree(N, Pairs, Tree, []).

build_tree(N, Pairs, Tree, Rest) :-
    (   N =:= 0
    ->  Tree = t, Rest = Pairs
    ;   NL is (N - 1) // 2, NR is N - 1 - NL,
        build_tree(NL, Pairs, L, [K-V|Mid]),
        build_tree(NR, Mid, R, Rest),
        Tree = t(L, K, V, R)
    ).

tree_lookup(t(L, K, V, R), Key, Val) :-      % fails on the empty tree `t`
    compare(Ord, Key, K),
    (   Ord = (=) -> Val = V
    ;   Ord = (<) -> tree_lookup(L, Key, Val)
    ;   tree_lookup(R, Key, Val)
    ).

collect_deps(Cat, Name, Ver, Reqs) :-
    (   dep_index(Cat, T)
    ->  ( tree_lookup(T, Name-Ver, Reqs0) -> Reqs = Reqs0 ; Reqs = [] )
    ;   depends_list(Cat, Ds), matching_deps(Ds, Name, Ver, Reqs)
    ).

candidates_high_first(Cat, Name, C, Ver) :-
    \+ excluded_name(Cat, Name),
    matching_versions_in(Cat, Name, C, Vs),
    sort_versions_desc(Vs, Desc),
    member(Ver, Desc).

matching_versions_in(Cat, Name, C, Vs) :-
    (   pkg_index(Cat, T)
    ->  ( tree_lookup(T, Name, All) -> filter_satisfies(All, C, Vs) ; Vs = [] )
    ;   packages(Cat, Ps), matching_versions(Ps, Name, C, Vs)
    ).

filter_satisfies([], _C, []).
filter_satisfies([V|Vs], C, Out) :-
    (   satisfies(V, C) -> Out = [V|Os] ; Out = Os ),
    filter_satisfies(Vs, C, Os).
```

Variant **G1-ks**: drop the position tag and use `keysort/2` on `K-Row`
pairs (measured as P2 above: ≈ 20 % fewer build instructions). It is
correct only if the leg's `keysort/2` is **stable**; see (e).

Sub-indexes (same machinery, each optional and separately gated):

- **G1c — held versions.** Flatten `Base ++ Layers` in `lookup_held/3`'s
  visiting order (depth-first through `layer/2`), sort by `Name-Pos`, keep
  the **first** row per name, tree it; `base_ver/3` becomes a lookup.
  Entailment: `lookup_held/3` commits (`->`) to the first `item_ver/3`
  match in that exact order, so "first row per name in visiting order" is
  its definition. With an unbound `Name` both fail (`N == Name` never holds
  for a fresh variable; `compare/3` sends a variable left to the empty
  tree). Measured (P3): `lookup_held/3` 950,464 → 720,794 B2 instructions,
  the residue being callers that hold the raw `Cat` (`explain_*`,
  `freeze_audit`, `layer_closure`'s topo pass). Small win; take it only if
  the raw-`Cat` callers are also routed through the index.
- **G1d — conflicts.** Two trees: rows by `Name` and rows by `Other`.
  `no_acc_conflicts(Cat, Pkg, Ver, Acc)` becomes: look up forward rows of
  `Pkg`, fail if any has `V == Ver` and `Other` selected in `Acc`; look up
  reverse rows keyed `Other == Pkg`, fail if any `(N, V)` is in `Acc`.
  Entailment: the predicate is a test; the new form computes the same
  boolean `¬∃ O-OV ∈ Acc . conflicts(Pkg,Ver,O) ∈ Cs ∨ conflicts(O,OV,Pkg) ∈ Cs`.
  Worth 2.7 % of B2; nothing on B3 (no conflicts). Not prototyped.
- **G1e — provides by virtual name.** `provides_sat/5` and
  `provider_candidate/5` walk the provides list per `Acc` entry / per
  request. Group rows by `Virtual` in list order. Entailment for
  `provides_sat/5`: a test. For `provider_candidate/5` the **order** of
  solutions must be the provides-list order restricted to that virtual,
  which a stable grouping preserves. Worth 3.4 % of B2; nothing on B3. Not
  prototyped; the order requirement makes it the one sub-index where the
  reviewer should insist on a solution-order probe (§6).

**(c) Entailment.**

1. *Same rows, same order.* `matching_deps(Ds, Name, Ver, Out)` is, by its
   clauses, "the list of `dep_to_req(D, C)` for every row
   `depends(N, V, D, C)` of `Ds`, in list order, with `N == Name` and
   `V == Ver`". The index sorts `(N-V)-Pos-Req` triples by standard order.
   Because `Pos` is distinct per row, no two triples are equal, so `sort/2`
   drops nothing and within one key the triples are in ascending `Pos`,
   i.e. list order. `group_keyed/2` gathers consecutive equal keys
   (`==`), so the value stored under `Name-Ver` is exactly the list above.
   The only way the two could differ is if some pair of ground keys were
   `==`-identical yet not adjacent after sorting, or adjacent-and-equal
   under standard order yet not `==`. Standard order on ground terms is a
   total order with equality iff structural identity (SWI: `1` and `1.0`
   compare *unequal*, Float before Int), so neither can happen. Same
   argument for `package(N, V)` rows grouped by `N`: the version list is
   the catalog-order list of that name's versions, duplicates included.
2. *Same candidate sequence.* `candidates_high_first/4` used to compute
   `Vs` = catalog-order versions of `Name` that satisfy `C`; it now filters
   the indexed catalog-order list with the same `satisfies/2` test, giving
   the same `Vs`; `sort_versions_desc/2` is unchanged, so the `member/2`
   enumeration is identical, including its choice points.
3. *Missing key ≡ empty scan.* A name or `Name-Ver` absent from the tree
   yields `[]` for deps and no candidates, exactly what the scan yields.
4. *No new observable.* `icat/3` is consumed only through the accessors and
   the two lookups; it is never part of a solution, a `blocked/3` term or
   an exception. Every accessor has an `icat` clause, so no clause of the
   search can see a different catalog than before. `explain_blocked/3`,
   `layer_closure/3` (its topo pass), `upgrade_set/4`, `freeze_audit/2`,
   `dependents/3` and `removal_orphans/3` keep taking the raw catalog and
   are byte-for-byte unaffected unless step 3 of §5 routes them too.
5. *Determinism preserved.* `index_catalog/2` is det (its helpers are
   if-then-else recursions and builtins); it leaves no choice point, so
   the API-edge cut sees the same choice-point stack as before.

**(d) Expected effect.** Per query: scans × rows → rows + picks × log₂(keys).
5k catalog on wamjs: ≈ 7.9M → 675k instructions (11.7×); 1,000 packages
8.8×. B2: −25 % instructions, +11–14 % wall (index build not amortised).
SWI: −50 % inferences, +45 % wall (see §1.5 reading 1).

**(e) Risks.**

- *`keysort/2` stability (G1-ks only).* ISO requires it; the JS runtime
  passed 0 divergences with it, but a leg whose `keysort/2` is a plain
  unstable sort would reorder a package's dependency requests and thereby
  change search order. The position-tagged `sort/2` form has no such
  assumption and is the recommended default; adopt G1-ks per leg only after
  the stability probe in §6.
- *Standard order across legs.* `sort/2` on `Name-Ver` keys is already
  exercised by every selection the legs return (`sort(Acc, Selection)`),
  including `deb/3` keys in the P3 rows, so `compare/3`/`sort/2` agreement
  on these shapes is gate-covered. `compare/3` exists in the JS, Go, Rust,
  ClojureScript and F# runtimes (checked in the runtime sources).
- *Duplicate catalog rows.* Duplicate `package(N, V)` rows produce duplicate
  candidate branches today; the index preserves multiplicity, so nothing
  changes. `sort_versions_desc/2` dedups after the filter in both worlds.
- *Lowered-tier recursion (wamjs).* The new list walkers are lowered by T4
  and recurse on the JS stack; above ≈ 4,500 cells the default stack
  overflows. The baseline walkers were interpreted. Either run the term
  leg with `node --stack-size`, or keep the index builders on the
  interpreter tier. This does not affect the store leg (`resolver_store.pl`
  has its own seek-based search and needs no index).
- *Rust quadratic build.* See §1.5 reading 3 and §5 step 1c.
- *Cost on tiny catalogs.* A size threshold (`length(Ds, N), N >= 64 ->
  index ; raw`) is sound — both branches are the same relation — and would
  keep B2 at baseline. It is an implementation option, not a semantic one.
- *Layered `from_base` re-expansion stays.* G1 makes each re-expansion a
  log-time lookup; it does not remove it (that would be R2).

### G2 — conflict test before dependency expansion (accept; minor)

**(a) Site.** `resolve_pending/5`, the `from_catalog` arm.

**(b) Sketch.**

```prolog
    ;   pick_need(Mode, Cat, Name, C, Acc, Pkg, Ver, Origin),
        (   Origin = from_base
        ->  collect_deps(Cat, Pkg, Ver, DepReqs),
            append(DepReqs, Rest, More),
            resolve_pending(Mode, Cat, More, Acc, Sel)
        ;   no_acc_conflicts(Cat, Pkg, Ver, Acc),      % moved up
            collect_deps(Cat, Pkg, Ver, DepReqs),
            append(DepReqs, Rest, More),
            resolve_pending(Mode, Cat, More, [Pkg-Ver|Acc], Sel)
        )
```

**(c) Entailment.** Today the arm is `collect_deps, append, (from_base ->
… ; no_acc_conflicts, …)`. `collect_deps/4` and `append/3` (first two
arguments bound) are det and always succeed; `no_acc_conflicts/4` is a
test on ground arguments (`Pkg`, `Ver` are bound by `pick_need/8`, `Acc` is
ground). Permuting a ground test past det always-succeeding goals changes
neither the solutions nor their order nor the choice points; it only
avoids the two goals on the failing path. The `from_base` arm is
unchanged.

**(d) Effect.** One dependency lookup/scan saved per conflict-rejected
candidate. Classic mode only in practice (layered commits to one
candidate). With G1 in place the saving is a tree lookup; without G1 it is
a full depends scan per rejected candidate. Zero on B3.

**(e) Risks.** None found. It is included because it is free and because
it is the shape the store adapter (`resolve_pending_store/5`) should mirror
if the two searches are kept in lockstep.

### G3 — single-pass `no_acc_conflicts/4` (accept if G1d is not taken; minor)

**(a) Site.** `no_acc_conflicts/4`.

**(b) Sketch.** Replace |Acc| × 2 scans of the conflicts list by one scan
that checks each row against `Acc`:

```prolog
no_acc_conflicts(Cat, Pkg, Ver, Acc) :-
    conflicts_list(Cat, Cs),
    \+ (   member(conflicts(N, V, O), Cs),
           (   N == Pkg, V == Ver, member_selected(Acc, O, _)
           ;   O == Pkg, member_selected(Acc, N, V)
           )
       ).
```

**(c) Entailment.** Both forms decide
`¬∃ O-OV ∈ Acc . conflicts(Pkg,Ver,O) ∈ Cs ∨ conflicts(O,OV,Pkg) ∈ Cs`; the
new one enumerates the same conjunction in the other nesting order. It is
a test (called with `Pkg`, `Ver`, `Acc` ground) so enumeration order is
unobservable. One subtlety the reviewer should check: `member_selected/3`
unifies `Name-Ver` against `Acc` entries; with `O` and `N`, `V` ground from
the row, that is a membership test, as before.

**(d) Effect.** O(|Acc|·|Cs|) → O(|Cs|). 2.7 % of B2. Nothing on B3.
Superseded by G1d if that is implemented.

### G4 — layered version-ceiling fail-early at expansion (accept, optional; flagged)

**(a) Site.** `resolve_pending/5`, immediately after `collect_deps/4` in
both arms, layered mode only.

**(b) Sketch.**

```prolog
        collect_deps(Cat, Pkg, Ver, DepReqs),
        no_doomed_req(Mode, Cat, DepReqs),           % new
        append(DepReqs, Rest, More),

no_doomed_req(classic, _Cat, _Reqs).
no_doomed_req(layered, Cat, Reqs) :-
    \+ (   member(req(D, C), Reqs),
           atom(D),
           base_ver(Cat, D, BV),
           \+ satisfies(BV, C),
           \+ provides_mentions(Cat, D)
       ).

provides_mentions(Cat, D) :-
    provides_list(Cat, Prs),
    member(Row, Prs),
    provide_row(Row, P, _V, Virt, _VV),
    ( P == D -> true ; Virt == D ).
```

**(c) Entailment.** Claim: in layered mode, if `DepReqs` contains
`req(D, C)` with `D` an atom, `base_ver(Cat, D, BV)`, `¬satisfies(BV, C)`,
and `D` occurring in no provides row (as provider or as virtual), then
`resolve_pending(layered, Cat, DepReqs ++ Rest, Acc', Sel)` has **no**
solution for any `Acc'` reachable from here, so failing now prunes only
non-solutions and the enclosing enumeration (next candidate, next
provider, next alternative) proceeds exactly as it would after the doomed
subtree failed.

Proof sketch. To yield a solution the pending list must be consumed in
order, so `req(D, C)` is eventually reached with some `Acc''` ⊇ `Acc'`
(the accumulator only grows along a branch). At that point the clause for
`req(D, C)` tries, in order: (1) `Name = alternatives(_)` — no, `D` is an
atom; (2) `selected_ver(Acc'', D, _)` — `D` can enter `Acc` in layered
mode only through `pick_need/8`'s `from_catalog` arms, and the real-name
arm is unreachable for a held name (the `base_ver` arm commits first), so
only the provider arm could add `D`, which requires a provides row with
`P == D`; excluded by hypothesis. (3) `already_provided(Cat, Acc'', D, C)`
requires a provides row with `Virt == D`; excluded. (4) `pick_need(layered,
…, D, C, …)`: `base_ver` succeeds, commits, and `satisfies(BV, C)` fails —
by hypothesis, and `base_ver`/`satisfies` depend only on `Cat` and `C`, not
on `Acc''`. Hence the branch fails. ∎

The two hypotheses are not decoration: probes CE3 and CE4 (§6.1) show a
held name entering `Acc` at another version through a provider, and a held
name with an unsatisfiable ceiling being satisfied by a provider in `Acc`
— exactly the cases the `provides_mentions/2` check excludes. A version of
G4 without that check is **unsound** (it would fail CE4).

**(d) Effect.** Saves the linear work between the expansion and the doomed
request (sibling picks, their lookups, their own expansions), then the
backtracking that would re-reach it. Zero on B3 (nothing is blocked). On
the real bookworm slice with Essential holds pinned old (the D57 demo,
`libc6` blocked), it fails at the first expansion that mentions `libc6`
rather than after resolving the siblings. Bounded by §1.3: layered has no
version fan-out, so this is a constant-factor saving, not an asymptotic
one. Cost: one `base_ver/3` per request at expansion (cheap with G1c), and
a provides-list scan only on the rare hit.

**(e) Risks / flags.** This is the one guard whose soundness rests on an
enumeration of "how can `D` become satisfied later"; §6 restates it as
three falsifiable assertions and asks the reviewer to try to break each.
Interaction with explanation terms: none (`explain_*` does not call
`resolve_pending/5`). Interaction with alternatives: `req(alternatives(_),
any)` entries are skipped by `atom(D)`, and members of an alternatives
group are only ever requested through `resolve_alternatives/6`, which
re-enters `resolve_pending/5` and thus gets the same treatment at *its*
expansion. Interaction with H1: none — the commit in `pick_need/8` is what
makes step (4) of the proof a plain failure.

---

## 3. Rejected and uncertain

### 3.1 Rejected families (with counterexamples the reviewer can run)

**R1 — most-constrained-first, or any reordering of the pending list.**
The first solution of a depth-first search depends on the order in which
choice points are created. Probe CE1: `app` depends on `b` and `c`; `b-2.0`
needs `d = 1.0`, `c-2.0` needs `d = 2.0`. With the depends rows in order
`[b, c]` the answer is `[app, b-2.0, c-1.0, d-1.0]`; with `[c, b]` it is
`[app, b-1.0, c-2.0, d-2.0]`. Any ordering heuristic that can swap two
requests changes answers on catalogs like this one, and the seeded
differential contains many diamonds. Rejected outright; there is no sound
variant, because the invariant is *this* order.

**R2 — memoising `from_base` expansions (skip the second walk of a held
package).** The second expansion pushes the held package's requests to the
*front* of the pending list, ahead of whatever the current package still
had pending. Skipping it moves those requests later. Probe CE2: base
`base0` depends on `a, b`; `a` depends on `base0, x`; `b-2.0` needs `d =
1.0`, `x-2.0` needs `d ≥ 1.0`. Current order `a, b, x` answers `[a, b-2.0,
d-1.0, x-2.0]`; the memoised order `a, x, b` picks `d-2.0` for `x` and then
`b-2.0` fails on `d = 1.0` — and, because of H1, layered does not retry
`b-1.0`, so the query **fails**. A filtered variant ("push only the
not-yet-settled requests") is also unsound because `selected_ver/3`
returns the *newest* entry and H2 (below) lets a second version of a name
be pushed later, so "settled" is not monotone. With G1 the re-expansion
costs one lookup, which removes the motivation.

**R3 — alternatives-group pruning (skip an alternative whose head is
excluded or conflicted).** An excluded name can still be satisfied: by a
loaded layer (`pick_need(layered)` consults `base_ver/3` before
`excluded_name/2` is ever reached) or by a provider already in `Acc`
(`already_provided/4`); exclusion filters *candidate generation* only, per
the README. A name-level conflict against `Acc` blocks `from_catalog`
picks of that name but not `from_base` use, nor providers of it (which are
different packages). The sound residue — "do not scan the package list for
an excluded name" — is already the first goal of `candidates_high_first/4`,
and "do not expand deps of a conflicting candidate" is G2. Nothing
separate survives.

**R4 — propagating version ceilings to non-held dependencies (prune
candidate versions that a later constraint would reject).** Constraints on
a name arrive in pending-list order; a version rejected by a *later*
request is exactly what the search rejects when it gets there, and
pre-empting it requires knowing the future `Acc`, which R2's argument
shows is not monotone. In classic mode this would also change *which*
version is tried first when a later constraint is only reachable on some
branches. Rejected.

**R5 — deduplicating or dropping "already satisfied" requests from the
pending list at push time.** A request whose name is selected but whose
constraint is *not* satisfied fails the branch today (the diamond commit);
dropping it would turn a failure into a success. Keeping only the failing
ones is the same work as processing them. Rejected as a guard; note it is
what makes R2's filtered variant unsound too.

**R6 — early ceiling check in classic mode.** No base is consulted in
classic mode; there is nothing to check. Not applicable.

### 3.2 Pre-existing hazards found (preserved by this design; owner's call)

Each was confirmed on the committed `resolver.pl` with a probe in §6.1.

- **H1 — layered mode does not backtrack over candidate versions since
  P3** (§1.3, CE6). P0.5 did. Probably unintended; the corpus has no
  layered scenario that needs a lower version, so the gate cannot see it.
- **H2 — one name, two versions, via the provider path** (CE3). `resolve`
  on `app` needing `mawk = 1.0` and virtual `awk` provided only by
  `mawk-2.0` answers `[app, mawk-1.0, mawk-2.0]`; layered with `mawk-1.0`
  held answers `[app, mawk-2.0]`. The diamond rule ("a second version of
  the same name is never added") is enforced on the *requested* name, not
  on the *selected* provider.
- **H3 — `resolve_layered` and `explain_blocked` disagree when a held
  name's ceiling is bypassed by a provider** (CE4): the resolve succeeds
  through `bar` providing `foo ≥ 2.0` while `explain_blocked_list` reports
  `blocked(foo, needs(gte(2.0)), base_has(1.0))`.
- **H4 — cyclic dependencies among held packages do not terminate in
  layered mode** (CE5): `from_base` entries never enter `Acc`, so nothing
  detects the cycle. Classic mode is safe (the name is in `Acc`).
- **H5 — named-layer providers are unreachable.** The second clause of
  `layer_provider/5` and the third of `layer_satisfies/3` call
  `lookup_held(Pkgs, P, PV)` with `P` unbound; `item_ver/3` tests `N ==
  Name`, which never holds for a fresh variable, so these clauses never
  succeed. Dead code, or a bug, depending on intent.
- **H6 — the JS term leg's 2,000,000-step cap** silently returns `fail`
  (the shim reports `{"fail":true}`) instead of an error, which makes a
  large catalog indistinguishable from an unsatisfiable request.

None of these is changed by G1–G4. H1 and H4 in particular are things a
future "fix" would change answers on; if the owner fixes them, the
invariants in §4 must be re-baselined first.

---

## 4. Invariants (every implementation must preserve)

1. **Same first solution** for `resolve/3` and `resolve_layered/3` on every
   catalog: the 50-scenario corpus and the 2,600-case term differential
   (`gen_catalogs.mjs`, seeds unchanged) must be **byte-identical** between
   the pre-change and post-change SWI oracle outputs (`diff_runner.pl`
   → `swi.jsonl`), not merely 0 divergences against the legs. The 503-case
   store differential likewise (the store adapter is untouched by G1; if G2
   is mirrored there, the same identity test applies).
2. **Same explanation terms**: `explain_blocked/3` enumerations and
   `explain_blocked_list/3` results identical, including `providers([...])`
   and `alternatives([...])` reason lists and their order.
3. **All ten queries** byte-identical, both worlds (`v/3` and `deb/3`), all
   catalog arities (`catalog/6`, `/9`, `/10`).
4. **Determinism policy unchanged**: real package before providers,
   providers in catalog-list order, highest version first, layered's
   commit (H1) included.
5. **No `assert`/`retract`, no side effects**; the index is a term built
   per call and dropped with it.
6. **No new cuts** outside the API edges; if-then-else only, as the file
   already does.
7. **No catalog-shape change**: `icat/3` (or whatever the wrapper is
   named) never escapes `resolver.pl` and is never accepted as input.
8. **Solution multiplicity** for enumerating predicates (`explain_blocked/3`,
   internal `provider_candidate/5`) unchanged, duplicates included.
9. **Leg parity**: each leg that is built from `resolver.pl` re-runs its
   corpus and differential at 0 divergences; the cut-semantics probes
   (§9 of `WAM_BACKEND_CONVENTIONS.md`) stay green since no cut moves.

---

## 5. Implementation plan (one round; each step has a measurement and a gate)

**Step 0 — baseline the instruments (½ day).** Record, per leg, the numbers
the later steps compare against, on one box:
- SWI: `statistics(inferences)` and cputime over 50 repetitions of the 5k
  `resolve_layered(p30)` (this document's `timing.pl` shape), plus B2 wall.
- wamjs: `UW_PROFILE=json` totals for B2 and for prefix slices of the 5k
  catalog at 250 / 500 / 1,000 packages (prefix slices are closed under
  dependencies because the generator only depends backwards; `p30`'s
  answer is identical at every size ≥ 31).
- ClojureScript: the D54 step census (`bench_scale.cljs`) at 5k.
- Rust: `uw_resolve --bench` at 500 / 1,000 / 2,000 / 5,000.
- Go: blocked until the lane is P3-ported (D57 residual); record that.
Gate: none; this is the yardstick.

**Step 1 — G1a + G1b, position-tagged `sort/2` form (1–2 days).** Land the
wrapper, the ten accessor clauses (each next to its accessor, so no
`discontiguous` warnings), `index_catalog/2`, the two lookups, and route
only `resolve/3` and `resolve_layered/3` through it. Sub-steps:
- **1a.** SWI corpus 50/50; SWI differential byte-identical (invariant 1);
  store differential unchanged (nothing touched).
- **1b.** wamjs build; corpus 51/51; differential 0 divergences; B2 profile
  and the prefix-slice profiles. Expected: −25 % B2 instructions, ≥ 8× at
  1,000 packages. Decide the size-threshold option on the measured B2 wall.
  Run the 5k case with `node --stack-size=200000` and the step cap raised
  or profiled (`UW_PROFILE` disables the cap) to reproduce the 675k figure.
- **1c. Rust: find the quadratic before enabling.** Build the leg and run
  the size ladder (§1.5). The bisection in §1.5 reading 3 already
  separates three super-linear shapes on this runtime: `keysort/2` /
  `sort/2` on a 10⁴-element list (mild), and two predicates that hand a
  suffix of a long list back through an output argument — `same_key/4`
  and `build_tree/4` (severe). Steps, in order: (i) reproduce each with a
  ten-line probe program (a `keysort/2` of 15k pairs; a `split_at/4`
  returning the tail; a `build_tree/4` over 7k pairs) and profile with
  `perf`/callgrind as D55 did — the expected hotspot is a dereference or
  copy of the whole remaining list per output binding; (ii) if the fix is
  a runtime change, land it as its own probe-pinned row; (iii) if it is
  not fixable in the round, try the tree-free grouped list plus a
  `same_key/4` written without the returned suffix (e.g. a single pass
  that emits `Key-Group` pairs and the leftover in one accumulator), and
  measure again. This is a runtime finding to record in
  `WAM_RUST_STATUS.md`, not a spec change. Do not merge a Rust regression:
  if the cause is not fixable in the round, gate the index behind the
  size threshold **on all legs** (it is one Prolog edit) and record the
  Rust curve.
- **1d.** ClojureScript step census at 5k. Expected ≈ 10× fewer steps; the
  lane's per-step cost (~5 µs, D54) makes this the largest wall win in the
  fleet.
Gate for the step: invariants 1–9; the D57 cross-lane note applies (Go and
ClojureScript must be P3-ported to build at all).

**Step 2 — G2 (½ day).** Reorder the `from_catalog` arm. Gate: invariant 1
byte-identity; wamjs B2 profile (expect a small instruction drop only on
conflict-heavy cases). Mirror into `resolve_pending_store/5` only if the
store search is meant to stay textually parallel; if mirrored, the store
differential must be byte-identical too.

**Step 3 — G1c/G1d/G1e and routing the read-only queries (1 day,
optional).** Route `explain_blocked*`, `layer_closure`'s topo pass and
`upgrade_set*` through the index (they call `collect_deps/4`,
`depends_in/5`, `base_ver/3`). `depends_in/5` is an enumerator (used with
`Dep`, `C` unbound by `dep_targets/5` and `follow_dep_name/5`); an indexed
form must enumerate the same rows in list order — the grouped list under
`Name-Ver` does exactly that. Gate: invariants 1–3 and 8, B2 profile
(`lookup_held/3`, `dep_breaks/5`, `provides_sat/5` shares).

**Step 4 — G4 (½ day, optional, layered only).** Add `no_doomed_req/3`.
Gate: byte-identity plus the three assertions in §5 turned into corpus
rows (CE4's catalog **must** be added as a scenario so the
`provides_mentions/2` hypothesis is pinned by a red bar if someone drops
it). Measure on the bookworm slice with the D57 Essential-holds
configuration.

**Step 5 — ledger and benchmarks (½ day).** Add the D-row; refresh
`BENCHMARKS.md` B3 per leg with before/after; record H1–H6 in the resolver
README's deferred list or as filed issues, owner's choice.

Sizing: steps 0–2 are the round; 3–4 are stretch. The measurement for
every step is "instructions per leg (wamjs `UW_PROFILE`, ClojureScript
census, SWI inferences) and wall, before and after, same box", and the
gate for every step is "SWI byte-identical on 2,600 + 503 cases, every
built leg at 0 divergences".

---

## 6. Review protocol

For each guard the reviewer should try to **falsify** the assertion, not
confirm the argument. Concrete catalogs to try are in §6.1; the prototype
recipe in §6.2 re-creates the measurements.

**G1 (index).**
- A1. For every ground `Ds` and every `Name`, `Ver`:
  `collect_deps` via the tree ≡ `matching_deps(Ds, Name, Ver, _)`, as
  lists (same elements, same order, same multiplicity). Try: duplicate
  rows; two rows differing only in constraint; `deb/3` keys that are
  `version_lt`-equal but not `==` (e.g. differing only in a `~`
  segment's spelling) — these must stay **separate** keys, because
  `matching_deps` compares with `==`.
- A2. For every `Name`: the indexed version list ≡ catalog-order versions
  of `Name`, duplicates included, before `satisfies/2` filtering.
- A3. Every accessor has an `icat` clause and `icat` never appears in any
  output (grep the corpus and differential outputs for the functor).
- A4. `index_catalog/2` is det: `findall(_, index_catalog(Cat, _), L),
  length(L, 1)` on every corpus catalog, and no choice point remains
  (`deterministic/1` in SWI).
- A5 (G1-ks only, per leg). `keysort/2` stability: sort
  `[b-1, a-1, b-2, a-2, b-3]` and require `[a-1, a-2, b-1, b-2, b-3]`
  on that leg's runtime, on lists of 10⁴ pairs too.
- A6 (per leg). Standard order on `Name-Ver` keys agrees with SWI for
  `v/3` and `deb/3`, including `s([],0)` padding and `~` segments —
  already gate-covered by `sort(Acc, Selection)`, but state it.

**G2.** A7. `no_acc_conflicts/4` is called with `Pkg`, `Ver`, `Acc` ground
at that site (bound by `pick_need/8` and the accumulator); confirm no
clause of `pick_need/8` can leave `Ver` unbound (e.g. a `provides/3` row
whose package has no `package/2` row — `provider_candidate/5` checks
`package_in/3`, so no).

**G3.** A8. Truth-table the two forms on a 3-entry `Acc` against a
conflicts list containing forward, reverse, and both-direction rows and
rows for unrelated names.

**G4.** Three assertions; the hypotheses are the interesting part:
- A9. In layered mode a held name `D` enters `Acc` **only** through
  `provider_candidate/5` with `Pkg == D`. Try to find another path
  (aliases at the request edge? `resolve_alternatives/6`? a held name that
  is also a virtual?).
- A10. `req(D, C)` with `D` held, `¬satisfies(BV, C)`, `D ∉ Acc`, is
  satisfied by `already_provided/4` **only** if a provides row has
  `Virt == D`. Try `provides/4` with a versioned virtual equal to `D`.
- A11. `base_ver/3` and `satisfies/2` at the request site depend only on
  `Cat` and `C`, not on `Acc` — i.e. nothing in the search mutates the
  base (there is no write path; confirm no accessor is ever called with a
  modified catalog).
- Flag: the author is **less than certain** that A9's enumeration is
  exhaustive for catalogs where the *same* atom is used as a real
  package name, a held name, and a virtual name simultaneously (nothing
  forbids it). CE3/CE4 cover two of the three pairings; the reviewer
  should build the third.

**H1–H6.** Not review items for this design, but the reviewer should
confirm each probe reproduces on the committed file before the round
starts, so that a later fix is not mistaken for a pruning regression.

### 6.1 Probes (run against `examples/pkg_resolver/resolver.pl`)

```prolog
:- use_module('examples/pkg_resolver/resolver').
v1(v(1,0,0)). v2(v(2,0,0)).

% CE1 — pending order decides the first solution (R1).
ce1 :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(b,V2), package(b,V1), package(c,V2),
          package(c,V1), package(d,V2), package(d,V1)],
    Dd = [depends(b,V2,d,eq(V1)), depends(c,V2,d,eq(V2))],
    resolve(catalog(Ps, [depends(app,V1,b,any), depends(app,V1,c,any)|Dd], [], [], [], []), [app], S1),
    resolve(catalog(Ps, [depends(app,V1,c,any), depends(app,V1,b,any)|Dd], [], [], [], []), [app], S2),
    S1 \== S2.
%   S1 = [app-1, b-2, c-1, d-1],  S2 = [app-1, b-1, c-2, d-2]

% CE2 — skipping a from_base re-expansion changes the answer (R2).
ce2 :-
    v1(V1), v2(V2),
    Ps = [package(base0,V1), package(a,V1), package(b,V2), package(b,V1),
          package(x,V2), package(x,V1), package(d,V2), package(d,V1)],
    Dd = [depends(b,V2,d,eq(V1)), depends(x,V2,d,gte(V1))],
    Ds1 = [depends(base0,V1,a,any), depends(base0,V1,b,any),
           depends(a,V1,base0,any), depends(a,V1,x,any)|Dd],
    Ds2 = [depends(base0,V1,a,any), depends(base0,V1,b,any),
           depends(a,V1,x,any)|Dd],                 % == "second expansion pushes nothing"
    resolve_layered(catalog(Ps, Ds1, [], [base0-V1], [], []), [base0], S1),
    \+ resolve_layered(catalog(Ps, Ds2, [], [base0-V1], [], []), [base0], _),
    S1 == [a-V1, b-V2, d-V1, x-V2].

% CE3 — two versions of one name via the provider path (H2).
ce3 :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(mawk,V2), package(mawk,V1)],
    Ds = [depends(app,V1,mawk,eq(V1)), depends(app,V1,awk,any)],
    Pr = [provides(mawk,V2,awk)],
    resolve(catalog(Ps, Ds, [], [], [], [], [], [], [], Pr), [app], S1),
    S1 == [app-V1, mawk-V1, mawk-V2],
    resolve_layered(catalog(Ps, Ds, [], [mawk-V1], [], [], [], [], [], Pr), [app], S2),
    S2 == [app-V1, mawk-V2].

% CE4 — held ceiling bypassed by a provider; explain disagrees (H3; G4 hypothesis).
ce4 :-
    v1(V1), v2(V2),
    Ps = [package(app,V1), package(bar,V1), package(foo,V2)],
    Ds = [depends(app,V1,bar,any), depends(app,V1,foo,gte(V2))],
    Cat = catalog(Ps, Ds, [], [foo-V1], [], [], [], [], [], [provides(bar,V1,foo,V2)]),
    resolve_layered(Cat, [app], S), S == [app-V1, bar-V1],
    explain_blocked_list(Cat, app, L),
    L == [blocked(foo, needs(gte(V2)), base_has(V1))].

% CE5 — cyclic held deps do not terminate in layered mode (H4).
ce5 :-
    v1(V1),
    Cat = catalog([package(a,V1), package(b,V1)],
                  [depends(a,V1,b,any), depends(b,V1,a,any)], [], [a-V1, b-V1], [], []),
    catch(call_with_time_limit(2, resolve_layered(Cat, [a], _)), time_limit_exceeded, true).

% CE6 — layered commits to the highest candidate (H1).
ce6 :-
    v1(V1), v2(V2),
    Ps = [package(b,V2), package(b,V1), package(x,V2), package(x,V1), package(d,V2), package(d,V1)],
    Ds = [depends(b,V2,d,eq(V1)), depends(x,V2,d,eq(V2))],
    Cat = catalog(Ps, Ds, [], [zz-V1], [], []),
    resolve(Cat, [b,x], S1), S1 == [b-V2, d-V1, x-V1],
    \+ resolve_layered(Cat, [b,x], _).
```

All six succeed on `dadbed63` (`swipl -q -g "ce1,ce2,ce3,ce4,ce5,ce6"
-t halt probes.pl`). CE1, CE2, CE3, CE4 and CE6 also succeed on prototypes
P1–P3, which is the point: the guards preserve the hazards.

### 6.2 Reproducing the measurements

- 5k catalog: `node examples/pkg_resolver/store/gen_scale_catalog.mjs DIR`;
  the SWI loader in `store/scale_demo.pl` (`load_rich_catalog/2`) builds
  the `catalog/9` term with base `[p0-v(0,0,0)]`.
- SWI profile: `profile(resolve_layered(Cat, [p30], _))` then
  `profile_data/1`; inferences via `statistics(inferences, _)` around 50
  repetitions.
- wamjs: `swipl -q -g main -t halt examples/pkg_resolver/wamjs/build.pl --
  SRC OUTDIR`, copy `resolver.mjs` / `diff_runner_wamjs.mjs` beside
  `OUTDIR/js`, then `UW_PROFILE=json node diff_runner_wamjs.mjs <
  case.jsonl` (the case is `cljs/scale_to_catalog.mjs`'s catalog JSON
  wrapped as `{"id","catalog","query":"resolve_layered","args":["p30"]}`;
  prefix slices keep packages `p0..p(n-1)` and their rows). `UW_PROFILE`
  also lifts the 2,000,000-step cap; add `--stack-size=200000` above
  ≈ 2,000 packages.
- Rust: `examples/pkg_resolver/rust/build.pl -- SRC PROJ`, copy the shim,
  `cargo build --release --bin uw_resolve`, then `uw_resolve --bench <
  case.json` with `rust/scale_to_case.mjs DIR N` (strip the P3 overlay
  rows from `rich.jsonl` first; the pre-P3 shim panics on `deb` versions).
- Differential identity: `swipl -q -g main -t halt
  examples/pkg_resolver/diff_runner.pl < cases.jsonl` for the baseline and
  for a copy of the runner whose `use_module` points at the prototype;
  `cmp` the two outputs.
