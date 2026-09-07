<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Deterministic recursion taxonomy (survey for a general lowering recognizer)

Status: **Survey** (2026-09-07). Scope: enumerate every deterministic recursion
*shape* the project actually uses — in PLAWK (`examples/plawk/`) and the
uw-resolve resolver (`examples/pkg_resolver/resolver.pl`, read-only, unmodified
by this survey) — so that a single, target-agnostic eligibility recognizer can
be designed for the WAM lowered tiers, per
[`WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md`](WAM_RUST_LOWERED_TIER_THROUGHPUT_PLAN.md)
§6c's generalization directive: "this project has more deterministic recursion
patterns than tail recursion, and the lowering recognizer must target the whole
family, not assume a tail-recursive loop." This document does not propose code;
it is the classification the recognizer design should be built from.

**Read-only.** No `resolver.pl`, `resolver_store.pl`, spec, template, or target
file was modified to produce this survey.

## 0. Dimensions, not silos

The eight named patterns below are not eight disjoint boxes — every concrete
predicate in the project is a *point* in a small space of orthogonal
dimensions, and the taxonomy is really a classification of that space. Reading
the space this way is itself the survey's main finding, because it is what
makes a *single* recognizer possible instead of one hack per named pattern:

1. **Self-call arity per clause** — how many recursive calls (to the predicate
   itself, or to a member of its mutual-recursion SCC) does one clause body
   make? 0 (base case), 1 (linear), 2+ (tree).
2. **Position** — is the last such call the clause's last goal (*tail*), or
   does the clause do more work after it returns (*non-tail* — a post-order
   combine, or a second helper walk)?
3. **Call-graph shape** — does the recursion stay within one predicate, or
   does it cross a strongly-connected component of two or more predicates
   calling each other (*mutual recursion*)? This is orthogonal to 1 and 2: a
   mutual-recursion SCC can itself be tail (region 2's `segs_lt`/`segs_lt_1`)
   or non-tail (`topo_all`/`topo_one`, §5 below).
3b. **Meta-call indirection** — does a clause body reach its next step via
   `call/N` with a callee resolved only at specific call sites (PLAWK's
   `process_all/4`), rather than a lexically fixed predicate? This adds a
   defunctionalization step in front of every other dimension (§6).
4. **Contained nondeterminism** — does the body call a genuinely
   nondeterministic goal, and if so, is it *committed* (`once/1`, cut, or an
   `->` condition) before the recursive step, or *aggregated* (`findall`/
   `bagof`/`setof`) into one list before the recursive step, or does it
   escape uncommitted to the caller (the class-(c) case, which cannot lower
   the cheap way — see `docs/reports/wam_rust_nondet_driver_classification.md`)?
5. **Termination argument** — structural decrease of an input list/integer
   (the common case), or a finite-domain-plus-visited-set argument (the
   transitive-closure case, §8), which needs its own soundness note because
   it is not literally a shrinking first argument.

A predicate can sit anywhere in this space: `topo_one` (§5, §8) is
non-tail + mutual-recursion + aggregate-then-iterate (inner `findall`) +
finite-domain-Seen-termination, all at once. The recognizer design in §11
is built to compose these dimensions rather than pattern-match a fixed list
of named shapes.

---

## 1. Tail recursion → native loop

### 1. Definition
A predicate (or its recursive clause, when a base case exists as a separate
clause) whose body's *last* goal is a call to itself, with any per-iteration
work done before that call and the accumulated result threaded forward (either
as an explicit accumulator argument, or as an as-yet-unbound tail variable
filled in by the recursive call — a difference-list-style "build top-down").

### 2. Concrete examples
- `lookup_held/3`, `examples/pkg_resolver/resolver.pl:138-142` — single
  self-call in the else-branch of a committed if-then.
- `matching_deps/4`, `resolver.pl:608-616` (region 1), `matching_versions/4`,
  `resolver.pl:281-288` (region 2), `key_dep_rows/3`, `resolver.pl:410-414`
  (region 3a) — all "walk a list, decide per element via a committed `->`,
  cons the kept element onto a still-open output tail, recurse last."
- `filter_satisfies/3`, `resolver.pl:273-279` — same shape as
  `matching_versions/4` (a per-element `satisfies/2` filter, committed
  `->`, tail self-call) but over the *indexed* path; **not** covered by any
  region (see §9 gap 3).
- `key_pkg_rows/3`, `resolver.pl:416-419` — same shape as `key_dep_rows/3`
  minus the per-element dependency lookup; **not** covered by any region.
- `tree_lookup/3`, `resolver.pl:451-458` — binary-search-tree descent;
  `compare/3` picks exactly one of two tail self-calls (`L` or `R`); **not**
  covered by any region despite sitting directly behind every indexed
  `matching_versions_in/4` / `collect_deps/4` call.
- `alias_lookup/3` (`resolver.pl:298-303`), `selected_ver/3`
  (`resolver.pl:317-321`), `active_member/4` (`resolver.pl:545-551`),
  `seen_name/2` (`resolver.pl:799-803`), `hold_reason/3`
  (`resolver.pl:942-946`), `scan_base_holds/3` (`resolver.pl:924-936`),
  `long_enough/2` (`resolver.pl:401-406`) — the F11 accessor-bank family
  named in the throughput plan §9.
- `inst_walk/6`, `resolver.pl:889-899` — the transitive-closure worklist
  variant (§8); tail-recursive at the Prolog-clause level even though it
  implements a BFS.

### 3. Why it's deterministic
Exactly one clause head matches a given call shape (a `[]`/`[_|_]` split, or
a `compare/3` trichotomy), and the committed `->` inside the body means the
per-element decision has at most one outcome. No choice point is created by
the walk itself, so at-most-one-solution holds by structural induction on the
list/tree being consumed.

### 4. Recognizable marker
`Body = (Cond -> Then ; Recurse)` (or `Recurse` unconditionally in the base
case) where `Recurse` is the ONLY call to the predicate itself and is the
clause's syntactically last goal, `Cond`/`Then` contain no call to the
predicate, and the output argument(s) are either a fresh unbound var filled by
`Recurse` (difference-list build) or a value computed before `Recurse` and
passed as its argument (accumulator-pass). This is exactly `wam_rust_region4_build_tree_ok`'s
sibling shape for the single-self-call case, and it is the mprolog **F11**
gate (`independ_head` + "single recursive clause, tail position").

### 5. Native emission shape
`loop { }`: bind the head, run the per-element body, on the tail self-call
rebind the argument registers/locals and `continue`. Regions 1/2/3a additionally
fuse a **called-but-not-self** deterministic callee (`dep_to_req`, `satisfies`/
`version_lt`) inline as a private native copy (P1), so the loop body is not
just the walk but the walk-plus-decision.

### 6. Status
**Region 1/2/3a: banked ON**, each behind its own flag and its own real A/B
(`docs/reports/wam_rust_stage2_region{1,2}.md`, region 3a in `region3.md`).
Recognizer: per-region hand-built structural (`=@=`) shape checks
(`rust_region1_applicable/1`, etc.) — **not** a single general recognizer.
**F11 (the plain accessor bank, no fused callee): built but left OFF** —
region 4's report §"Task B" measured it net **+4.2% slower** even under the
P2-style minimal-locals idea, because F11's emission re-dispatches through
`vm.step(&Instruction::...)` (a per-instruction `String`-allocating dispatch)
and still needs a *full* per-iteration register snapshot (the loop rebinds
argument registers, unlike the regions, which never mutate a register) — so
the "P2 minimal snapshot" lever that won for regions 1-4 does not apply to
F11's emission as built. **Gap:** `filter_satisfies/3`, `key_pkg_rows/3`,
`tree_lookup/3` are structurally identical to already-lowered siblings but
have no region of their own — the clearest evidence in this survey that a
general recognizer (§11) would pay for itself over one-hack-per-predicate.

### 7. Soundness notes
G-1 (minimal saved set) and G-2 (determinism guard, no CP/no interpreted code)
are satisfiable exactly as regions 1-4 already prove: the walk reads its
argument registers into locals and mutates none of them, so the P2 three-scalar
snapshot suffices. G-3 (cut across tiers) is vacuous when the committed `->`
compiles to a host `if`/`else` with no real choice point. G-4 (at-most-one-solution)
rests on the `[]`/`[_|_]` (or `compare/3`) exclusivity plus the commit, **not**
on head non-unifiability alone when a fused callee (`dep_to_req`) has
unifiable heads (region 1's report is explicit about this). G-5 is vacuous —
no resume-state CP is ever created. F11's failure is not a soundness gap; it
is a cost-model miss specific to Rust's current F11 emission (§6, above).

---

## 2. Linear recursion (one recursive call per clause, not necessarily tail)

### 1. Definition
Exactly one self-call (or one call into the predicate's mutual-recursion SCC)
per clause body, but — unlike §1 — the self-call is **not** required to be
the clause's last goal. The two shapes actually found in the project:

- **(2a) Tail** (§1 is the fully-worked special case of this).
- **(2b) Non-tail / post-order.** The clause does further work — typically
  building a result that depends on BOTH the current node's own contribution
  AND the recursive call's result — *after* the self-call returns.

### 2. Concrete examples
- **(2a):** every example in §1.
- **(2b):** `topo_one/7`, `resolver.pl:838-850` (clause 2): calls
  `topo_all/7` (its mutual-recursion partner, §5) to process the current
  node's dependency names, and only AFTER that call returns does it prepend
  the current node: `Acc = [Name-Ver|Acc1]`. This is a classic post-order
  graph-traversal accumulate — the node is added to the output *after*, not
  before, its subtree is fully processed (so that dependencies precede
  dependents in the emitted `Layer` list — the whole point of `topo_sort_sel`,
  §8). No single-predicate non-tail example was found in `resolver.pl`
  outside a mutual-recursion SCC; PLAWK's `plawk_i64_expr_ir` binary-expr
  clause (§4) is the project's cleanest non-tail example, but it is
  arity-2 (tree recursion), not arity-1.

### 3. Why it's deterministic
Same clause-exclusivity + commit argument as §1 for the self-call itself; the
"more work after" step (`Acc = [Name-Ver|Acc1]`) is a plain deterministic
unification, so it adds no choice point of its own. Determinism of the whole
clause is therefore inherited from the self-call's determinism, composed with
a deterministic post-step.

### 4. Recognizable marker
Same head-exclusivity + commit test as §1, but WITHOUT requiring the self-call
to be the syntactically last goal — instead require that every goal AFTER the
self-call is provably deterministic (by the same det-lattice check used
elsewhere) and makes no further recursive call. `Self(...), G1, ..., Gn` with
`Gi` det and non-recursive for all `i`.

### 5. Native emission shape
Cannot be a bare `loop{}` (§1's shape) because the post-step needs the
recursive call's *result* before it can run — the natural host form is a
plain (non-tail) native function call/return (the host language's own call
stack does the "resume after" bookkeeping), OR, if bounded depth is required
for stack-safety, an explicit native stack recording "resume point + locals
needed for the post-step" (the same idea as §3/§4's explicit-stack/bounded
recursion, specialized to arity 1).

### 6. Status
**Not lowered.** No region targets `topo_one`; it sits on the `layer_closure/3`
path, which is not part of the B2/B3 differential corpus, so it has not been
measured or prioritized. No general recognizer exists for the "non-tail,
det-after" shape; it would need the compositional det-lattice check described
in §11 (an ordinary tail-recursion recognizer rejects this shape outright,
since the self-call is not last).

### 7. Soundness notes
G-1/G-2 apply as in §1, with the added requirement that the post-step's own
register/local use be included in the liveness computation (the values it
needs — here, `Name`, `Ver`, and the self-call's own output `Acc1` — must
still be live after the self-call returns, which for a native call/return
emission is automatic but for an explicit-stack emission must be an explicit
saved-locals entry, generalizing G-1 to a *frame*, not just an entry
snapshot). G-4 must be proven for the self-call in isolation (as in §1); the
post-step needs no separate G-4 argument because it is a plain deterministic
unification. G-5 is vacuous (no CP is ever left). See also §8, where this
exact predicate is the transitive-closure/mutual-recursion combined example.

---

## 3. Non-tail recursion via explicit accumulator / nested loop

### 1. Definition
A predicate whose OWN recursion is last-call (§1's shape), but which, before
recursing, calls a **helper predicate** that itself walks a variable-length
run of the input and must fully complete before the outer recursion can
continue at the correct resumption point. The pair is non-tail *as a unit*
even though neither individual predicate's own self-call breaks tail form —
the outer loop cannot be a single flat `loop{}` because it needs the inner
loop's "how far did you get" result to know where to resume.

### 2. Concrete examples
- `group_keyed/2` ⊕ `same_key/4`, `resolver.pl:421-433` — region 3b.
  `group_keyed` walks a sorted keyed-row list; per group it calls `same_key`
  to consume the whole run of matching keys, then recurses on
  `same_key`'s leftover (`Rest1`).

### 3. Why it's deterministic
Both predicates individually satisfy §1's determinism argument (clause
exclusivity on `[]`/`[_|_]`, committed `->` on `K2==K`, `==/2` is a pure
structural test that never binds). The pair composes deterministically
because `same_key`'s output (`Rest1`) is a pure function of its input.

### 4. Recognizable marker
Predicate `P`'s recursive clause calls a distinct helper `H` (itself matching
§1's tail-recursion shape) whose LAST output argument becomes part of `P`'s
own recursive call's input, and `H` is called from nowhere else in the module
(so its behavior can be inlined without touching any other call site — the
"shared callee, inline a private copy" choice regions 1/2 already made
explicit for `dep_to_req`/`satisfies`).

### 5. Native emission shape
Two nested native loops (or one loop with an inner index-scanning loop) over
the SAME materialized input: an outer index `i` marks the start of each
group, an inner index `j` scans the run, the outer loop resumes at `i = j`.
No resume-state choice point.

### 6. Status
**Region 3b: banked ON** (`docs/reports/wam_rust_stage2_region3.md`).
Recognizer: hand-built per-predicate-pair (`rust_region3b_group_keyed_ok` +
`rust_region3b_same_key_ok`) — not general.

### 7. Soundness notes
G-1: backward liveness reduces to the argument registers at entry (both
predicates never clobber a register, so the P2 three-scalar snapshot
suffices even for the nested-loop shape). G-2: no CP, no interpreted call
(by construction). G-3: vacuous (no cut crosses a tier boundary — `same_key`'s
`K2==K ->` is a host `if`). G-4: both predicates are independently
at-most-one-solution (region 3's report gives the full argument); the PAIR's
at-most-one-solution is the composition of two at-most-one-solution
predicates, which is at-most-one-solution. G-5: vacuous.

---

## 4. Tree recursion (multiple recursive calls per clause)

### 1. Definition
A clause body makes **two or more** recursive calls to itself (or to fellow
SCC members), and combines their results afterward — a genuine
divide-and-conquer or tree-walk, not last-call in either sub-call.

### 2. Concrete examples
- `build_tree/4`, `resolver.pl:439-448` — region 4. Splits `N` into `NL`/`NR`
  (balanced, differing by at most 1), recurses on each half, then builds
  `Tree = t(L, K, V, R)`.
- **PLAWK, meta-level:** `plawk_i64_expr_ir(Expr, ...)` for a binary
  arithmetic expression, `examples/plawk/codegen/plawk_native_codegen.pl:5540-5554`:
  dispatches on `plawk_i64_binary_expr(Expr, LLVMOp, _, Left, Right)`
  (`add_i64/sub_i64/mul_i64/div_i64/mod_i64`, facts at
  `plawk_native_codegen.pl:4969-4973`, mutually exclusive functors), then
  recursively compiles `Left` and `Right` (two self-calls) and combines their
  IR via `plawk_i64_binary_op_lines/5`. This is a real tree-recursion instance
  in the codebase, but it runs at PLAWK's own **compile time**, under the
  host SWI-Prolog, generating LLVM IR text — it is never itself a candidate
  for WAM lowering (there is no WAM bytecode for it to be lowered from; it
  IS the compiler). It is included here as a definitional example of the
  shape, not a lowering target — see the Status note below.

### 3. Why it's deterministic
`build_tree/4` has one clause; its `->` on `N =:= 0` is an arithmetic test
(never binds, hard commit); both self-calls are at-most-one by induction on
`N` (a strictly decreasing, non-negative integer — ordinary structural
termination, not the Seen-based argument of §8); the head-unification on the
threaded difference-list argument can only fail, never branch. PLAWK's
`plawk_i64_expr_ir` is deterministic because `plawk_i64_binary_expr/5`'s
facts have mutually exclusive functors (region 1/2's G-4 argument, applied to
a fact table) and the whole codegen pipeline is written to be a function
(one `Expr` term always yields exactly one IR string, by construction of the
generator, not proven by any runtime gate — it is host meta-code, not
WAM-compiled code, so the G-1..G-5 gates do not apply to it at all).

### 4. Recognizable marker
Single clause (or clause set with mutually exclusive heads), body contains
`->` (or is unconditional) with **two or more** occurrences of a call to the
predicate itself (or an SCC member) whose arguments are computed from a
provably-terminating split of an input (a numeric halving, or a list split at
a computed index), followed by a combine step using both calls' outputs.

### 5. Native emission shape
A **bounded native recursion** over the materialized input (build_tree walks
the input list once into a `Vec`, then does a small associated Rust-level
recursion of depth O(log N) for a balanced split) — equivalently, an explicit
stack of depth O(log N). Not a flat `loop{}` (unlike §1) because both branches'
results must be combined after both return.

### 6. Status
**Region 4: banked ON** (`docs/reports/wam_rust_stage2_region4.md`, −41% B3
median on top of regions 1-3). Recognizer: `rust_region4_build_tree_ok` /
`rust_region4_is_build_tree/1` — hand-built for this one predicate, not
general (there is no second tree-recursive predicate in `resolver.pl` to
generalize against). PLAWK's `plawk_i64_expr_ir`: **not applicable** — it is
compiler meta-code, never itself compiled through any WAM target; noted here
purely so the taxonomy is honest about what "PLAWK carries more of this
pattern" actually means (the pattern is well-represented in the *tooling*,
not necessarily in code the recognizer will ever see).

### 7. Soundness notes
G-1: backward liveness reduces to the argument registers at entry
(`N`, `Pairs`, `Rest`), forced live by never being clobbered before the two
final unifies. G-2: no CP, no interpreted call, by construction. G-3:
vacuous (the `->` is a host `if`, no cut). G-4: single clause, so clause
selection is trivial; the body's determinism rests on the arithmetic test
being a hard commit and both self-calls being at-most-one by numeric-descent
induction — this is the ONE case in the survey where the termination
argument is the classic structural-decrease kind (§0 dimension 5, first
branch), not the Seen-based kind (§8). G-5: vacuous — the recursion is a
bounded native recursion, never a resume-state CP.

---

## 5. Mutual recursion (predicates calling each other cyclically)

### 1. Definition
Two or more predicates form a call cycle: P calls Q (recursively or
otherwise) and Q calls P (directly or through further members of the cycle),
with no single predicate's own self-call visible in isolation.

### 2. Concrete examples
- **`lookup_held/3` ↔ `item_ver/3`**, `resolver.pl:138-149`.
  `lookup_held([H|T], Name, Ver)` calls `item_ver(H, Name, V0)`; `item_ver`'s
  third clause, `item_ver(layer(_L, Pkgs), Name, V) :- lookup_held(Pkgs, Name, V)`,
  calls back into `lookup_held` on the layer's nested package list. A layer
  entry can itself (in principle) contain further `layer(...)` terms, so the
  cycle is genuine, not just a two-hop chain — though in the corpus, layers
  are not nested in practice.
- **`segs_lt/2` ↔ `segs_lt_1/2`**, `resolver.pl:172-188`, with `order_lt/2`
  (`resolver.pl:190-204`) as a third member reachable from `segs_lt_1` but
  not calling back into `segs_lt`/`segs_lt_1` itself (so the strict SCC is
  `{segs_lt, segs_lt_1}`, with `order_lt` a downstream deterministic callee).
  `segs_lt(A, B)` pads both sides and calls `segs_lt_1`; `segs_lt_1`'s
  recursive branch (`O1==O2, N1=:=N2, segs_lt(T1,T2)`) calls back into
  `segs_lt`. This is the project's cleanest small mutual-recursion example.
- **`topo_all/7` ↔ `topo_one/7`**, `resolver.pl:833-850` — see §8 for the
  full transitive-closure treatment; noted here because it is simultaneously
  the project's clearest **non-tail** mutual-recursion example (the `Seen`
  argument is what makes the cross-predicate recursion terminate).

### 3. Why it's deterministic
Determinism must be proven for the **whole SCC**, not per-predicate: each
predicate's own clause-exclusivity/commit argument (§1's or §2's shape) holds
locally, and by induction over the (finite, well-founded) argument that
decreases across the *whole cycle* — not necessarily within one predicate's
own recursive step — the SCC as a unit has at most one solution for any
ground call. For `segs_lt`/`segs_lt_1` the decreasing measure is the combined
length of the two segment lists (each hop through either predicate consumes
one segment pair or terminates). For `lookup_held`/`item_ver` it is the size
of the (in-practice non-cyclic, but not statically bounded) layer nesting.

### 4. Recognizable marker
Build the call graph among the module's predicates restricted to calls with a
provable determinism-modulo-recursion (§1/§2's per-clause shape, checked
ignoring whether the callee is "self" or "another SCC member"); run Tarjan
SCC decomposition; for every SCC of size ≥ 2, the pattern applies. The
per-predicate local checks (clause exclusivity, commit position) are
UNCHANGED from §1/§2 — mutual recursion is a graph-level classification
layered on top of, not a replacement for, those checks. This is why it is
listed here as a dimension composing with tail/non-tail (§0) rather than a
ninth, disjoint shape.

### 5. Native emission shape
Two working precedents in the codebase, at opposite ends of the difficulty
spectrum:
- **When the whole SCC's recursion is tail-shaped** (as `segs_lt`/`segs_lt_1`
  is): fuse both predicates into ONE native function with a `loop{}` whose
  body is a `match`/`if` over "which predicate's clause am I logically in
  right now" — exactly what region 2 already built (`region_segs_lt` and
  `region_segs_lt_1` are both native functions in the SAME fused region,
  calling each other as plain Rust functions, `docs/reports/wam_rust_stage2_region2.md`
  Files-changed list). No separate "mutual recursion" mechanism was needed —
  P1 (direct native calls) already handles a cycle of plain Rust function
  calls just as well as a chain, because Rust's own call stack (not a WAM
  choice-point stack) carries the recursion, and the whole SCC was already
  proven acyclic-in-solutions (at-most-one) before code generation.
- **When the whole SCC's recursion is non-tail** (`topo_all`/`topo_one`):
  needs an explicit-stack emission generalizing §3/§4 to a cycle of two
  predicates rather than one predicate calling one distinct helper — see §8.

### 6. Status
**`segs_lt`/`segs_lt_1`/`order_lt`: already lowered**, but only as a
*side-effect* of region 2's "3-deep chain" fusion (`docs/reports/wam_rust_stage2_region2.md`
lists `region_segs_lt` and `region_segs_lt_1` in its Files-changed section);
the region 2 report never frames this as "mutual recursion," and the
recognizer that enabled it (`rust_region2_applicable/1`'s eight structural
checks) is a hand-built, named-predicate-list recognizer, not a general
mutual-recursion detector. **`lookup_held`/`item_ver`: not lowered** — no
region targets it; it is cheap (accessor-sized) and cold relative to the
regions' hot-path share, but it is the clearest available demonstration that
today's mechanism finds mutual recursion only when a human names both
predicates in a hand-written shape check. **`topo_all`/`topo_one`: not
lowered** — see §8.

### 7. Soundness notes
G-1/G-2 as in §1, generalized to a frame per SCC member if the SCC's
recursion is non-tail (§2's soundness note). G-3: vacuous when no cut crosses
a tier boundary in any SCC member (true for both examples here). G-4: MUST be
proven for the SCC as a whole — region 2's report proves it for
`segs_lt`/`segs_lt_1`/`order_lt` together ("Transitively the fused region has
at most one solution"), which is the right unit of proof; a per-predicate G-4
check in isolation is insufficient for a genuine cycle (a predicate that
looks at-most-one "assuming its callee is," where the callee's own proof
circularly assumes the caller, needs the induction to be over the SCC's
well-founded measure, not over one predicate's argument alone). G-5: vacuous
for the tail-shaped SCC (`segs_lt`); for the non-tail SCC (`topo_all`/
`topo_one`) it stays vacuous too, precisely BECAUSE the closure/reachability
argument (§8) shows no choice point is ever created — the mutual-recursion
dimension does not, by itself, introduce a new soundness obligation beyond
"prove G-4 over the SCC, not the predicate."

---

## 6. Committed-choice recursion

### 1. Definition
A recursion whose body performs (or calls a predicate that performs) a
genuinely nondeterministic sub-search, but the nondeterminism is **committed**
— via `once/1`, a cut, or the condition of an `->` — strictly BEFORE the
recursive step, so backtracking can never re-enter an earlier iteration's
choice. One solution overall, despite the nondet-looking ingredient.

### 2. Concrete examples
- `dep_breaks/5`, `resolver.pl:1008-1014` — `dep_breaks_need/4`
  (`resolver.pl:1016-1021`, nondet: `member(dep(D,COut),Alts)` inside) sits
  inside the `->` CONDITION; the tail self-call is confined to the
  else-branch. Fully analyzed in
  `docs/reports/wam_rust_nondet_driver_classification.md` §1 — already
  deterministic-by-construction, no `resolver.pl` change needed.
- `close_moving/3` ⊕ `first_broken/4` ⊕ `pick_repair/4`,
  `resolver.pl:982-1032` — a fixpoint loop for `upgrade_set`/`safe_upgrade`:
  each iteration finds the first broken hold (`first_broken`, itself calling
  the nondet `dep_breaks_moving`/`dep_breaks` chain, consumed via its own
  `->` commit at `resolver.pl:999-1001`) and, if broken, calls
  `pick_repair(Cat, N, Acc, NewV) -> close_moving([N-NewV|Acc], Result) ; ...`
  — `pick_repair` is nondet (`candidates_high_first` backtracks over
  versions), but it sits inside an `->` condition, committing to the FIRST
  repairing version before the tail self-call. Same shape as `dep_breaks/5`,
  not previously classified anywhere in the project's documents.
- **PLAWK's `process_all/4`**, `examples/plawk/core/plawk_core.pl:38-47`
  (quoted in full in §0's framing and in the escalation below) — the
  project's own README calls it "a deterministic Prolog core"
  (`examples/plawk/README.md:16`). The per-record `call(Handler, Item, State1, State2, Continue)`
  is a meta-call, not a lexically nondet goal, but it plays the same role:
  its result is consumed and immediately tested (`Continue == yes -> ... ; ...`)
  before the tail self-call, so no alternative handler outcome survives past
  one record. See §6's meta-call preconditions below — this example is
  committed-choice PLUS meta-call indirection, composed.

### 3. Why it's deterministic
The committing operator (`->`, `once/1`, or a cut) discards every alternative
of the contained nondet goal before control ever reaches the recursive call;
by induction, if no earlier iteration left a choice point, no later one does
either, so the whole recursion has at most one solution.

### 4. Recognizable marker
`Body = ( Cond -> Then ; Self(...) )` (§1's marker) generalized to
**not require `Cond` itself to be det** — only that `Cond` is evaluated under
`->` (or `once/1`, or immediately followed by a cut) and that the ONLY
recursive self-call is confined to a branch that runs AFTER the commit
(`dep_breaks/5`'s else-branch; `close_moving`'s then-branch after
`pick_repair`'s own commit). This is exactly the generalization recommended
in `wam_rust_nondet_driver_classification.md` §1 ("accept an arbitrary
committing condition — including one that calls a nondeterministic
predicate — provided (i) the only recursive self-call is the else-branch
tail, and (ii) the condition is committed by `->`/`once`/cut"), extended
here to also cover the symmetric case where the commit is in the *then*-branch
before recursing (`close_moving`).

**Meta-call extension (for `process_all/4`-shaped predicates).** When the
"contained nondet-looking goal" is a `call(G, ...)` rather than a lexically
fixed predicate, the marker additionally requires — this is a firm ruling,
not a hedge, but it is firm only WITH all three preconditions:

1. **Closed-world target set.** The complete set of concrete predicates `G`
   can resolve to at this call site must be statically known and PROVABLY
   closed — the same whole-program closure the existing meta-call dispatch
   table already enumerates (`examples/plawk/probes/generate_meta_call_probe.pl`),
   provided nothing can add a target the table does not see: no
   `assert`/`retract` of a meta-called predicate, and no runtime-constructed
   goal (`=..`, `functor/3` on non-compile-time data) reaching the call site.
   Open-world at this call site ⇒ decline to the interpreter.
2. **Closure/partial-application resolution.** `call(Handler, ...)` where
   `Handler` is itself a partial application must be resolved to the
   underlying predicate at its FULL effective arity (captured arguments
   included), and the check must recurse through any further meta-call
   nested inside that target. Captured *data* is harmless; a captured *goal*
   that is itself later called must be followed.
3. **Mode-sensitivity.** At-most-one-solution is mode-dependent (`member/2`
   is semidet with a bound first argument and nondet with an unbound one) —
   so each concrete target must be proven at-most-one-solution UNDER THE
   ARGUMENT INSTANTIATION (mode) actually used at this call site, not
   abstractly. For `process_all/4`, `Reader`/`Handler` are always called in a
   fixed state-threading mode (`Item` produced, `State0→State1` consumed), so
   the obligation is "every concrete `Reader`/`Handler` in the closure is det
   in THAT mode." An undeterminable call-site mode, or any concrete target
   that is nondet in that mode, ⇒ decline.

The existing meta-call dispatch table gives DISPATCH (which concrete function
to jump to), not a determinism PROOF — the mode-sensitive per-target check
above is the new layer needed on top of it before a meta-call can be folded
into this class.

### 5. Native emission shape
Identical to §1's `loop{}` — a plain native loop with no resume-state choice
point — PROVIDED the recognizer's marker holds; the committed nondet call (or
meta-call) compiles to a plain conditional/native-dispatch, exactly as
`dep_to_req`'s cut became a Rust `if`/`else` in region 1. For the meta-call
case, the loop body's per-iteration step becomes a native `match`/jump-table
over the closed set of concrete targets (the meta-call dispatch table
mechanism already built) with each arm's body being that target's OWN
(separately proven, separately lowered) native function.

### 6. Status
**`dep_breaks/5`: classified, recognizer widening recommended, not yet
built** (`wam_rust_nondet_driver_classification.md` §1/§5, recommendation 1 —
"widen the deterministic single-clause `->`-commit recognizer (region-4
family) to accept a committing condition that contains a nondet sub-goal ...
This is a recognizer widening, not new runtime machinery"). **`close_moving`
fixpoint chain: not previously classified anywhere** — a genuine new finding
of this survey; low priority because it sits on the `safe_upgrade`/
`upgrade_set` path, not the B2/B3 differential hot path. **PLAWK's
`process_all/4`: not lowered** — the probe README
(`examples/plawk/probes/README.md`) states plainly that "All listed PLAWK
helper predicates currently compile through the WAM fallback path," i.e. it
runs interpreted today, even though its shape (once the meta-call extension
above is granted) is exactly this class. This is the survey's highest-value
PLAWK-side gap (§10).

### 7. Soundness notes
G-1/G-2 as in §1 (the region-4-family argument already covers this shape for
the non-meta-call case). G-3: vacuous when the contained nondet/meta call has
no cut that must cross a tier boundary — true for `dep_breaks_need` and
`pick_repair` (their own cuts, if any, are internal to the committed goal and
never need to prune anything above the commit); for a meta-called target with
its own cut, that cut must stay internal to the target's own native body
(never escape past the call boundary), which is guaranteed once the target
itself is independently proven G-3-clean. G-4: the crux of this whole class —
"determinism ≠ non-unifiable heads" (Kimi K2 gate) generalizes cleanly to
"determinism of the committed goal under its call-site mode," which is
exactly the mode-sensitivity precondition above, stated in full generality.
G-5: vacuous — no resume-state CP is ever created, by construction of the
commit.

---

## 7. Aggregate-then-iterate

### 1. Definition
A nondeterministic sub-search is bounded inside `findall`/`bagof`/`setof`/
`aggregate_all`, which is itself deterministic (it fully explores the goal and
returns exactly one list); the useful work downstream is then a deterministic
fold/iteration over that list. Backtracking is contained entirely at the
aggregate boundary.

### 2. Concrete examples
- `virtual_provider_ceilings/4`, `resolver.pl:767-773` — internally
  `findall(blocked(...), (provides_for(...), base_ver(...), \+ provide_satisfies(...)), Reasons)`;
  the caller (`blocked_from/4` clause 3, `blocked_acc/5`) then only tests
  `Reasons \== []` and, if so, uses it as-is — a trivial (identity) fold, but
  the pattern is present.
- **PLAWK, throughout `plawk_native_codegen.pl`** (53 occurrences of
  `findall`/`bagof`/`setof`/`aggregate_all` total; the clean
  `findall(X, member(X,L), Xs)`-family idiom the throughput plan §6b cites
  by name): e.g. `plawk_dyncall_at_support_ir/3`,
  `plawk_native_codegen.pl:1123-1135`, `findall(S, (member(N, Arities),
  plawk_dyncall_at_shim_off_ir(N, S)), Shims)` — aggregate one IR snippet per
  arity, then join them; `plawk_prolog_block_preds/2`,
  `plawk_native_codegen.pl:26-44`, `findall(user:PI, member(PI, PIs), Preds)`.
  As with §4's tree-recursion PLAWK example, these run at PLAWK's own compile
  time (they generate code) — not themselves WAM-lowering targets — but they
  are the concrete, in-project instance of the idiom the recognizer's marker
  is meant to describe.
- `topo_one/7`'s inner step, `resolver.pl:842`,
  `findall(D, follow_dep_name(Cat, Name, Ver, Sel, D), Ds0), sort(Ds0, Ds)`
  — an aggregate-then-iterate step NESTED inside the mutual-recursion/
  transitive-closure shape of §5/§8; see §8 for the full treatment and the
  "reproduce the reduction operator verbatim" rule.

### 3. Why it's deterministic
`findall/3` (and `bagof`/`setof`/`aggregate_all`) is itself a deterministic
builtin: for any goal and template, it produces exactly one list, in one
fixed (generation) order, regardless of how nondeterministic the goal is
internally. The fold/iteration that follows is then just an ordinary
deterministic recursion over that list (§1's shape), so the composite has at
most one solution.

### 4. Recognizable marker
At the call site: `findall(Template, Goal, List)` (or `bagof`/`setof`/
`aggregate_all`) immediately followed (in the same clause, or in the
predicate that consumes `List`) by a deterministic fold over `List`
(`foldl/4`, or a §1-shaped member-driven recursion whose per-element choice
is itself committed). The aggregate call is the recognizable boundary; goal
`Goal` itself needs NO determinism proof at all (this is what makes the class
useful — it launders arbitrary internal nondeterminism for free).

### 5. Native emission shape
The aggregate call itself stays an interpreted builtin (or is lowered
separately, later, as its own project); what lowers deterministically is the
FOLD over its result — a native loop (§1's shape) over the (already fully
materialized) list.

### 6. Status
**No target lowers this today.** Per `wam_rust_nondet_driver_classification.md`
§3: "No target currently lowers a `findall`-bounded generator as a native
all-solutions region; `findall/3` stays an interpreted builtin (correctly).
Realizing the win requires two new recognizer/emitter cases ... Until both
exist, the marker documents intent but the aggregate is interpreted." This
survey confirms that status is still current and extends the marker (above)
to be genuinely target-agnostic (it was previously stated only in the context
of a specific proposed `blocked_from/4` rewrite, which the same document
recommends NOT undertaking near-term — see §9 gap discussion).

### 7. Soundness notes
G-4 is, in a real sense, moot for the aggregate call itself — `findall`'s
own semantics already guarantee at-most-one-solution regardless of `Goal`'s
internal nondeterminism, so there is nothing to prove about `Goal`. The
soundness weight moves entirely onto (a) the aggregate operator being
reproduced VERBATIM if the fold is ever lowered to run the search itself
natively rather than calling the interpreted `findall` (a native
re-implementation of `Goal`'s search must produce the identical list, in the
identical order — this is exactly the "high risk" the `blocked_from/4`
class-(b) rewrite proposal flags, and why it is recommended against as a
near-term change), and (b) the fold afterward being independently proven
det (§1's G-1/G-2/G-4). G-1/G-2/G-3/G-5 for the fold itself are exactly §1's.

---

## 8. Transitive closures (reachability over a relation)

### 1. Definition
A recursive walk over a graph/relation (dependency edges, install closure,
layer nesting) that computes a *set* of reachable nodes (or a derived
ordering over them), terminating because a **finite domain + visited-set
("Seen") accumulator** guarantees no node is processed twice and the domain
is finite — NOT because a first argument structurally shrinks every step (the
termination argument is genuinely different from every other pattern in this
document, and needs its own soundness note, §0 dimension 5).

### 2. Concrete examples — two sub-shapes
- **(8a) Worklist / BFS, tail-recursive, single predicate:**
  `inst_walk/6`, `resolver.pl:889-899` (via `inst_closure_names/5`,
  `resolver.pl:886-887`, and reused by `needed_names/4`,
  `resolver.pl:901-904`). An agenda list (`[Name-Ver|Rest]`) is threaded
  forward; each step either skips an already-`Seen` node (tail self-call,
  unchanged `Seen`/`Acc`) or aggregates the node's children
  (`findall(D-DV, (follow_dep_name(...), member(D-DV,Inst)), Kids)`),
  appends them to the agenda (`append(Kids, Rest, More)`), marks the node
  `Seen`, and recurses (tail) with the extended agenda. This is §1's
  tail-recursion shape PLUS an aggregate-then-iterate step (§7) PLUS the
  Seen-based termination argument.
- **(8b) Post-order DFS, mutual recursion:** `topo_sort_sel/3`
  (`resolver.pl:822-827`) driving `topo_all/7` ↔ `topo_one/7`
  (`resolver.pl:833-850`) — see §2's and §5's entries for this same example.
  `topo_one` aggregates a node's not-yet-seen dependency names
  (`findall` + `sort`, §7's shape) and recurses (via `topo_all`) into all of
  them BEFORE prepending the current node to the output — a post-order
  topological sort. Non-tail, mutual-recursion, Seen-based termination, all
  at once.

### 3. Why it's deterministic
The per-step nondeterminism (`follow_dep_name/5`, itself nondet via
`member/2` over `Depends`/`Provides` rows) is fully contained inside a
`findall` (§7) before it can affect the walk — so each step is a
deterministic function of (current agenda/node, Seen). Termination: `Seen`
only grows, is bounded by the (finite) set of package names in the catalog,
and every recursive step either shrinks the agenda without growing `Seen`
(the "already seen, skip" branch) or grows `Seen` by exactly one name (the
"process it" branch) — so the combined measure `|AllNames| - |Seen|`
strictly decreases or the agenda strictly shrinks, on every step, guaranteeing
termination in at most `|AllNames|` "process" steps.

### 4. Recognizable marker
(a) A per-step nondeterministic sub-search fully closed inside `findall`/
`bagof`/`setof` before the recursive step (§7's marker), AND (b) a
Seen/visited accumulator argument that is checked (membership test,
committed) before processing a node and extended (by exactly the current
node) after, over a domain the predicate's own call structure guarantees is
finite (bounded by an input catalog/graph size, not by an unbounded
generative process). Tail-vs-non-tail (§1 vs §2) and single-predicate-vs-SCC
(§5) are ORTHOGONAL to this marker, not part of it — §8 is the composition of
those dimensions with "Seen-bounded, aggregate-per-step" termination, not a
disjoint fifth shape.

**Reduction-operator-verbatim rule (firm).** The lowering's job is to
reproduce the source predicate's exact result, not to choose a "better"
aggregate/dedup strategy. If the source uses `sort/2` after the per-step
`findall`, the emitted native code must reproduce a SORTED, DEDUPLICATED set
— not substitute `list_to_set` (which preserves first-occurrence order
instead) or raw generation order. If the source uses `list_to_set`, reproduce
first-occurrence order; if the source uses a bare `findall` with no dedup
step, reproduce generation order, duplicates included. Order/multiplicity
preservation is guaranteed by construction ONLY if the reduction operator is
copied verbatim — never inferred or "improved." (For the two examples in
this survey, both use `sort/2` and both only observe the resulting *set* of
reachable names downstream, so there is no first-occurrence-provenance
question to resolve for them specifically — but the rule above is what makes
that true generally, not an assumption specific to these two predicates.)

### 5. Native emission shape
(8a): a native `loop{}` over an explicit agenda `Vec`/queue (§1's shape,
specialized: the "output" is the growing `Seen` set/`Acc` list, and the
per-step aggregate is a native inner loop over the relevant edges, reusing
whatever native form the per-step `findall`'s goal would take if lowered
under §7's rule — or, short of that, calling the interpreted `findall`
per step and only fusing the agenda-walk itself).
(8b): an explicit native stack (§3/§4's shape, generalized to a
mutual-recursion SCC): each stack frame records "which node, and how far
along its not-yet-processed children list," so the post-order prepend can
happen when a frame's children are exhausted — structurally the same
explicit-stack transform `group_keyed`/`same_key` (§3) already validated,
extended from one helper predicate to a 2-member SCC.

### 6. Status
**Not lowered; no recognizer exists for either sub-shape.** Neither
`inst_walk` nor `topo_all`/`topo_one` is on the B2/B3 differential's hot path
(`layer_closure/3` and `removal_orphans/3` are not part of the measured
`resolve`/`resolve_layered` corpus), so this is a correctness/generality gap
rather than a measured-throughput gap today — but it is the pattern most
likely to recur broadly outside the resolver (any project computing a
reachability closure, dependency graph, or topological order will hit this
shape), which is exactly the kind of pattern §6c of the throughput plan asks
this survey to surface even when it is not yet hot in this one corpus.

### 7. Soundness notes
G-1/G-2: as in §1/§2/§3, generalized to whatever frame shape the emission
needs (agenda entry for 8a; explicit stack frame for 8b). G-3: vacuous when
no cut crosses a tier boundary (true for both examples — `topo_one`'s
`member(Name,Seen), !` commit is internal, never needing to prune anything
above itself). G-4: rests on the Seen-bounded-termination argument above,
which is NOT the same argument as G-4's usual "clause exclusivity + commit"
— it is an INDUCTIVE argument over the finite domain, and a recognizer must
verify the domain-finiteness precondition explicitly (the Seen argument's
possible values are drawn from a statically-bounded source — here, the
catalog's own `Packages`/`Depends` — not from an open-ended generative
process) rather than assuming any Seen-accumulator pattern is automatically
safe. G-5: vacuous for both sub-shapes (no resume-state CP is ever created;
8b's explicit stack is a plain data structure, not a WAM choice point).

---

## 9. Summary table

| # | Pattern | Marker (target-agnostic) | Native emission | Status |
|---|---|---|---|---|
| 1 | Tail recursion | `(Cond -> Then ; Self)`, `Self` last & only self-call, `Cond`/`Then` non-recursive | `loop{}` | Regions 1/2/3a **ON**; F11 accessor bank built, measured net-negative, **OFF**; siblings `filter_satisfies`/`key_pkg_rows`/`tree_lookup` **gap** |
| 2a | Linear recursion, tail | = pattern 1 | = pattern 1 | = pattern 1 |
| 2b | Linear recursion, non-tail (post-order) | `Self(...), G1..Gn` with every `Gi` det & non-recursive | native call/return, or explicit resume-frame if depth-bounded needed | **Gap** — no recognizer (`topo_one`) |
| 3 | Non-tail via explicit accumulator (helper-consumes-a-run) | `P`'s recursive clause calls helper `H` (itself pattern-1-shaped, sole caller) whose output feeds `P`'s own recursive call | two nested native loops over materialized input | Region 3b **ON** (hand-built recognizer, not general) |
| 4 | Tree recursion | ≥2 self-calls per clause from a provably-terminating split, combined after both return | bounded native recursion / explicit stack, depth O(log N) or input-bounded | Region 4 **ON** (`build_tree`, hand-built); PLAWK's `plawk_i64_expr_ir` is compiler meta-code, **N/A** as a lowering target |
| 5 | Mutual recursion | SCC (Tarjan) over calls satisfying 1/2/3/4's local shape; G-4 proven over the SCC, not per-predicate | tail-shaped SCC → one fused native function, cycle as plain host calls (region 2's `segs_lt`/`segs_lt_1`); non-tail SCC → explicit stack over the SCC (pattern 8b) | `segs_lt`/`segs_lt_1`/`order_lt` **already ON** (side-effect of region 2, unlabeled); `lookup_held`/`item_ver` **gap**; `topo_all`/`topo_one` **gap** (= pattern 8b) |
| 6 | Committed-choice recursion | `(Cond -> Then ; Self)` with `Cond` (or `Then`, symmetric) containing an uncommitted-looking nondet/meta call that is itself committed by the `->`/`once`/cut before any recursive step | `loop{}`, contained call → host `if`/`match` (or native dispatch table for a meta-call) | `dep_breaks/5` **classified, recognizer widening recommended, not built**; `close_moving` fixpoint chain **new gap, this survey**; PLAWK `process_all/4` **gap, needs 3-precondition meta-call extension** |
| 7 | Aggregate-then-iterate | `findall`/`bagof`/`setof`/`aggregate_all` boundary, followed by a deterministic fold over the result | aggregate stays interpreted; fold lowers as pattern 1 | **No target implements this yet** (per nondet-driver-classification doc); `virtual_provider_ceilings` and PLAWK's `findall(X,member(X,L),Xs)` idioms are the in-project instances of the marker |
| 8a | Transitive closure — worklist/BFS, tail | pattern 1 + pattern 7 (per-step) + finite-domain Seen-accumulator termination | native loop over explicit agenda `Vec` | **Gap** (`inst_walk`) |
| 8b | Transitive closure — post-order DFS, mutual | pattern 2b + pattern 5 + pattern 7 (per-step) + finite-domain Seen-accumulator termination | explicit stack over the SCC | **Gap** (`topo_all`/`topo_one`) |

---

## 10. Gaps, ranked by value

Ranking criterion: (i) measured or plausible hot-path share in the resolver's
B2/B3 differential or PLAWK's core loop, (ii) how much a general recognizer
(vs. one more hand-built region) would leverage across MULTIPLE gaps at once,
(iii) risk.

1. **Committed-choice recognizer widening for `dep_breaks/5`** — already
   analyzed and explicitly recommended
   (`wam_rust_nondet_driver_classification.md` §5, recommendation 1) as
   "recognizer-only... lowest risk, clear win on its dispatch share." No new
   soundness argument needed beyond what that document already established;
   this survey's contribution is showing `close_moving`'s fixpoint chain is
   the SAME shape (a second beneficiary of the same widening) and that
   PLAWK's `process_all/4` is a THIRD beneficiary once the meta-call
   extension (§6) is added — i.e., building this one general recognizer
   pays for three gaps, not one.
2. **PLAWK's `process_all/4` (committed-choice + meta-call)** — highest
   PLAWK-specific value: it is the outer driver loop of every PLAWK program
   compiled through the Phase-0 text-record path (the loop/meta-call probes
   exist specifically to validate it compiles through WAM/LLVM at all), and
   it is confirmed still running the "WAM fallback path" (interpreted) today.
   Requires the 3-precondition meta-call extension in §6 — the highest-risk
   item in this ranking because it is new machinery (whole-program
   closed-world target enumeration + mode-sensitive per-target determinism
   check), not a pure recognizer widening.
3. **The "sibling gap" family** (`filter_satisfies`/`key_pkg_rows`/
   `tree_lookup` next to already-lowered `matching_versions`/`key_dep_rows`/
   the indexed lookup path) — individually cheap, but structurally identical
   to shapes already proven safe and profitable; the clearest ROI argument
   for building the GENERAL pattern-1 recognizer (§11) instead of a fourth,
   fifth, sixth hand-written region. Low risk (pattern 1 is the most-validated
   shape in the whole survey), moderate-to-low measured value per predicate,
   but the recognizer built to close this gap is reusable for any future
   pattern-1 predicate the resolver or another project adds — the payoff is
   in avoiding future one-off region work, not just these three predicates.
4. **Mutual recursion as an explicit, general graph-level classification**
   (§5) — `segs_lt`/`segs_lt_1` proves the MECHANISM (fused native functions
   calling each other) already works when a human hand-identifies the SCC;
   the gap is purely the RECOGNIZER (Tarjan SCC + per-member local check),
   not new runtime machinery. `lookup_held`/`item_ver` would be closed for
   free by this recognizer once built (low individual value, but zero marginal
   cost once the SCC detector exists for `segs_lt`-class cases).
5. **Aggregate-then-iterate as a general call-site pattern** — explicitly
   flagged in the nondet-driver-classification doc as needing genuinely NEW
   emitter machinery (lowering a `findall`-bounded generator's search itself,
   not just the fold after it) to realize any win; the `blocked_from/4`
   subsystem-level rewrite that would benefit most is explicitly flagged
   HIGH RISK and NOT recommended near-term. Rank this below the
   recognizer-only items because it requires new machinery with real
   order/multiplicity risk, for a benefit (`blocked_from/4`, ~4.9% of B2
   shared among three drivers) that is smaller than any item above.
6. **Transitive closure (8a/8b)** — not on any measured hot path in THIS
   corpus (`layer_closure`/`removal_orphans` are cold relative to
   `resolve`/`resolve_layered`), so it ranks last by measured value here —
   but it is flagged as the pattern most likely to matter for OTHER projects
   built on this recognizer (any reachability/closure computation), which is
   why §6c of the throughput plan asks for it to be enumerated even though it
   is not hot today. Recommend building its recognizer opportunistically
   alongside #4 (mutual recursion) rather than as a standalone priority.

## 11. Recommendation: a general `deterministic_recursion_class/2` recognizer

Design as a **compositional pipeline**, not a flat pattern-matcher over the
eight named shapes — §0's dimensions ARE the pipeline's stages, and every
region already built (1-4) is reproducible as one path through it:

```
deterministic_recursion_class(+PredIndicators, -Class)
```

where `Class` is one of `tail_loop/2`, `nontail_linear/3`, `explicit_helper/3`,
`tree/3`, `mutual/2` (wrapping any of the above over an SCC), `committed/3`,
`aggregate_fold/3`, `closure/4` (wrapping `tail_loop` or `mutual` with a
Seen-termination witness) — or `decline(Reason)`.

**Pipeline stages** (each stage either narrows the classification or declines
outright — never guesses):

1. **Call-graph + SCC.** Build the "provably-det-modulo-recursion" call graph
   restricted to this predicate's module-local callees; Tarjan-decompose it.
   An SCC of size 1 is the ordinary single-predicate case (§1-4); size ≥2
   triggers the `mutual/2` wrapper (§5) and requires the SUBSEQUENT stages'
   determinism proof to be over the whole SCC, not one member.
2. **Meta-call resolution.** For every `call(G, ...)` reachable from the
   predicate under analysis, require (in order): (a) a provably closed-world
   target set (no `assert`/`retract` of a meta-called predicate, no
   runtime-constructed goal reaching the site) — else decline; (b) resolve
   partial applications to their full effective arity and recurse this
   resolution through any further nested meta-call; (c) defer the
   determinism check on each resolved target to stage 5, under the SPECIFIC
   mode the call site uses — never check a meta-called target "in the
   abstract."
3. **Self-call arity + position, per clause.** Count recursive/SCC calls per
   clause body; classify each clause as 0 (base), 1-tail, 1-non-tail, or N
   (tree). A predicate with a mix across clauses takes the LEAST restrictive
   emission its worst clause requires (a tree-shaped clause anywhere forces
   the tree/explicit-stack emission for the whole predicate).
4. **Commit / aggregate boundary detection.** Scan each clause for `->`,
   `once/1`, a cut, or a `findall`/`bagof`/`setof`/`aggregate_all` wrapping a
   goal that is NOT independently provable det. If found, and the ONLY
   recursive call(s) are confined to run after that boundary (or, for the
   aggregate case, the boundary's result is consumed by a deterministic fold
   — pattern 7): classify as `committed/3` or fold the aggregate result into
   whatever pattern 1-5 shape the surrounding recursion otherwise has. If a
   contained nondet/meta goal is NOT committed/aggregated before a recursive
   step reachable from it: **decline** (class-(c), per the
   nondet-driver-classification doc's own finding for `pick/7` and
   `blocked_from/4` — this is not a recognizer gap, it is the correct,
   permanent answer for genuinely alternative-exposing predicates).
5. **Whole-predicate/SCC at-most-one-solution proof.** The generalized G-4:
   NOT "pairwise non-unifiable heads" (insufficient, per the Kimi K2 review)
   but a transitive proof combining (a) clause-head exclusivity, (b) each
   committed/aggregated sub-goal's determinism UNDER ITS CALL-SITE MODE
   (stage 2's deferred obligation, resolved here), and (c) — for the SCC
   case — induction over a well-founded measure spanning the whole cycle,
   not one member. Memoize this as a fact per predicate/SCC (the same
   greatest-fixpoint idea the existing cp-clean restriction already uses),
   so a predicate calling an already-proven-det callee does not re-derive
   the proof.
6. **Termination-argument classification.** Distinguish (a) structural
   decrease of an explicit argument (lists/integers — patterns 1-4) from
   (b) a finite-domain-plus-Seen-accumulator argument (pattern 8) — the
   latter requires an EXPLICIT domain-finiteness check (the Seen argument's
   possible values are drawn from a statically-bounded source, such as an
   input catalog), not an assumption that any "accumulator that only grows"
   pattern is automatically safe.
7. **Emission selection**, purely mechanical given stages 1-6's classification:
   tail+SCC-size-1 → `loop{}` (pattern 1); non-tail+SCC-size-1 →
   native call/return or explicit resume-frame (pattern 2b); tail/non-tail
   with a helper-consumes-a-run → nested native loops (pattern 3); tree →
   bounded native recursion / explicit stack (pattern 4); any SCC-size-≥2 →
   fused native functions calling each other directly if tail-shaped, else an
   explicit stack over the SCC (pattern 5, reusing pattern 3/4's mechanism);
   committed → same as the underlying shape, contained call becomes a host
   `if`/`match` or (meta-call case) a native dispatch table over the closed
   target set; aggregate boundary → aggregate call stays interpreted, fold
   lowers per its own shape; closure (8a/8b) → agenda loop or explicit stack,
   with the REDUCTION-OPERATOR-VERBATIM rule (§8.4) governing how the
   aggregated/Seen-deduplicated result is materialized.
8. **Runtime decline guard, always.** Regardless of the compile-time
   classification, every emission keeps the region 1-4 discipline: a runtime
   post-call choice-point-depth assertion (the generalized G-2 guard) and a
   minimal-locals rollback (P2) on any off-shape input detected at runtime —
   never a silent wrong commit. This is what makes "decline to interpreter"
   always safe even if a compile-time proof turns out to have a gap the
   review process missed.

This design is target-agnostic by construction: every stage operates on the
WAM/Prolog-clause-level shape, never on Rust/C++/Go specifics — exactly the
"shared front-end, per-target back-end" split that T2/T5/T6 already
established in `WAM_LOWERING_TAXONOMY_AND_MATRIX.md`. The per-target work
that remains is ONLY stage 7's emission code generation; stages 1-6 are one
Prolog analysis shared by every hybrid WAM target.

## Attribution

The loop shape (F11) and the resumable-choice-point shape (F3) referenced
throughout are ideas adopted from Kenichi Sasagawa's M-Prolog / N-Prolog
(`https://github.com/sasagawa888/mprolog`, Modified BSD), never code, per
`MPROLOG_MINING_NOTES.md`. The committed-choice and aggregate-then-iterate
transforms are standard deterministic-lowering patterns, also used by this
project's PLAWK native codegen, and are not mprolog-specific.
