<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (@s243a) -->

# uw-resolve C WAM smoke lane

Compile the frozen P3 package resolver (`../resolver.pl`) through
`write_wam_c_project/3` and compare **grounded** `resolve/3` answers
against SWI for a named smoke slice. This is **not** full corpus parity.

The C runtime header is copied from
`src/unifyweaver/targets/wam_c_runtime/wam_runtime.h`. Kernels are
suppressed (`no_kernels(true)`). Lowered helpers stay off
(`lowered_helpers(false)`).

## Reverse Implementation Checkpoint (2026-09-09)

**WIP, not merge-ready.** The initial reverse suite completed successfully, but
the latest strengthened revision is not independently verified. Its last gated
execution failed and the following retry reached the repeated-command escalation
threshold. The worker's final success claim is not sufficient evidence to override
those events. Do not treat the current 31-case suite as a confirmed pass.

Independent source review also identified pending corrections: `sort/2` must
unwind partial output bindings on failed unification; transient failed-query heap
growth needs regression coverage; and the resolver smoke driver must retain the
output heap handle instead of reading a potentially overwritten argument register.
Generation must also fail on SWI load errors without accepting stale generated
files, and the smoke wrong-answer control must use a successful case rather than
an already-blocked case. These are pending review findings, not completed fixes.
Re-run fresh compiled tests through the confined gate after these corrections.

The conservative forward `reverse/2` subset has a behavioral test suite
in `tests/test_wam_c_reverse.pl` (12 ground comparisons against SWI, 19 token/property
checks including prebound matching/mismatching, heap growth to 1024 elements with buffer
reallocation, input preservation, shared variable identity, distinct variable preservation,
cell-level independent variable binding, compound terms with shared variables, positive
binding control, non-vacuous backtracking rollback, positive mismatch control, unifier
mismatch rollback, direct C unifier trail unwinding, caller continuation, and unsupported
shape checks for unbound, cyclic, open, non-list, and improper lists).

Key test-evidence enhancements:
- **Non-vacuous backtracking rollback**: Replaced `reverse([X, 1], [2, 1])` (which failed on `1 != 2`
  before binding `X`) with `reverse([1, X], [2, 1])` (reversing to `[X, 1]` which matches `[2, 1]`
  and binds `X = 2`), accompanied by positive control `wam_reverse_bind_control/1` proving `X=2`
  before explicit failure triggers backtracking, and verifying `X` is restored to unbound.
- **Unifier mismatch rollback**: Replaced `reverse([X, a], [b, c])` with `reverse([a, X], [b, c])`
  (reversing to `[X, a]`, so unifier head traversal unifies `X` with `b` first before tail mismatch
  `a \== c`), accompanied by positive control `wam_reverse_mismatch_positive/1` proving `X=b` and
  direct C unifier rollback probe `c_unifier_rollback` verifying trail unwinding on failure.
- **Stable output handles**: `examples/pkg_resolver/c/reverse_driver.c` preserves the original
  output-variable heap handle (`r_ref` with `VAL_REF` to heap slot) across all queries rather than
  relying on clobbered caller argument registers (`state.A[1]`).

The runtime allocates a new list spine while sharing element values and variable identities
via `wam_sort_identity_value` without deep-copying terms. First arguments that are unbound,
cyclic, open, improper, or non-lists report `WAM_ERR_UNSUPPORTED` with diagnostic operator
`"reverse/2"` and arity 2.

The worker's earlier smoke report records 134 generated predicates and GCC success. It reports
`empty_requests` (`ok []`). The three nonempty smoke cases (`single_package`,
`backtrack_conflict_deeper`, `unsatisfiable_missing`) all progress past `reverse/2` and
reach the next unsupported builtin: `append/3` (driver exit 4).

Execution command:
```bash
swipl --on-error=halt -g run_tests -t halt tests/test_wam_c_reverse.pl
```
Result: the earlier revision exited 0; the strengthened revision needs verification.

Smoke slice:
```bash
bash examples/pkg_resolver/c/build_and_smoke.sh
```

| Case | Earlier Worker-Reported Result (not fresh independent verification) |
| --- | --- |
| `empty_requests` | MATCH `ok []` (driver 0) |
| `single_package` | unsupported `append/3` (driver 4) |
| `backtrack_conflict_deeper` | unsupported `append/3` (driver 4) |
| `unsatisfiable_missing` | unsupported `append/3` (driver 4) |

Combined verification script `examples/pkg_resolver/c/verify_reverse.sh` runs `tests/test_wam_c_reverse.pl`
then `examples/pkg_resolver/c/verify_maplist.sh`, preserving failure from either.
The next reached unsupported builtin is `append/3`, replacing `reverse/2` as the smoke blocker.
Scope was stopped at `append/3` per instructions. No general parity or merge-ready claims.

## Maplist Implementation Checkpoint (2026-09-09)

The conservative `maplist/2` subset now passes all 24 original behavior cases,
one non-maplist query-observation control, and source/runtime eligibility checks.
It accepts finite proper lists and literal atom goals naming a canonically
verified single bodyless unary fact. Empty lists succeed irrespective of goal.
Other goals and malformed lists report unsupported, not logical failure.

The runtime shares list-head heap cells, preserves caller registers and argument
contexts, and executes only fact-unification steps without nested `wam_run` or
callee `PROCEED` side effects. Normal caller backtracking unwinds shared bindings.
Query tests retain heap handles instead of assuming argument registers survive
calls. Existing member/sort/indexed/diagnostics tests were unchanged.

Independently reviewed and executed through the confined bridge gate:

```bash
bash examples/pkg_resolver/c/verify_maplist.sh
```

| Lane | Exit | Result |
| --- | --- | --- |
| maplist | 0 | 24 original cases + one control; metadata and comparator checks pass |
| member | 0 | 28 cases, including forced trail growth |
| diagnostics | 0 | unsupported/success/failure classification passes |
| sort | 0 | 24 cases |
| indexed | 0 | 11 C/SWI comparisons |
| resolver smoke | 1 | 134 predicates generate; GCC succeeds; 1 match, 3 explicit reverse/2 blocks |

The next reached unsupported builtin is `reverse/2`, replacing `maplist/2` as
the smoke blocker. The static inventory is only a hint and still lists some
implemented builtins. This is a provisional subset, not full resolver parity,
general nondeterministic maplist, or merge readiness. Independent merge review
and broader stress coverage remain necessary.

During this cycle an added C character literal broke the surrounding Prolog
quoted source. SWI printed syntax errors but returned zero without running this
suite. The quoted source was corrected; the verification wrapper now uses
`--on-error=halt`, and the final run produced actual case-by-case passes.

## Historical Maplist Test-First Checkpoint (2026-09-09)

`tests/test_wam_c_maplist.pl` now compiles and executes fresh C against 24
behavioral cases. The source-reviewed command was executed through the bridge's
confined command gate:

```bash
swipl -g run_tests -t halt tests/test_wam_c_maplist.pl
```

GCC succeeded; the suite exited 1 as an intentional red baseline. All 15
supported-behavior cases reported runtime errors because `maplist/2` is not yet
implemented. All 9 explicit unsupported-case checks passed. The WAM emission
and comparator controls passed; the generated-runtime handler check failed.
No emitter or runtime changes were made in this test-first cycle.

The proposed first subset is finite proper lists and atom goals naming a
statically proven single bodyless unary fact. Empty lists succeed without
validating the goal. Tests cover shared-variable conflict, rollback of bindings
from an earlier element, repeated-variable fact heads, and caller continuation.
Register preservation and frame lifetime still need implementation review.
This is not general maplist support, resolver parity, or merge readiness.

## Member checkpoint (2026-09-09, cut/trail follow-up)

`member/2` remains provisional: finite proper lists only, **not merge-ready**.
This packet fixed the two remaining acceptance probes. The
member-walk cursor still carries the list value across iterations (controller
fix; not undone). No `maplist/2` work.

The worker run and a later independent, source-reviewed sandboxed run used:

```bash
bash examples/pkg_resolver/c/verify_member.sh
```

Compilation succeeded before each binary ran. Exit statuses:

| Lane | Exit | Notes |
| --- | --- | --- |
| member | 0 | gcc `exit(0)`, runner `exit(0)`; `cut_all` and `trail_after_growth` now pass |
| diagnostics | 0 | still uses unsupported `reverse/2` |
| sort | 0 | 24 cases |
| indexed | 0 | 11 C/SWI comparisons |
| resolver smoke | 1 | 134 predicates generate; GCC 0; 1 match, 3 blocks |

Fixes and evidence:

1. **`findall(X, (member(X,L), !), Xs)` (`cut_all`)** — `!/0` used the
   clause/query call-base, below the inlined findall sentinel CP.
   `end_aggregate` then skipped sentinel restore, so the driver read `A1`
   as the input list rather than the singleton bag. Floor: never prune
   below the innermost open aggregate's `sentinel_b`. SWI oracle still
   requires `[a]`. Probe now `[PASS] cut_all`.
2. **`member(E,L), sort(Pad,_), E=C` after growth** — `TrailEntry` stored
   raw `H_array` pointers, unsafe if heap growth moves the allocation.
   Heap cells are now trailed by index. Member CPs also save the 32-register A
   window (not arity 2). Both changes landed together: the passing result
   does not isolate which caused the original end-to-end failure.
3. **No-growth control (`trail_no_growth`)** — same `wam_member_grow/4`
   with a 3-element pad. Distinguishes register/CP restoration from
   realloc: `cap 64→64`, `A0=c`, `[PASS]`. Growth probe: `cap 4096→8192`,
   `grew=1`, `H 2054→4104`, `A0=c`, `[PASS]`. Both succeeding means the
   continuation survived allocation, not only the no-growth path.
4. **Forced trail relocation** — independent inspection replaced subtraction
   between potentially unrelated pointers with aligned address classification.
   A driver self-check allocates a distinct heap while the old heap is live,
   moves its contents, and verifies unwind restores both the heap cell and an
   A-register cell. This check fails the driver on error and passed in the
   independent run; it does not depend on realloc choosing to move memory.

Retained coverage still includes basic member, alias identity, duplicate,
partial-unification, nested, invalid-list, and query-reuse. Sort and
indexed suites were not weakened.

Resolver smoke (actual execution, not the static inventory that still
lists member/sort as missing):

| Case | Result |
| --- | --- |
| `empty_requests` | MATCH `ok []` (driver 0) |
| `single_package` | unsupported `maplist/2` (driver 4) |
| `backtrack_conflict_deeper` | unsupported `maplist/2` (driver 4) |
| `unsatisfiable_missing` | unsupported `maplist/2` (driver 4) |

Wrong-answer control still rejects `[-(bar,v(9,9,9))]`. Next scoped
resolver builtin is `maplist/2`; not implemented here. No full corpus,
performance, memory-safety, or merge-readiness claim.

### Prior member checkpoint (same day, before this follow-up)

`member/2` was implemented with a live heap-sharing list cursor and a
member-specific choicepoint continuation. Two source-reviewed runs
compiled; basic enumeration, duplicates, prebinding, rollback, nested
enumeration, variable identity, invalid-list, and repeated-query checks
passed, but **growth followed by retry failed** (logical failure,
capacity 4096→8192) and `findall`-with-cut failed. A worker had removed
the cut test and turned growth into a non-failing diagnostic; the
controller restored both as required failing checks. The original
internally constructed compound wrapper also failed; its replacement
tests a supplied compound and does not qualify that wrapper shape.
The controller's cursor-value fix had not yet been executed in that
checkpoint. Independent gate verification after that restore showed
member suite exit 1 with only `cut_all` and `trail_after_growth`
failing — those are the probes this follow-up fixed.

## Smoke slice

Derived from `examples/pkg_resolver/test_resolver.pl`
(`scenario_catalog/2` ~40–82, `corpus_case/4` ~391–398):

| Case ID | Catalog | Query | SWI first answer |
| --- | --- | --- | --- |
| `empty_requests` | `empty` | `resolve([])` | `[]` |
| `single_package` | `single` | `resolve([foo])` | `[foo-v(1,0,0)]` |
| `backtrack_conflict_deeper` | `backtrack_conflict` | `resolve([a,b])` | `[a-v(1,0,0), b-v(1,0,0), c-v(1,0,0)]` |
| `unsatisfiable_missing` | `missing` | `resolve([foo])` | fail |

Wrappers bind catalog and request and leave `Sel` unbound. The driver
prints `write_canonical`-style terms. Query input is never unified with
the expected answer.

`run_smoke_c.sh` also compares the C `single_package` output against a
deliberately wrong term `[-(bar,v(9,9,9))]`. That comparison must fail;
if it matches, the harness is broken.

## Build / run

From the repo root (needs `swipl` 9.x, `gcc`, `timeout`):

```bash
bash examples/pkg_resolver/c/build.sh
bash examples/pkg_resolver/c/run_smoke_c.sh
```

`build.sh` is regenerable. Generated products live under `generated/`
and are gitignored. `write_wam_c_project/3` currently comments
`// Name/Arity: compilation failed` instead of failing; `build.pl`
treats that as a hard error (fail closed). Generator exit 0 alone is
not evidence of a runnable resolver.

## Status classes

`run_smoke_c.sh` does not treat every nonzero as “the query failed”:

| Class | Meaning |
| --- | --- |
| logical `STATUS ok` / `STATUS fail` | WAM produced a first solution or legitimate failure |
| generator_omission | `wam_run_predicate` returned `WAM_ERR_OOB` (predicate not registered) |
| runtime_error | unexpected WAM status, including `WAM_ERR_UNSUPPORTED` |
| timeout | `timeout(1)` 15s |
| crash_* | signal / abort |
| oracle_error | SWI dump failed |

The C `driver` exits 0 for both `ok` and `fail`. That is a produced
logical outcome, not a corpus pass. Comparison against SWI decides pass.
Unsupported builtin execution is not a logical outcome: the driver
prints `STATUS runtime_error`, `KIND unsupported_builtin`, and
`BUILTIN functor/arity`, then exits 4.

## Frozen

`resolver.pl` and `test_resolver.pl` stay frozen. Other target lanes
and shared WAM remain frozen. This C lane may update runtime/codegen,
the smoke driver/status handling, tests, generated products, and this
README when an authorized packet says so. Do not implement missing
builtin semantics from this directory unless that packet asks for it.

## Indexed dispatch

`wam_target.pl` emits indexed-dispatch `try` / `retry` / `trust`
label chains (`format_dispatch_chain/5`) for multi-clause first-arg
groups. These are **not** `try_me_else` / `retry_me_else` / `trust_me`:
the instruction target is the clause body, and the choice point holds
the next chain instruction (`P+1`). `order_lt/2` has two list-headed
clauses, so it emits `try L_order_lt_2_3_body`.

The C target parses and executes those ops as `INSTR_TRY` /
`INSTR_RETRY` / `INSTR_TRUST`. Repro (now expects compile success):

`swipl -q -g main -t halt examples/pkg_resolver/c/repro_try_dispatch.pl`

## Verification limits

### Sort verification (current)

`sort/2` now sorts and deduplicates finite proper lists, allocating output cons
cells while retaining the original element values and variable identities. It
uses a live-term comparator rather than the aggregate stored-term comparator.
This is not a performance benchmark or a claim of complete SWI term support.

Independent source-reviewed, sandboxed verification:

```bash
bash examples/pkg_resolver/c/verify_sort.sh
```

- Sort suite: exit 0, 24 cases covering ground ordering/deduplication, compound
  and list ordering, prebound output, mixed integer/float ordering, input
  preservation, shared/distinct variables, direct-cell variables, heap growth,
  and invalid-list diagnostics. Ground expectations are computed by SWI from
  matching inputs; property checks inspect actual C values and alias identity.
- A new direct-cell variable regression failed before the controller corrected
  a by-value copy that lost the original heap cell identity. It now passes.
- Existing compiled-C diagnostics and all 11 indexed-dispatch comparisons pass.
- All 134 resolver/wrapper predicates generate; GCC exits 0.
- `empty_requests`: C and SWI both return `ok []` (driver exit 0).
- The other three cases: first reached unsupported operation is `member/2`
  (driver exit 4). Overall smoke exits 1: one verified match, three blocks.

Limits: invalid/open/cyclic input lists are explicitly unsupported rather than
implementing SWI exception/cyclic-list semantics. List length is bounded at
262,144 items and recursive comparison depth at 256. Non-finite numeric cases,
full term-domain parity, concurrency, full C suite, and performance have not been
qualified. The static inventory still falsely lists `sort/2` as missing; actual
execution is authoritative. Next scoped work is backtracking-safe `member/2`.

### Builtin diagnostics verification (before sort)

The diagnostics cycle was independently verified with:

```bash
bash examples/pkg_resolver/c/verify_diagnostics.sh
```

This command was source-reviewed and executed inside the bridge sandbox. The
focused generated/compiled-C diagnostics exited 0, including unknown integer,
atom, and arity-3 calls, normal success/failure, comparisons, backtracking,
compiled/meta aggregates, and repeated query reuse beyond aggregate-stack
capacity. Error exits release query-local aggregate frames and restore the
query's heap, trail, environment, and argument-context positions. The existing
indexed-dispatch suite exited 0 with all 11 C/SWI comparisons passing.

Resolver generation and GCC exited 0. Smoke exited 1 with these actually reached
first unsupported operations (not inferred from the static inventory):

| Case | First unsupported operation | Driver exit |
| --- | --- | --- |
| `empty_requests` | `sort/2` | 4 |
| `single_package` | `member/2` | 4 |
| `backtrack_conflict_deeper` | `member/2` | 4 |
| `unsatisfiable_missing` | `member/2` | 4 |

At that checkpoint the smoke result was **four runtime blocks, zero verified matches**.
The former matching failure was not trustworthy. The wrong-answer smoke control
still rejects its comparison, but currently sees a runtime error rather than a
successful selection; it does not establish successful-answer formatting parity.
The indexed-dispatch comparator has its own wrong-answer regression.

The full C-target suite was not run in that cycle. Its next bounded work was to add
and behaviorally test the missing builtins, starting with `sort/2` for the empty
case or `member/2` for the three nonempty cases; nondeterministic membership must
preserve backtracking. No full corpus or performance claim is made.

Resolver generation compiles 134 predicates and GCC succeeds. Smoke
execution is authorized in this lane. A static `INSTR_BUILTIN_CALL`
inventory is only a hint: escaping can mislead matching, and listing an
op does not prove it is reached. Reached unsupported builtins now
surface as `WAM_ERR_UNSUPPORTED` rather than `STATUS fail`. Ordinary
logical failure (`fail/0`, false comparisons) stays `STATUS fail`.

The diagnostics cycle did **not** implement missing builtin semantics. Expected
smoke successes may still BLOCK on the first reached unsupported
operation. `unsatisfiable_missing` matching SWI fail is not trusted
parity if an unsupported builtin is reached first.

The dedicated indexed-dispatch suite has 11 compiled-C/SWI comparisons. Two
initial test shapes failed and are not both represented in that passing set:

- `idx_first_b(Y) :- idx_p(b, Y)` returned `b` when the driver inspected A0,
  while SWI returned `2`. That wrapper test was removed. Whether this is a
  driver/query-result convention issue or a runtime gap needs a separate pin.
- `idx_all_list(Ys) :- findall(Y, idx_list([x], Y), Ys)` returned failure rather
  than `[1,2]`. The passing test instead supplies `[x]` as an input to
  `idx_all_list(L, Ys)`. It exercises indexed list dispatch, not construction
  of the list within the compiled aggregate wrapper.

These exclusions do not establish that the observed failures are unrelated to
the patch. They remain unresolved until independently reproduced and classified;
do not count the passing focused suite as coverage of either original shape.
