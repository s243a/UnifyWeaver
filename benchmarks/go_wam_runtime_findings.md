# Go hybrid WAM runtime — debugging notes

Working notes from debugging why the Go hybrid-WAM target's effective-distance
benchmark returns `tuple_count=0` after the gen-side fixes from
`988f0a6` and `e047afd5` made it build.

## Method

Added a `debug_probe.go` to the generated benchmark project, called from
`main()` before the timed query loop. Walks up the predicate stack from
leaf to root: `category_parent/2` (single-table fact lookup) →
`category_ancestor/4` (recursive) → `effective_distance_sum_selected/3`
(which the bench actually calls). Three rounds of bug-hunting, three
real runtime bugs found.

## Bug 1 — `bindUnbound` corrupting A1 (fixed in `26810733`)

`bindUnbound` did an unconditional `vm.putReg(u.Idx, val)` after the
alias-rewrite loop. For runtime-constructed Unbounds (where `Idx` is a
real X-register slot) this was a no-op atop the loop. For
caller-constructed Unbounds without `Idx` set — e.g. a benchmark driver
doing `&Unbound{Name: "weight"}` to ask for an output register — `Idx`
defaults to zero and the line silently overwrote `Regs[0] = A1`.

Reproducer (before fix): `category_parent("1640s", X)` returned
`first=[17th_century, 17th_century]` instead of `first=[1640s, 17th_century]`.
After fix: returns the correct row.

Fixed in `templates/targets/go_wam/state.go.mustache`.

## Bug 2 — `PutVariable` leaving stale `Bindings` entries (THIS PR)

The runtime's `Bindings` map is keyed by the Unbound's `Idx`, which for
WAM-created Unbounds is the X-register slot index. When a predicate is
re-entered (recursive call, or just a second activation), `PutVariable`
creates a fresh Unbound at the same slot — but the previous activation
may have left a `Bindings[Xn]` entry from a successful unification.
Without clearing it, `deref` of the "new" Unbound chases the stale
binding and the next `Unify(unbound, Integer(N))` compares the leftover
value against `N` instead of binding cleanly. For the
effective-distance benchmark, this manifested as `length(Path, N)`
inside the recursive clause of `category_ancestor/4` returning false
because the freshly-allocated `N` register was already "bound" to the
length from the *previous* recursion frame.

Reproducer (before fix), via instruction trace:

```
[trace] PC=24061 PutVariable X200 A2     ; A2 = new unbound (Idx=200)
[trace] PC=24062 BuiltinCall length/2    ; length([Chemistry, BoC], A2)
[length/2] -> 2 elements; arg2=__R200 (type *main.Unbound)
[length/2] Unify returned false          ; ← bug
```

`Unify(__R200, Integer(2))` returned false because `deref(__R200)`
followed `Bindings[200] = Integer(1)` (left over from the outer call's
length/2) and then compared `Integer(1) == Integer(2)`.

Fix: `PutVariable` now trails and deletes `Bindings[Xn]` before
allocating the fresh Unbound, so deref sees the cleared slot:

```go
case *PutVariable:
    vm.trailBinding(i.Xn)
    delete(vm.Bindings, i.Xn)
    v := &Unbound{Name: ..., Idx: i.Xn}
    ...
```

The `trailBinding` ensures backtrack restores the prior entry.

Fixed in `src/unifyweaver/targets/wam_go_target.pl` (gen template,
`wam_go_case('PutVariable', ...)`).

## Bug 3 — `pushChoicePoint` only saving A-registers (THIS PR)

`pushChoicePoint` saved `vm.Regs[:arity]` — i.e. the first N register
slots, where N is the predicate's arity. But the WAM body uses
X-registers (`Regs[100..]`) and Y-registers (`Regs[200..]`) too, and on
backtrack those slots may hold heap-Refs to addresses that
`heapTrimTo(cp.HeapTop)` just shrunk past. The next time anything
called `deref` on one of those Refs, it would crash with
`index out of range [N] with length N` reading off the end of `vm.Heap`.

Reproducer (with bug 2 fixed but bug 3 still present):

```
ancestor("Quantum_mechanics", "Branches_of_chemistry") PANIC: index out of range [21] with length 21
ancestor("Branches_of_chemistry", "Physics") PANIC: index out of range [15] with length 15
ancestor("Branches_of_chemistry", "Natural_sciences") PANIC: index out of range [15] with length 15
ancestor("Natural_sciences", "Physics") PANIC: index out of range [3] with length 3
```

Fix: snapshot the full register file at CP push time, so backtrack's
`copy(vm.Regs[:len(cp.SavedRegs)], cp.SavedRegs)` restores X- and
Y-registers along with A-registers. The fixed cost (copying a
`[512]Value` array per CP) is smaller than the existing cost of
`copyStack(vm.Stack)` per CP.

Fixed in `templates/targets/go_wam/state.go.mustache`
(`snapshotAllRegs` helper).

## Bug 4 — `Idx` collision across activations (THIS PR)

Adding deeper instrumentation to `is/2` revealed why 3-hop recursion
still failed: at depth 3, the outer activation's `X206 is X207 + 1`
computation evaluated `X207` to the *inner* activation's bound
integer, producing `1 is 2` and failing.

`PutVariable` was setting `Unbound.Idx = i.Xn` (the X-register slot
index). Bindings is keyed by `Idx`, so a recursive call's
`PutVariable X207` reused `Idx=207` and shared `Bindings[207]` with
the outer call's `Unbound{Idx:207}`. The fix attempted in bug 2
(`delete(Bindings, i.Xn)` at PutVariable time) papered over the issue
for 2 hops but at 3 hops would actively erase a still-live binding
the outer needed.

Fix: allocate a globally-unique `Idx` for every new logical variable
via a `NextVarId` counter on the WamState. Different activations'
PutVariables get different Bindings keys, so neither activation can
clobber the other's binding. Same change applied to the SetVariable
and UnifyVariable handlers (they also create Unbounds with `Idx=i.Xn`).

`WamState.Clone` had a sister bug — it didn't copy `NextVarId` to the
sub-VM in `executeAggregate`, so the cloned VM's counter started at
zero, allocVarId clamped to the 1000 floor, and the sub-VM's "fresh"
Unbounds collided with parent-bound Idx values in the cloned
Bindings. Fixed by carrying NextVarId through Clone.

After bug 4:

```
ancestor("Quantum_mechanics", "Natural_sciences", H, [QM])  -> 2 solutions  ✓ 3-hop
```

## Bug 5 — `extract_shared_start_pc` taking the wrong digits (THIS PR)

The Go gen's `extract_shared_start_pc` parsed `"label": NNNNN,` from
lib.go to learn the WAM start PC for the bench. It used
`sub_string(Content, After, _, _, Rest0)` with both length and
leading-position unbound, which makes sub_string nondeterministically
enumerate substrings in increasing length order (0, 1, 2, ...).
`string_digits_prefix` accepted the FIRST length where it could pull
at least one digit — length 2, giving the substring `" 2"` (leading
space + the first digit of `24145`). The bench was being told to
start the WAM at `PC=2` (which is `max_depth/1`'s `GetConstant 10 A1`)
instead of the actual `PC=24145` (`effective_distance_sum_selected/3`).
Every bench query failed at the very first instruction.

Fix: pin the sub_string length to a fixed 32 chars so the digit
prefix sees the entire numeric token.

This was the single biggest unblock: the bench wasn't running the
benchmark at all, just hitting an immediate unification failure on
every seed and reporting `tuple_count=0`.

## Bug 6 — atom equality always doing string compare (mitigated)

The user observed that the bench's slowness suggested missing atom
interning. `Atom.Equals` was doing `v.Name == o.Name` unconditionally;
the lowered emitter already produces a shared `wamAtom_*` table for
literals it emits, but `Atom.Equals` didn't take advantage of it.

Cheap mitigation: pointer-identity short-circuit at the top of
`Atom.Equals`. When atoms come from the shared intern table they're
the same pointer and equality is O(1). Falls through to string
compare for un-interned atoms (raw `&Atom{Name:"..."}` literals
scattered through `lib.go`'s WAM bytecode and the bench's
`&Atom{Name: category}` arguments). A more thorough fix would emit
the WAM bytecode atoms via the same intern table so EVERY atom is
shared by pointer; that's a bigger refactor and out of scope here.

## After all six fixes — what works

```
$ ./wam-go-effective-distance-bench  (scale 300, kernels_off)
mode=accumulated_go_wam
kernel_mode=kernels_off
load_ms=0
query_ms=216015      (~216s)
total_ms=216015
seed_count=386
tuple_count=12       (was 0)
article_count=74     (was 31, all distance=1.0)
```

The bench now actually runs the benchmark. Output rows have real
fractional distances (`0.993865` etc.) instead of all-1.0 sentinels
that tagged-directly-with-Physics articles got pre-fix.

## What's still open

- **Speed.** 216 s for 12 tuples is way too slow versus Rust's
  `total_ms=33`. Most likely culprits: (1) unique-Idx-counter inflates
  the Bindings map and trail, (2) `executeAggregate` clones the full
  WAM state per sum loop iteration, (3) atom comparisons fall back to
  string compare for non-shared atoms (mitigated above but the bulk
  of WAM bytecode atoms aren't shared yet).
- **Result completeness.** 74 articles out of 271 expected. Some
  category chains aren't being walked all the way, or some seeds
  aren't producing weight tuples. Probably another register/heap
  management edge case at depth >= 4 — needs another instrumented
  trace.

## Reproducing

```bash
swipl -q -s examples/benchmark/generate_wam_go_effective_distance_benchmark.pl \
    -- data/benchmark/300/facts.pl /tmp/uw/go-bench accumulated kernels_off
cd /tmp/uw/go-bench

# Drop in a debug_probe.go that calls collectSolutions on
# category_ancestor/4 with a 1-element list as the path argument, add
# debugProbeRuntime() at the top of main() before computeResults(),
# build and run.
go build ./... && ./wam-go-effective-distance-bench
```

Expected probe output (1-hop and 2-hop ancestor return 1 solution
each; no panics) confirms bugs 2+3 are fixed.

## Followup — `kernels_on` mode regression and the "remaining drift" non-bug

After the Y-reg save/restore landed (`fix/wam-go-depth-completeness`,
commit `f6620806`), `kernels_off` produced 271 rows matching the
optimized Prolog target byte-for-byte. The cross-target script
defaults to **`kernels_on`** though, and that mode was producing only
31 rows / `tuple_count=0` — i.e. only direct article→Physics seeds
contributed, every recursive weight query returned no solutions.

Root cause: `collect_wam_predicates(Module, kernels_on, ...)` in
`generate_wam_go_effective_distance_benchmark.pl` listed the kernel
predicates and `category_ancestor/4` but **omitted
`category_parent/2`**. The kernel `category_ancestor$power_sum_bound/3`
calls `category_ancestor/4`, which calls `category_parent/2` — so
without the leaf fact predicate compiled into the WAM bytecode,
recursion has nothing to call and every weight aggregation returns
`sum=0` (which the kernel rejects via `C > 0`).

Fix: add `Module:category_parent/2` to the kernels_on list. After the
fix, both modes generate byte-identical bytecode (`cmp lib.go atoms.go
→ 0`) and produce identical output at dev (19/19 articles, 16 tuples)
and scale-300 (271/271 articles, 211 tuples).

Separately, the "remaining numerical drift" called out earlier
(`Brownian_motion 0.993865 vs ref 0.993717`,
`2008_in_science 4.37 vs ref 3.73`) turned out **not** to be a Go WAM
bug. The Go output matches the optimized-Prolog-target output
byte-for-byte at dev scale; the optimized-Prolog output also matches
running `effective_distance.pl` directly through unoptimized
SWI-Prolog. All three implementations honour the workload's
`max_depth(10)`. The reference TSV in `data/benchmark/300/` was
generated with a much deeper cutoff (matches reference exactly at
`max_depth ≥ 50`, close-but-not-exact at `max_depth=20`). Bumping
`max_depth/1` in `examples/benchmark/effective_distance.pl` to ~50
brings everything back into agreement, but that's a workload-level
decision affecting all targets and is left as a follow-up.

## Still open

- **Speed at scale-300.** The bench now produces the right answer in
  both modes but takes ~340 s for 211 tuples (Rust does the same in
  ~23 ms). Likely culprits unchanged from above: (1) `[100]Value`
  Y-reg snapshot at every Allocate, (2) per-iteration `Clone()` in
  `executeAggregate`, (3) atom comparisons that miss pointer identity
  for non-bytecode atoms.
- **Workload `max_depth(10)` vs reference `max_depth >= 50`.** Either
  bump the workload to match the reference, or regenerate the
  reference at depth 10. Pick one and apply it across all targets so
  the cross-target diff stays clean.

## Followup #2 — perf optimizations that didn't pan out

After the `perf/wam-go-aggregate-and-yreg` patch (Regs[512] → Regs[320]
+ resolve `default` labels in SwitchOnConstant; merged) brought
scale-300 down to ~178s, I investigated three additional targets the
profile flagged. All were prototyped, measured, and reverted because
the speedup either failed to materialise or only showed up at the
warm-cache 50-seed scale and inverted at full scale-300.

Profile of the post-`perf` baseline (scale-300, 50 seeds, kernels_off,
~24s sample):
- `runtime.memmove`: 16.9% flat
- `runtime.bulkBarrierPreWrite`: 25.6% cum (write barriers per pointer
  store)
- `runtime.mallocgc + gcBgMarkWorker`: ~33% cum combined (heap allocs
  and GC marking)
- `backtrack` line `copy(vm.Regs[:len(cp.SavedRegs)], cp.SavedRegs)`:
  9.98s / 23.82s = 41.9% of total
- `backtrack` line `vm.Stack = copyStack(cp.Stack)`: 4.55s = 19.1%

### What I tried

1. **`Ctx.SaveRegBound` + shrunk snapshot range**: walk Code at
   `NewWamContext`, find the highest `Ai`/`Xn` ever referenced
   (turned out to be 207 for the bench), snapshot only `Regs[:208]`
   instead of `Regs[:320]`. **Theoretical**: 35% smaller copy. **Real**:
   no measurable improvement at full scale-300; runs alternated between
   "slightly faster" and "slightly slower" with the variance band
   (~±10%) swamping any real signal. Reverted.

2. **`SavedRegs` pool**: replace `make([]Value, ...)` in `snapshotAllRegs`
   with a per-`WamState` free-list seeded by `truncateChoicePoints`
   (called from every CP-removal site: indexed/foreign-results
   exhaustion, TrustMe, CutIte, !-cut). Required `Clone()` to
   deep-copy each cloned CP's `SavedRegs` so a sub-VM's pool can't
   recycle a slice the parent VM still references. **Theoretical**:
   eliminates the per-CP heap alloc, drops `mallocgc`/`gcBgMarkWorker`
   share. **Real**: ~3% improvement at the 50-seed scale, **~6%
   regression at full scale-300** (185s vs 175s baseline). Best guess:
   the deep-copy in `Clone` plus the recycled-slice scanning costs
   exceed the alloc savings on the longer run, where GC has more
   headroom to keep up. Reverted.

3. **Pre-sized `Bindings` map** (`make(map[int]Value, 4096)`): no
   improvement, marginal regression. The map's hash-table overhead
   dominates over its initial-bucket allocation, which Go's allocator
   handles cheaply enough that pre-sizing doesn't help.

### Variance observation

Wall-clock at scale-300 / 50 seeds shows ~±10% variance run-to-run
even with the same binary. Optimisations smaller than ~15% can't be
distinguished from noise without a longer measurement campaign or
external CPU isolation. The remaining bottlenecks (snapshot+restore
copy, write barriers per `Value` store, choicepoint stack copy) are
all O(N) in the snapshot size, so chipping away at constants probably
won't deliver a step-change.

### What would actually move the needle

The code paths above all stem from one design choice: the runtime
treats the env-frame stack and the snapshot-on-CP-push register file
as deep-copyable. A real WAM keeps env frames alive across choicepoints
(environment trimming via `B0`) and saves only an `E` pointer / `B`
mark on CP push, never copying the stack. That removes both
`copyStack` and the reason `snapshotAllRegs` has to capture Y-regs at
all (Y-regs are *in* env frames). The `EnvFrame` struct already
carries `B0`; making `Deallocate` honour it (don't pop while a
younger choicepoint references the frame) and switching `pushChoicePoint`
to a length-mark would deliver the structural fix that none of the
constant-factor patches above could.

A second-order win is dropping the global `Bindings map[int]Value` in
favour of a slice indexed by `Idx - allocVarIdBase`: `deref` and
`unwindTrailTo` would become array indexing instead of map ops,
which the profile's `runtime.mapaccess2`/`mapassign` allocations
account for ~10% of total. Slice-grow-on-demand handles the rare
high-Idx case.

Both are non-trivial rewrites and out of scope for the 1-2 commit
patches in this branch series.
