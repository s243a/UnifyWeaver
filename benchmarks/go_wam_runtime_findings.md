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
