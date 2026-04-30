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

## After bugs 1+2+3 — what works, what's still open

After all three fixes:

```
ancestor("Quantum_mechanics", "Branches_of_chemistry", H, [QM]) -> 1 solutions  ✓ 1-hop
ancestor("Quantum_mechanics", "Subfields_of_physics", H, [QM])  -> 1 solutions  ✓ 1-hop
ancestor("Subfields_of_physics", "Physics", H, [SoP])           -> 1 solutions  ✓ 1-hop
ancestor("Quantum_mechanics", "Physics", H, [QM])               -> 1 solutions  ✓ 2-hop
ancestor("Quantum_mechanics", "Chemistry", H, [QM])             -> 1 solutions  ✓ 2-hop
ancestor("Quantum_mechanics", "Natural_sciences", H, [QM])      -> 0 solutions  ✗ 3-hop
```

So the runtime now reliably computes ancestor relationships up to 2
hops. Direct queries (1-hop) and one level of recursion (2-hop) work.
**3-hop+ recursion still fails** — this is a separate, deeper bug,
likely in how nested choice points + heap trims interact when the
recursion gets deep enough to push multiple CPs at once.

The full benchmark `tuple_count` is still 0 because
`effective_distance_sum_selected/3` goes through `power_sum_bound/3`'s
`BeginAggregate` flow, and that path returns 0 even for
direct-edge cases that work via the `category_ancestor/4` probe. The
aggregate Clones the VM state and runs a sub-state to collect
solutions — something in that Clone+sub-Run chain is dropping
solutions that the direct path finds.

## Suggested next steps

1. **3-hop recursion bug.** Add a deeper trace and find why the third
   nested `category_ancestor/4` call doesn't see Subfields_of_physics's
   parent. Probably another register-snapshot / Bindings-key collision
   variant; the recursion at higher depths hits new register slots
   that weren't in the trail.
2. **`BeginAggregate` solution dropping.** The sub-VM in
   `executeAggregate` (`vm.Clone()` + `runUntilPC`) returns 0
   solutions for direct-edge ancestors. Either Clone is missing some
   field or runUntilPC exits before the aggregate accumulates.
   Compare against the Rust runtime's aggregate impl, which works on
   the same workload.
3. **Trail trimming on heap restore.** Audit whether
   `heapTrimTo(HeapTop)` should also unwind any Bindings entries
   whose values now live past the new heap top.

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
