# Go hybrid WAM runtime — debugging notes

Working notes from debugging why the Go hybrid-WAM target's effective-distance
benchmark returns `tuple_count=0` after the gen-side fixes from
`988f0a6` and `e047afd5` made it build.

## Method

Added a `debug_probe.go` to the generated benchmark project, called from
`main()` before the timed query loop. Probe walks up the predicate
stack from leaf to root: `category_parent/2` (single-table fact lookup)
→ `category_ancestor/4` (recursive) → `effective_distance_sum_selected/3`
(which the bench actually calls).

## Confirmed bug — fixed

**`bindUnbound` writes the bound value into a register slot keyed by the
Unbound's `Idx` field, which corrupts A1 when callers don't set `Idx`.**

Reproducer, before the fix:

```go
rows := collectSolutions(pc,
    &Atom{Name: "1640s"},                  // A1
    &Unbound{Name: "_p"})                  // A2 — Idx defaults to 0
// expected: [Atom("1640s"), Atom("17th_century")]
// actual:   [Atom("17th_century"), Atom("17th_century")]   // A1 wiped
```

The runtime's `bindUnbound` (in `templates/targets/go_wam/state.go.mustache`)
unconditionally does `vm.putReg(u.Idx, val)` after the alias rewrite
loop. When the for-loop already finds and rewrites every register that
points at `u`, that follow-up call is redundant. When the caller
constructs an Unbound without setting `Idx` (e.g. a benchmark driver
asking for an output register at A3), `Idx` defaults to zero and the
fallback silently overwrites Regs[0] = A1 with the bound value.

**Fix landed:** removed the unconditional `putReg(u.Idx, val)` — the
alias-rewrite loop above already handles every register that genuinely
holds `u`. Also defensively set `Idx: 2` on the bench-side
`&Unbound{Name: "weight"}` so the benchmark doesn't depend on the
runtime fix being present.

After the fix, the probe shows:

```
category_parent("1640s",            X) -> 5 solutions first=[1640s, 17th_century]
category_parent("Physics",          X) -> 2 solutions first=[Physics, Natural_sciences]
category_parent("Quantum_mechanics", X) -> 2 solutions first=[Quantum_mechanics, Branches_of_chemistry]
```

i.e. the input register is no longer being clobbered.

## Still open — recursive predicate fails

Even after the `bindUnbound` fix, `category_ancestor/4` returns zero
solutions for any query I tried:

```
category_ancestor("Quantum_mechanics", "Physics", H, P=unbound)         -> 0
category_ancestor("Branches_of_chemistry", "Physics", H, P=unbound)     -> 0
```

The known-good answer set says e.g. `Quantum_mechanics → Branches_of_chemistry → Natural_sciences → Physics`
exists, so `category_ancestor` should return a solution. The base
clause `category_ancestor(Start, Target, 1, _) :- category_parent(Start, Target), \+ member(...)`
fails for `Quantum_mechanics → Physics` (no direct parent edge), so
the recursive clause has to fire and walk via `Branches_of_chemistry`.

A separate path (passing `Path = &List{Elements: [&Atom{Name: "Quantum_mechanics"}]}`)
panics inside `listToSlice` → `deref` with `index out of range [3] with length 3`,
which suggests that bench-constructed `*List` values aren't being
heap-laid-out in the form the WAM list-traversal builtins expect.

So at least one of the following is broken:

- **Multi-clause backtracking.** TryMeElse / RetryMeElse / TrustMe
  choice-point management. When clause 1 fails on the
  `category_parent('Quantum_mechanics', 'Physics')` lookup, the runtime
  may not be backtracking into clause 2 cleanly.
- **List representation / heap layout.** Bench-constructed `*List`
  values panic during `listToSlice` because `deref` reaches off the end
  of `vm.Heap`. The WAM list builtins (member/2, length/2, append/3)
  apparently expect a heap-cell-based list, not a `*List` value with
  inline `Elements`. The two representations need to round-trip but
  don't.
- **Recursive call / continuation-pointer chain.** The recursive call
  to `category_ancestor/4` (inside clause 2) plus the surrounding
  `BeginAggregate` / `BuiltinCall is/2` / `EndAggregate` interaction in
  `power_sum_bound/3` may corrupt CP across the recursion.

I didn't get to the bottom of any of these in this pass.

## Reproducing

```bash
# Build the bench with kernels_off (forces category_parent into WAM):
swipl -q -s examples/benchmark/generate_wam_go_effective_distance_benchmark.pl \
    -- data/benchmark/300/facts.pl /tmp/uw/go-bench accumulated kernels_off

# Drop debug_probe.go into /tmp/uw/go-bench/, add a `debugProbeRuntime()`
# call at the top of main() before computeResults(), build and run:
cd /tmp/uw/go-bench && go build ./... && ./wam-go-effective-distance-bench
```

Look for the `=== DEBUG PROBE ===` block on stderr.

## Suggested next steps

1. **Add an instruction trace to `vm.Run()`** (gated behind an env var)
   that logs PC + opcode + key registers per step. Pipe through `head -200`
   for one `category_ancestor("Quantum_mechanics", "Physics", H, P)` call
   and compare against the equivalent Rust trace. The first divergence
   pins the failure to one of the three suspects above.
2. **Reconcile the two list representations.** Either teach
   `listToSlice` to handle `*List` directly, or have the bench pass
   lists as cons-cells (`*Compound{Functor: ".", Args: [head, tail]}`)
   that go through the heap-allocation path the WAM expects.
3. **Audit the runtime's choice-point backtrack vs. trail unwind**
   for symmetry — the `bindUnbound` bug suggests this code path hasn't
   been exercised enough to shake out the rest.
