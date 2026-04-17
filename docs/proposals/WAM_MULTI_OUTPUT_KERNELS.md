# Proposal: Multi-Output Kernels for the WAM Haskell Target

**Status:** Proposed
**Motivation:** Port generally-useful graph kernels from the Rust WAM target
**Complexity:** Moderate — contained to the executeForeign generator + templates
**Related:** `src/unifyweaver/core/recursive_kernel_detection.pl`,
`src/unifyweaver/targets/wam_haskell_target.pl` (lines ~256-449)

## Problem

The current Haskell WAM kernel infrastructure only handles kernels with
a **single output register**. Looking at the generator (`wam_haskell_target.pl:289-290`):

```prolog
include(is_output_reg, RegSpecs, OutputRegs),
OutputRegs = [output(OutRegN, OutType)|_],  % <-- takes first, ignores rest
```

The `emit_stream_binding` helper (lines 419-444) binds the kernel's
result stream to exactly one register.

**This works for the two currently-ported kernels:**
- `category_ancestor/4`: output is `Integer` (hop count)
- `transitive_closure2/2`: output is `Atom` (target node)

**It blocks porting the useful multi-output kernels** from the Rust
target:

| Rust kernel | Outputs | Why useful |
|---|---|---|
| `transitive_distance3/3` | target, distance | BFS + distance tracking |
| `weighted_shortest_path3/3` | target, distance | Dijkstra |
| `transitive_parent_distance4/4` | target, parent, distance | Path reconstruction |
| `astar_shortest_path4/4` | target, distance | Heuristic-guided |

Without multi-output support, porting any of these requires either:
- Contriving a single-output variant (loses info)
- Encoding outputs as a compound term (`Str "pair" [Atom t, Integer d]`),
  which requires `get_structure` on the WAM side to unpack — adds
  interpreter overhead that defeats the kernel's performance benefit

## Design

### 1. Register layout — no change needed

The existing `kernel_register_layout/2` metadata already supports
multiple output specs:

```prolog
kernel_register_layout(transitive_distance3, [
    input(1, atom),         % source
    output(2, atom),        % target
    output(3, integer)      % distance
]).
```

The generator just doesn't use them all yet.

### 2. Kernel result type — tuple

The kernel function returns `[(Int, Int)]` (interned target IDs paired
with distances) instead of `[Int]`. Generalized: `[(T1, T2, ..., Tn)]`
for n-output kernels.

Existing single-output kernels keep returning `[Int]` — this is
backward-compatible if we special-case n=1 (no tuple wrapping).

### 3. Stream binding — extend bindResult

Currently:

```haskell
bindResult rv =
  case outReg of
    Unbound vid -> s { wsPC = retPC
                     , wsRegs = IM.insert outRegN (wrapResult rv) (wsRegs s)
                     , wsBindings = IM.insert vid (wrapResult rv) (wsBindings s)
                     , ...
                     }
    _ -> s { wsPC = retPC, wsRegs = IM.insert outRegN (wrapResult rv) (wsRegs s) }
```

Extended to multi-output:

```haskell
bindResult (rv1, rv2) =
  let -- Bind output 1
      outReg1 = ... -- deref reg at OutRegN1
      -- Bind output 2
      outReg2 = ... -- deref reg at OutRegN2
      -- Update both in one go
      newRegs = IM.insert OutRegN1 (wrap1 rv1)
              $ IM.insert OutRegN2 (wrap2 rv2)
              $ wsRegs s
      newBindings = ... -- same pattern for unbound vars
      newTrail = ... -- trail entries for both
  in s { wsPC = retPC, wsRegs = newRegs, wsBindings = newBindings
       , wsTrail = newTrail, wsTrailLen = wsTrailLen s + boundVars }
```

The trail must grow by 1 per bound variable (up to N for N outputs).

### 4. Choice point for stream results

Current code stores remaining results in `HopsRetry !Int ![Int] !Int`
(variable ID, remaining values, returnPC). For multi-output, this
needs to carry the remaining tuples:

```haskell
-- Single-output (existing)
data BuiltinState
  = HopsRetry !Int ![Int] !Int

-- Multi-output (new variant, or generalize)
  | MultiHopsRetry ![Int] ![(Int, Int)] !Int
  --               ^^^^^  ^^^^^^^^^^^^^  ^^^^
  --               varIDs, remaining     returnPC
  --                       tuples
```

Alternatively, keep it generic as `[[Value]]` and dispatch based on
arity in the retry logic.

### 5. Generator changes

Three places in `wam_haskell_target.pl` need updating:

**a) `emit_parameterized_execute_foreign_entry` (line ~280)** — look up
all output regs, not just the first.

**b) `emit_stream_binding` (line ~418)** — generate multi-output bindResult.
Parametrize over number of outputs.

**c) `result_wrap_expr/2` (line ~448)** — extend to per-output-position
wrapping, or inline the wrap logic in bindResult.

### 6. Kernel templates

Add new templates:
- `kernel_transitive_distance.hs.mustache`
- `kernel_weighted_shortest_path.hs.mustache`

Example for `transitive_distance`:

```haskell
nativeKernel_transitive_distance
  :: IM.IntMap [Int] -> Int -> [(Int, Int)]
nativeKernel_transitive_distance edges source =
  bfs [(source, 0)] IS.empty []
  where
    bfs [] _ acc = acc
    bfs ((node, dist):queue) visited acc
      | IS.member node visited = bfs queue visited acc
      | otherwise =
          let visited' = IS.insert node visited
              neighbors = IM.findWithDefault [] node edges
              expanded = [(n, dist + 1) | n <- neighbors,
                                          not (IS.member n visited')]
              acc' = if node == source then acc else acc ++ [(node, dist)]
          in bfs (queue ++ expanded) visited' acc'
```

### 7. Detector — needs per-kernel implementation

Each kernel kind needs its own `detect_X` predicate in
`recursive_kernel_detection.pl`. For `transitive_distance3/3`, the
expected Prolog shape:

```prolog
transitive_distance3(Source, Target, Distance) :-
    transitive_closure2(Source, Target),
    shortest_path_length(Source, Target, Distance).
```

Or inline:

```prolog
tdist(Source, Target, 0) :- Source = Target.
tdist(Source, Target, D) :-
    edge(Source, Mid),
    tdist(Mid, Target, D1),
    D is D1 + 1.
```

The detector matches either shape and extracts the edge predicate
from config.

## Test plan

1. Implement multi-output generator support (no kernel yet)
2. Port `transitive_distance3/3` — verify against a small hand-computed
   reachability graph
3. Add unit tests in `test_wam_haskell_target.pl` checking generated code
   contains both output register bindings
4. Add an end-to-end benchmark using transitive_distance3 on the
   effective-distance graph (reachability distance from seed categories)
5. Confirm no regression on existing `category_ancestor` and
   `transitive_closure2` kernels

## Open questions

### Should all outputs be atoms, or mixed types?

`transitive_distance3` has `(atom, integer)`. Some kernels might have
`(atom, atom, integer)`. The generator needs to handle mixed output
types — not too hard, but each position needs its own wrap expr and
interning/de-interning rules.

### Should the tuple be a Haskell `(a, b)` or a list `[Value]`?

Tuples are more type-safe but require per-arity code generation
(`bindResult (a, b)` vs `bindResult (a, b, c)`). Lists are uniform
but lose static arity checking.

**Recommendation:** Use tuples — the generator can emit the right
arity based on kernel metadata. Haskell's type system then catches
wiring mistakes at compile time.

### Interaction with lowered emitter

The lowered emitter (`wam_haskell_lowered_emitter.pl`) already calls
foreign kernels via `CallForeign`. Multi-output kernels should work
without changes — the lowered code path just invokes `executeForeign`
and lets the WAM dispatch handle the binding.

## Rough effort estimate

Not a time estimate, but ordered from simpler → harder:

1. Generator modifications (emit_stream_binding, emit_parameterized_execute_foreign_entry)
2. `BuiltinState` variant for multi-output retry
3. Kernel template + detector for `transitive_distance3`
4. Tests
5. Benchmark + perf comparison against Rust

Steps 1-4 are each local and testable. Step 5 gives us the comparison
we'd want to document in the perf log.

## Alternative considered: single-kernel generic implementation

Rather than adding more kernels, we could have a single "graph query"
kernel that takes a kind parameter:

```haskell
nativeKernel_graph :: GraphKind -> IntMap [Int] -> Int -> [(Int, Int, Int)]
```

This collapses the kernel count but loses the specialization that
makes each kernel fast (e.g., transitive_closure doesn't need distance
tracking). **Rejected** — the Rust target took the specialized route
for good reason, and our metadata-driven pipeline makes adding kernels
cheap.

## Related work

- `docs/design/WAM_HASKELL_LOWERED_IMPLEMENTATION_PLAN.md` — the broader
  lowering architecture this fits into
- `docs/design/WAM_PERF_OPTIMIZATION_LOG.md` — current perf state
- `docs/vision/HASKELL_TARGET_ROADMAP.md` — this is Phase-3-adjacent
  (expanded lowering) but distinct — kernels bypass the interpreter
  entirely rather than lowering into native Haskell functions
