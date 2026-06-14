# Rust → F# reverse-parity brief

Date: 2026-06-12. Companion to `rust_fsharp_parity_campaign.md` (the
F#→Rust direction, completed P1–P5) — this is the verified gap list in
the OTHER direction, for when the F# stream wants it. Verified by
grepping the F# BuiltinCall arms and kernel wiring against what the
Rust target gained in P1–P5; owner guidance: the new-graph-theory
extension work waits on the graph agent, but porting established
machinery is fair game.

## 1. Graph kernels (largest gap)

F# wires only TWO kernel kinds (`category_ancestor`,
`bidirectional_ancestor`). Rust has seven. Missing in F#:

- `transitive_closure2` (DFS + visited set)
- `transitive_distance3` (BFS + distance)
- `transitive_parent_distance4`, `transitive_step_parent_distance5`
- `weighted_shortest_path3` (Dijkstra, Rust uses BinaryHeap —
  F# equivalent: sorted set or a simple priority list)
- `astar_shortest_path4` (goal-directed, dimensionality-aware
  heuristic)

The shared registry (`recursive_kernel_detection.pl`) already carries
detectors, register layouts, and `kernel_native_call` specs for ALL of
these — F# needs the `executeForeign` dispatch arms and the native
kernel functions (model on Rust `compile_collect_native_*_to_rust`
emitters at `wam_rust_target.pl:2251+` and on the existing F#
category_ancestor wiring at `wam_fsharp_target.pl:3501+`).

## 2. Meta-predicate family (Rust P5)

Absent from F#: `maplist/2..5`, `include/3`, `exclude/3`,
`partition/4`, `foldl/4`, `foldl/5`. F# already has the meta-call
substrate these need — `runGoalInChild` inside the catch/3
implementation (`wam_fsharp_target.pl:2540+`) is the analog of the
Rust `call_goal_value` these were built on in P5. Port the P5 design:
per-element first-solution calls, fresh-variable lists for unbound
list arguments, trial-and-unwind for include/exclude.

## 3. Builtins (Rust P3/P5)

Absent from F# (verified against its BuiltinCall arms):

- `between/3` (check mode + enumeration via choice point)
- `plus/3` (all three modes)
- `keysort/2` (stable; F# has sort/msort/compare already)
- `sum_list/2`, `sumlist/2`, `max_list/2`, `min_list/2`
- `atomic_list_concat/2`, `atomic_list_concat/3` (join + split)
- `char_type/2` (+Char mode)
- `ground/1`

Present in F# already (do NOT port): sort/msort/compare, term-order
ops, nth0/nth1, last, memberchk, select, delete, numlist, the
atom/string text family, succ (with ISO variants — F# is AHEAD here),
catch/throw, functor/arg/univ/copy_term.

## 4. Smaller items

- Bench-harness calibration hoisting (Rust P1.5): check whether the F#
  bidirectional benchmark loop in `program.fs.mustache` re-runs
  `calibrateGraph` per seed; Rust hoists it out of the seed loop
  (graph and root are loop constants).
- F# ISO-variant builtins (`is_iso`, `succ_iso`, comparison ISO/lax
  family) are an F#→Rust gap left OPEN in the forward campaign —
  noting here so the two briefs stay symmetric.

## Status of the related correctness work (this session)

The non-cyclic aliasing divergence flagged in the bind-through sweep
(side finding 4) was probed and CONFIRMED REAL in both F# and R
(`aliasfree(X) :- mk1(g(7), _W), var(X)` wrong-failed), and fixed in
both: the bind-through is now gated on the register class IN ADDITION
to the cycle check (`fsharp_wam_bindings.pl` addToBuilder BuildStruct
+ BuildList arms; `runtime.R.mustache` build finalize). With this, all
16 targets carry a register-class guard or are structurally immune.
