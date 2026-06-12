# T7 (parallel / Tier-2) on Rust — design validated by benchmark; gating is the crux

**Verdict: worth building, but only with an adaptive gate — which is now
designed and benchmark-validated.** Parallel fan-out of a forkable aggregate is
a real 2–3.7× speedup in Rust on expensive workloads, *catastrophically*
regresses cheap workloads if applied unconditionally (5–200× slower), and a
model-based runtime gate recovers best-of-both with no material regression. This
report records the evidence and the concrete build plan against the existing
Rust aggregate substrate. (Requested check: "make sure this is actually a
performance gain because Rust already has parallel capabilities" — it is, but
narrowly, and only behind the gate.)

## What T7 is, and the WAM-specific cost

T7 parallelises the independent solution-generating branches of a *forkable
aggregate* — `findall/3` / `aggregate_all/4` whose generator enumerates
independent branches (e.g. `findall(X, (member(Y,List), heavy(Y,X)), Xs)`).

"Rust already has parallelism" (rayon, threads) does **not** make this free: a
WAM machine is mutable shared state (heap, trail, registers, stack), so you
cannot `par_iter` over one machine. Each parallel branch must run on its **own
cloned machine**. That per-branch state fork is the cost sequential
backtracking avoids, and it dominates for cheap branches.

## Benchmark evidence (rustc -O, 4 cores)

Model: a `Machine { heap, trail, regs }`; sequential-backtrack `findall` (one
reused machine) vs parallel `findall` (`std::thread::scope`, machine forked,
results merged); per-branch work in synthetic ops.

### 1. Parallel is a real win — but only for expensive branches

| branches | work/branch | seq | par | speedup |
|---:|---:|---:|---:|---:|
| 1024 | 10 ops | 22µs | 256µs | 0.09× |
| 1024 | 200 ops | 449µs | 442µs | ~1× (break-even) |
| 1024 | 5 000 ops | 11.0ms | 3.36ms | 3.29× |
| 1024 | 100 000 ops | 224ms | 62.7ms | 3.52× |

Cheap branches (≤ ~200 ops — typical Prolog clause bodies) lose badly; the win
appears only past ~thousands of ops/branch, approaching the core-count ceiling.

### 2. Per-thread (chunked) cloning beats per-branch cloning

Clone the machine **once per thread** and backtrack between branches within the
chunk, rather than cloning per branch. This amortises the fork cost over the
chunk and is strictly better for large machine state (for a small heap the
~100µs thread-pool spawn dominates either way, so they tie). Chunked is the
correct substrate form — "the parallel solution that is better for small
workloads," as the gate's parallel arm.

### 3. A model-based adaptive gate gives best-of-both

Gate: run a small **probe** (≈8 branches) sequentially, measure per-branch cost,
then compare estimated costs —
`est_seq = per·n` vs `est_par = per·n / cores + pool_overhead` (pool_overhead
measured once, ≈100µs for 4 threads) — and fan out only when
`est_par · margin < est_seq` and there are ≥ cores branches left. Result across
the full work×branch grid:

- **No catastrophic regression.** The unconditional-parallel 5–200× slowdowns
  are gone; cheap workloads correctly stay sequential.
- **Clear wins captured** (work ≥ ~1 000 ops with enough branches): 2–3.7×.
- The only residual cost on trivial workloads is the probe's own ~10µs timing
  overhead on a ~20µs operation — negligible in absolute terms and removable by
  skipping the probe entirely when `n < cores·min_chunk`.

This validates the design choice: **the adaptive probe is mandatory, not
optional polish** — it is the component that prevents auto-parallelisation from
ever making real code slower, and it is what distinguishes "small vs large
branching workload" at runtime.

## Build plan (grounded in the existing Rust aggregate substrate)

The Rust WAM target already has the aggregate machinery to hook into:

- `Instruction::BeginAggregate(agg_type, value_reg, result_reg)` pushes an
  `aggregate_frame` choice point and clears `self.aggregate_acc`.
- `Instruction::EndAggregate(value_reg)` pushes the current solution into
  `aggregate_acc` and `backtrack()`s to enumerate the next — i.e. aggregates run
  by **sequential backtracking** today.

T7 intercepts that enumeration:

1. **Forkable detection.** At `BeginAggregate`, the generator's *outer* choice
   point (the branch enumerator, e.g. `member/2` over a list) is the fork point.
   Gate: only fork when the branches are independent (no shared mutable binding
   escaping the aggregate — the same witness-variable condition Elixir's
   `par_wrap_segment/4` checks) and `agg_type ∈ {findall, aggregate_all}`.
2. **Probe + decide.** Run the first ≈8 branches sequentially (the normal
   backtracking path), timing them; apply the model gate above. If it says
   sequential, continue the existing backtracking loop unchanged (zero new risk).
3. **Fork + merge.** If parallel: snapshot the machine at the fork choice point,
   spawn `cores` worker threads (`std::thread::scope`), give each a **clone** and
   a chunk of the remaining branches; each runs its branches' continuations to
   `EndAggregate`, collecting solutions into a local `Vec`; merge all locals into
   `aggregate_acc` (order then normalised as the sequential path would, since
   findall/aggregate order is by generator order — stable-merge by branch index).
4. **Adaptive carry-over.** Cache the per-branch estimate on the aggregate frame
   so a repeated aggregate (e.g. in a loop) skips re-probing — matching Elixir's
   `:go_parallel` carry-over.

Reference: Elixir's `wam_elixir_target.pl` (`par_wrap_segment/4`,
`in_forkable_aggregate_frame?`, `merge_into_aggregate/2`, the `:go_parallel`
probe). The only genuinely new Rust-side piece is the copy-on-fork of the
machine at the choice point + the `thread::scope` worker pool; everything else
mirrors the validated gate.

### Most promising refinement

If the generator can be made to emit a **plain value payload** per branch
(rather than threading through the full heap), the fork copies a small payload
instead of the whole machine — moving the break-even down substantially and
widening T7's applicable range well past expensive-only workloads.

## Status

Design + cost model + gate are benchmark-validated and the hook points exist.
The implementation is a substantial, semantics-changing runtime change
(machine-fork + worker pool + gate + forkable-independence analysis + a
concurrency exec harness proving parallel result-set == sequential and showing
speedup) — the next concrete step, not yet built.
