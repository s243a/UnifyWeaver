# T7 2b.2 — fork analysis (why the work-split is a VM-level change)

**Status:** acceptance harness built + green (`tests/test_wam_rust_aggregate_exec.pl`);
the work-splitting fork itself is scoped here as the next, larger step — it is a
VM-level change, not a "wire `gated_collect` in" call, and this report records why.

## What 2b.2 looked like vs. what it is

From the outside, 2b.2 reads as "have the parallel-eligible path call
`gated_collect`." Reading the generated interpreter (the `BeginAggregate` /
`EndAggregate` arms, `backtrack()`, and the `aggregate_frame` finalisation in
`wam_rust_target.pl`) shows the real shape:

- `BeginAggregate` clears `aggregate_acc` and pushes an `aggregate_frame` choice
  point.
- The generator runs; **each solution reaches `EndAggregate`, which pushes the
  value into `aggregate_acc` and calls `backtrack()`** to get the next.
- When the generator is exhausted, `backtrack` pops the `aggregate_frame` CP and
  finalises (sum/count/collect/max/min over `aggregate_acc`).

So the per-branch work (the expensive part the cost gate flags) happens **inside
the backtracking search, interleaved** between Begin and End. `gated_collect`
needs a `run_branch(&mut clone, k)` that runs *only* outer-branch k on a clone —
but the interpreter has **no clean seam** between "produce the next outer branch"
and "run the body for it":

- Outer alternatives are discovered **sequentially via backtracking** (a
  `try/retry/trust`-style CP chain), not known up front, so you cannot hand
  worker *k* "branch k" without first walking branches `0..k`.
- Skipping a non-assigned branch *before* its body runs would require
  identifying the generator/body boundary at runtime — which the WAM doesn't
  expose; the conjunction `(Enum, Body)` is compiled to one undivided block
  (`compile_aggregate_all` in `wam_target.pl`).

Doing it at the choice-point level (a per-worker stride filter on the outer CP)
is possible but **fragile**: it depends on exact CP-depth bookkeeping across
nested body choice points, trail/stack restoration across the clone boundary,
and order-preserving merge for `collect`. High risk of subtly-wrong generated
code — the one place not to rush.

## The two real implementation routes

1. **Compile-time generator/body split (recommended).** At
   `compile_aggregate_all`, split `InnerGoal` into `(Enum, Body)`, compile each
   as its own entry, and for parallel-eligible aggregates emit: enumerate `Enum`
   → `Vec<binding>` (cheap), then `gated_collect` mapping `Body` over the
   bindings on cloned machines, merging into `aggregate_acc`. This maps directly
   onto the substrate's `run_branch(&mut clone, k)` and keeps the parallelism out
   of the core step-loop. Cost: real compile-time work (split detection — only
   sound when `Enum` and `Body` share a clean variable frontier; two sub-programs;
   binding collection) and a Rust-side parallel driver. Order is recoverable by
   tagging bindings with their index.

2. **Interpreter choice-point partition (not recommended).** Keep one block;
   give each cloned worker a stride filter on the outer CP. Less compile-time
   work, but the fragile VM bookkeeping above.

## Recommendation

Land the foundations now — the **forkable machine** (`WamState: Clone + Send`,
2b.1) and this **exec acceptance harness** (the baseline `parallel == sequential`
any fork must preserve) — and do the work-split as a **focused, test-first
follow-up** along route 1. The harness is already the acceptance test: it builds
and runs count/sum/collect end-to-end and asserts the exact results; the fork is
done when the same harness passes with the parallel driver forced on.

## What is green today

- `WamState: Clone + Send`, compile-asserted in generated projects (2b.1, #3041).
- The aggregate path executes correctly on Rust: `aggregate_all(count)` → 3,
  `aggregate_all(sum)` → 6, `findall` → `[1,2,3]`, verified via `cargo test`
  (`tests/test_wam_rust_aggregate_exec.pl`).
