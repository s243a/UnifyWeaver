# WAM Clojure LMDB Desktop Transition

## Purpose

The Clojure hybrid WAM target has reached the point where additional
high-value LMDB work is no longer primarily about wiring. The remaining
questions are mostly:

- which cache policy actually helps
- whether the JVM/JNI reader seam behaves well under desktop load
- whether the current LMDB path is the right default for large workloads
- how the Clojure LMDB path compares to the Haskell reference on real
  hardware

Those are desktop-environment questions, not Termux questions.

This proposal defines the transition from **Termux-side implementation**
to **desktop-side measurement and policy selection**.

## Current State

The Clojure LMDB path already has:

1. JVM/JNI LMDB reader seam
2. thread-local reader reuse
3. L1 memoization
4. shared and two-level cache modes
5. cache stats seam
6. benchmark-side LMDB integration for `category_parent`
7. target-level LMDB foreign relation support for:
   - `category_parent/2`
   - `category_ancestor/4`
8. generated-project packaging of:
   - `lmdb-artifact-reader.jar`
   - `liblmdb_artifact_jni.so`
9. Termux-stable isolated Cargo target handling for repeated LMDB
   artifact builds

The work that remains is therefore not “make LMDB possible.” It is
“decide which LMDB mode is worth using and under what conditions.”

## Why Transition Now

### Termux has served its purpose

Termux has been useful for:

- code generation
- correctness checks
- JNI feasibility
- LMDB build/round-trip validation
- small-scale benchmark smoke tests

That work is largely done.

### Termux is now the wrong environment for the next questions

The remaining questions are sensitive to:

- JVM startup noise
- scheduler behavior
- mmap/page-cache behavior
- contention characteristics of shared caches
- repeated JNI/LMDB lookup throughput
- larger working sets

Those measurements are not trustworthy enough on this Termux setup to
select defaults or make strong architecture decisions.

### The Haskell path already showed the pattern

Recent Haskell LMDB work established the right order:

1. raw reader seam
2. thread-local reuse
3. L1 cache
4. L2/two-level policies
5. desktop-scale measurement

The Clojure target has now reached the same stage. Continuing to add
surface area without desktop measurement risks cargo-culting policy
instead of selecting it from data.

## Philosophy

The next phase should optimize for **decision quality**, not for
continued feature count.

That means:

1. stop adding new LMDB relation surfaces unless a concrete workload
   requires them
2. measure the policies we already have
3. use stats and desktop timings to choose defaults
4. keep benchmark-specific and target-level wiring stable while the
   measurement phase runs

The key change in mindset is:

- **Termux phase**: prove the mechanism
- **desktop phase**: choose the policy

## Scope of Desktop Phase

### In scope

1. Compare Clojure LMDB cache policies:
   - `none`
   - `memoize`
   - `shared`
   - `two_level`

2. Measure both:
   - overlap-heavy workloads
   - low-overlap workloads

3. Capture machine-readable stats from the existing cache seam:
   - local hits
   - shared hits
   - misses

4. Compare Clojure LMDB against:
   - Clojure sidecar/artifact fallback
   - Haskell LMDB reference path where useful

5. Determine:
   - best default for desktop
   - whether Termux should keep a different default
   - whether any mode should remain opt-in only

### Out of scope

1. adding more LMDB-backed relations
2. new cache tiers beyond `two_level`
3. speculative parallel LMDB pointer sharing designs
4. broad runtime refactors unrelated to policy measurement

## Required Deliverables

The desktop phase should produce:

1. a reproducible benchmark command set
2. a small result matrix covering all four cache modes
3. captured cache stats for those runs
4. a recommendation:
   - keep current default
   - switch default
   - keep policy workload-dependent only

5. a follow-up implementation decision if needed:
   - default change
   - memory-budget cap
   - stats/reporting refinement
   - no change

## Proposed Benchmark Matrix

Minimum comparison set:

1. `clojure-wam-seeded`
2. `clojure-wam-seeded-artifact`
3. LMDB `none`
4. LMDB `memoize`
5. LMDB `shared`
6. LMDB `two_level`

At minimum, run:

- `dev`
- one intermediate scale
- one larger scale that is meaningful on desktop

Each run should use multiple repetitions.

## Reporting Requirements

The next implementation step before desktop measurement should be a
small reporting surface that emits machine-readable LMDB policy and
cache stats, not just stderr debug strings.

That reporting should include:

- selected policy
- relation name
- local hits
- shared hits
- misses
- total runtime
- scale
- benchmark variant

This is the last useful code-first step before real desktop tuning.

## Exit Criteria

The desktop transition phase is complete when all of the following are
true:

1. all four cache modes were measured on desktop
2. cache stats were recorded for those runs
3. a concrete recommendation exists for Clojure LMDB policy defaults
4. the recommendation is documented
5. either:
   - a default change is implemented
   - or a decision is made to keep the current explicit/opt-in behavior

## Immediate Next Step

Before moving to another target, do one small follow-up on the Clojure
side:

1. add machine-readable benchmark reporting for LMDB policy and cache
   stats

Then stop Clojure LMDB feature work and move the decision-making phase
to desktop.

## Summary

The Clojure LMDB path is no longer blocked on implementation.

It is blocked on **measurement quality**.

That is the right point to transition this work from Termux-driven
development to desktop-driven validation.
