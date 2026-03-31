# CI Test Strategy Proposal

## Problem Statement

Some of the current CI failures are exposing tests that are too tightly coupled to planner or code-generation internals rather than to stable functional behavior.

Recent example:

- the C# query-target accumulation plan test failed after the planner began attaching richer `base_expression` and `recursive_expression` fields
- lowering was still correct
- generated runtime shape was still correct
- the failure came from exact dict equality against an internal plan node

This kind of test is useful in a narrow sense, but it becomes brittle when implementation details evolve while semantics remain unchanged.

## Goal

Shift CI toward a clearer testing pyramid:

1. functional semantics as the primary contract
2. generated runtime or API shape as the secondary contract
3. planner-internal structure checks only where the internal structure is itself the optimization contract

The objective is not to remove structural tests. It is to use them more intentionally, so normal implementation refinement does not create noisy CI failures.

## Proposed Test Layers

### 1. Functional Semantics Tests

These should be the most important tests in CI.

They should verify:

- result rows
- parameterized behavior
- recursion semantics
- aggregation semantics
- path semantics
- cross-target agreement where targets are expected to implement the same meaning

Examples:

- per-path visited recursion should reject repeated nodes in a path
- parameterized recursive queries should return the same subset as the unseeded form filtered by seed
- effective-distance style workloads should preserve path multiplicity when that is the intended semantics

These tests should generally avoid depending on:

- exact planner dict contents
- exact join ordering unless join ordering is the feature being tested
- exact formatting of emitted code

### 2. Runtime/API Shape Tests

These verify stable generated artifacts without asserting every implementation detail.

They should verify things like:

- specialized node selection such as `PathAwareTransitiveClosureNode`
- parameter seeding behavior
- selected target-specific runtime path
- emitted public-facing method or node structure where that structure is part of the supported contract

These tests are useful because they catch optimizer regressions that functional tests alone may not identify quickly.

But they should focus on required properties, not exhaustive structural equality.

Preferred style:

- assert node `type`
- assert required plan fields
- assert generated source contains the intended runtime node
- assert absence of clearly wrong fallback paths where that matters

Avoid:

- full dict equality for planner nodes that may gain additional metadata
- matching large emitted code blocks when a few stable substrings would do

### 3. Planner-Internal Structure Tests

These should exist, but be explicitly marked as implementation-sensitive.

They are appropriate when the planner shape is itself the feature being protected, for example:

- a join-ordering heuristic
- a pushdown optimization
- a grouping-key propagation rule
- a bounded recursive lowering that must compile to a specific internal node shape

These tests should be:

- narrow
- few in number
- clearly documented as structural tests

## Classification Guidance

When adding or revising a test, classify it first.

### Use a Functional Test When

- the behavior visible to a user or downstream target is what matters
- multiple internal implementations would be acceptable
- the query result is the real contract

### Use a Runtime/API Shape Test When

- the specialization itself matters
- there is a stable generated artifact worth protecting
- fallback to a generic path would be a real regression

### Use a Planner-Internal Test When

- the planner structure directly encodes the intended optimization contract
- the test is protecting a specific heuristic or invariant
- a more external test would not detect the regression early enough

## Migration Rules

### Rule 1: Prefer Required-Field Checks Over Exact Dict Equality

Instead of:

- exact equality on a full `root` dict

Prefer:

- `type`
- required relations
- required mode metadata
- required expression or grouping fields only where those are part of the contract

### Rule 2: Prefer Representative Output Checks Over Large Structural Assertions

Instead of asserting an entire emitted code region, check:

- selected node name
- parameter indices
- presence or absence of known fallback mechanisms
- representative runtime output rows

### Rule 3: Keep Cross-Target Semantic Suites Small but Strong

Add a compact set of shared semantics cases that are expected to agree across targets.

Candidate areas:

- counted closure
- per-path visited recursion
- weighted accumulation
- parameterized recursive reachability
- grouped recursive propagation

### Rule 4: Mark Implementation-Sensitive Tests Explicitly

Structural tests should include a short comment saying why the exact structure matters.

That makes future breakage easier to interpret and reduces accidental overuse of strict assertions.

## Immediate Opportunities

### C# Query Target

Good immediate candidates for cleanup:

- exact plan-node dict matches that can be replaced with required-key checks
- recursive plan tests that currently prove implementation shape but not runtime semantics strongly enough
- cases where generated source matching is too broad or too exact

The recent `path_aware_accumulation` failure is the model case:

- keep the node-type check
- keep the required relation checks
- keep the emitted runtime-node check
- avoid exact equality on the whole planner dict

### Rust and Other Targets

As native lowering becomes richer across targets, the same pattern will recur.

For Rust, Go, C#, and future target work, CI should prefer:

- semantic agreement tests
- target-runtime selection tests

over:

- strict matching of every internal lowering detail

## Stable vs Internal Planner Fields

We should document which planner fields are intended to be stable.

Suggested categories:

### Stable or Semi-Stable

- root node `type`
- selected relation predicates
- mode metadata where externally relevant
- grouping keys where part of semantics
- seed positions or parameterization metadata

### Internal or Advisory

- extra expression annotations
- helper metadata used only during rendering
- optimizer bookkeeping
- advisory width or depth metadata unless directly consumed by runtime behavior under test

This distinction can guide future test design.

## Suggested Rollout

### Phase 1

- adopt this classification for new tests
- stop introducing new exact-equality planner assertions unless justified

### Phase 2

- revise the most brittle C# query-target structural tests
- add a few stronger functional recursion cases

### Phase 3

- identify cross-target semantic fixtures that should agree
- reuse those fixtures across C#, Rust, and DFS comparisons where possible

### Phase 4

- document per-target expectations for optimization-shape tests
- keep those tests small and explicitly labeled

## Expected Benefits

- fewer noisy CI failures from harmless implementation refinement
- stronger protection for actual semantics
- clearer signal when an optimization regresses
- easier cross-target development as lowering generality increases

## Open Questions

- which planner fields should be formally treated as stable contracts
- whether some benchmark-derived semantic cases should be promoted into mandatory CI fixtures
- whether cross-target semantic fixtures should live in a shared helper module rather than target-specific test files

## Recommendation

Adopt the three-layer test strategy and begin by migrating brittle planner-equality tests in the C# query suite to required-field checks plus functional verification.

That gives better CI signal immediately, without reducing coverage of the optimizations we actually care about.
