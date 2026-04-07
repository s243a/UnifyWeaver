# WAM Haskell Transpilation: Implementation Plan

## Phase 0: Infrastructure (Current)

- [x] Create `wam_haskell_target.pl` scaffold
- [x] Define Haskell data types (Value, WamState, ChoicePoint, Instruction)
- [x] WAM instruction → Haskell expression translations
- [x] Backtrack and run loop in Haskell
- [x] Project generation (cabal file, module structure)
- [x] Design docs (philosophy, specification, this plan)

## Phase 1: Compile and Run a Simple Predicate

**Goal:** Generate a working Haskell project that compiles and runs `dimension_n(5)`.

### Tasks
- [ ] Implement `compile_wam_predicate_to_haskell/4` — convert WAM text to
      Haskell instruction list and label map
- [ ] Generate `Main.hs` that creates a WamState, loads the predicate, runs it
- [ ] Verify: `cabal run` produces correct output
- [ ] Add `step` function that dispatches on Instruction constructors

### Acceptance Criteria
- `cabal build` succeeds with no errors
- `cabal run` with `dimension_n(X)` query returns `X = 5`

## Phase 2: category_ancestor/4 with Facts

**Goal:** Run the effective-distance workload through the Haskell WAM.

### Tasks
- [ ] Implement fact loading from TSV files into `Map.Map String [String]`
- [ ] Generate `SwitchOnConstant` as `Map.lookup` dispatch
- [ ] Implement all WAM instructions needed for `category_ancestor/4`:
  - `get_constant`, `get_variable`, `get_value`
  - `put_constant`, `put_variable`, `put_value`
  - `put_structure`, `put_list`, `set_value`, `set_constant`
  - `allocate`, `deallocate`, `call`, `proceed`
  - `try_me_else`, `retry_me_else`, `trust_me`
  - `builtin_call` for `!/0`, `is/2`, `length/2`, `</2`, `\+/1`
- [ ] Implement `backtrack` with Data.Map reference swap
- [ ] Implement `\+/1` fast path for `member/2`
- [ ] Benchmark driver: load TSV facts, iterate seeds, compute effective distances

### Acceptance Criteria
- Haskell benchmark produces 213/213 tuples matching Prolog reference
- At least 270/272 exact line matches
- Runtime < 10s at 300 scale (vs Rust's 116s, Prolog's 338ms)

## Phase 3: Benchmark Integration

**Goal:** Wire into the existing benchmark harness.

### Tasks
- [ ] Create `generate_wam_haskell_effective_distance_benchmark.pl`
- [ ] Add `wam-haskell-accumulated` target to `benchmark_effective_distance.py`
- [ ] Add `wam-haskell` to `available_targets` in `benchmark_common.py`
- [ ] Run comparison: prolog-accumulated vs wam-rust-accumulated vs wam-haskell-accumulated

### Acceptance Criteria
- `python3 benchmark_effective_distance.py --targets wam-haskell-accumulated --scales 300,1k`
  produces matching output
- SHA256 digest matches Prolog reference

## Phase 4: Performance Optimization

**Goal:** Achieve <1s at 300 scale (competitive with SWI-Prolog's 338ms).

### Tasks
- [ ] Profile with GHC's cost-centre profiling (`-prof -fprof-auto`)
- [ ] Identify hot paths and optimize:
  - Use `Data.IntMap` for registers (integer keys faster than string keys)
  - Use `Data.HashMap.Strict` if hashing is faster than balanced tree
  - Strictness annotations to prevent thunk accumulation
  - Unbox `Int` fields in WamState and ChoicePoint
- [ ] Consider compiling WAM predicates to direct Haskell functions
      (not instruction arrays) for zero dispatch overhead
- [ ] Benchmark at 1k, 5k, 10k scales

### Acceptance Criteria
- 300 scale: <1s
- 1k scale: <2s
- Output identical to Prolog at all scales

## Phase 5: Native Lowering Integration

**Goal:** Use WAM only as fallback, with native lowering handling most predicates.

### Tasks
- [ ] Identify which predicates `haskell_target.pl` can already handle
- [ ] For the effective-distance workload, determine the minimal set that
      needs WAM fallback
- [ ] Implement hybrid: native Haskell for facts/simple predicates,
      WAM-compiled Haskell for complex backtracking predicates
- [ ] Compare hybrid vs pure-WAM performance

### Acceptance Criteria
- Hybrid produces identical output to pure-WAM
- Hybrid is faster (native lowering avoids WAM overhead for simple predicates)

## Risk Register

| Risk | Mitigation |
|------|-----------|
| GHC's lazy evaluation causes space leaks | Strict `Data.Map.Strict`, `!` annotations, `-O2` |
| Haskell list-based stack is slow | Profile first; consider `Data.Sequence` if needed |
| `Data.Map` O(log n) per access too slow | Benchmark; try `Data.IntMap` or `Data.HashMap` |
| WAM instruction dispatch overhead | Compile predicates to direct functions (Phase 4) |
| GHC not available on user's system | Provide pre-compiled binaries or fallback to Rust WAM |

## Dependencies

- GHC 8.10+ (for `Data.Map.Strict`, `StrictData`)
- `containers` package (for `Data.Map`, `Data.IntMap`)
- `cabal-install` or `stack` for project management
- Existing: `wam_target.pl` for WAM compilation, `haskell_target.pl` for native lowering
