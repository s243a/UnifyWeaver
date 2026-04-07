# WAM Indexed Architecture: Implementation Plan

## Benchmark Results (Phases 1-3)

All 386 seeds at 300 scale through the Prolog WAM emulator with indexed fact tables:

| Metric | Native Prolog | WAM Indexed | Improvement vs Unindexed |
|--------|--------------|-------------|--------------------------|
| Runtime (386 seeds) | 295ms | 1,953ms (6.6x) | from timeout (>30min) → 2s |
| Per-seed (Physicists) | 17ms | 3ms | 4197ms → 3ms (1400x) |
| Per-seed (Nuclear_physics) | 0ms | 4ms | 1661ms → 4ms (415x) |
| Tuples found | 213 | 25 | Cut barrier limits multi-parent paths |
| Total paths | 11,172 | 25 | Same cut barrier issue |

The 25/213 tuple gap is from `!/0` clearing too many choice points (not
performance). The indexed architecture correctly finds all single-parent
paths. Multi-parent paths require Phase 4 (cut barrier fix in three-layer
execution).

## Phase 0: Design and Documentation

- [x] Philosophy document
- [x] Specification document
- [x] Implementation plan (this document)
- [ ] Review with Perplexity for feedback

## Phase 1: Portable Dictionary Abstraction

**Goal:** Define `dict_*` operations in `wam_runtime.pl` that transpilers map
to native types.

### Tasks
- [ ] Create `wam_dict.pl` module with `dict_new/1`, `dict_lookup/3`,
      `dict_insert/4`, `dict_from_list/2`, `dict_to_list/2`, `dict_keys/2`
- [ ] Implement using SWI-Prolog `assoc` as default backend
- [ ] Add `dict_lookup_or_fail/3` (fails if key not found) and
      `dict_lookup_default/4` (returns default)
- [ ] Write tests for dictionary operations
- [ ] Document target mappings (Haskell `Data.Map`, Rust `HashMap`, etc.)

### Acceptance Criteria
- All existing WAM E2E tests pass with `dict_*` replacing direct `assoc` calls
- Dictionary tests cover insert, lookup, missing key, overwrite, conversion

## Phase 2: Fact Table Separation

**Goal:** Facts are loaded into indexed tables instead of compiled to WAM
instruction chains.

### Tasks
- [ ] Add `fact_tables` field to WAM state (dict of predicate → fact table)
- [ ] Create `load_facts/3` — takes a predicate, finds all `clause(Head, true)`
      facts, builds indexed dict grouped by first argument
- [ ] Update `call` instruction handling: if target is a fact predicate,
      dispatch to `call_fact/N` instead of jumping to WAM code
- [ ] Implement `call_fact/N` with first-argument lookup, multi-value iteration,
      and choice point for backtracking through matches
- [ ] Remove fact predicates from WAM compilation in `wam_target.pl` —
      they should be excluded from `compile_predicate_to_wam/3` output
- [ ] Update `prepare_code` to accept fact tables alongside WAM code

### Acceptance Criteria
- `category_parent(a, X)` resolves via table lookup, not WAM instructions
- Code size drops from ~24K instructions to ~50 (only rules)
- Existing tests pass (fact predicates work identically)
- Performance: 1-hop query on 6009 facts completes in <10ms

## Phase 3: Rule Head Separation

**Goal:** Rule heads are matched natively (pattern match / conditional),
only bodies remain as WAM instructions.

### Tasks
- [ ] Analyze rule heads to extract constant/variable patterns per argument
- [ ] Generate dispatch code: `match A3 == 1 → body1 ; → body2`
- [ ] Remove `get_constant`, `get_variable`, `get_value` from body code
- [ ] Remove `allocate`/`deallocate` from body code — handle at dispatch level
- [ ] Remove `try_me_else`/`retry_me_else`/`trust_me` — clause selection
      handled by rule dispatcher
- [ ] Body code starts after the last head instruction, ends at `proceed`

### Acceptance Criteria
- `category_ancestor/4` compiles to 2 rule entries with body-only WAM code
- Body code is ~15 instructions per clause (not ~25 with head matching)
- Clause dispatch is a native conditional, not WAM instruction chain

## Phase 4: Update WAM Runtime for Three-Layer Execution

**Goal:** The `step_wam` / `run_loop` handles the three layers cleanly.

### Tasks
- [ ] Refactor `run_loop` to check: fact table → rule index → WAM code
- [ ] `call(P/N)` dispatches:
  1. `dict_lookup(P/N, FactTables)` → `call_fact`
  2. `dict_lookup(P/N, RuleIndex)` → match heads, execute body
  3. `dict_lookup(P/N, Labels)` → old-style WAM code (fallback)
- [ ] Implement backtracking for fact iteration (choice points per match group)
- [ ] Implement backtracking for rule selection (choice points per matching clause)
- [ ] Ensure cut barrier works across all three layers

### Acceptance Criteria
- Full `category_ancestor` benchmark works through the three-layer dispatch
- Prolog WAM emulator runs 300-scale benchmark to completion
- Results match native SWI-Prolog (213/213 tuples)

## Phase 5: Transpiler Updates

**Goal:** Target transpilers generate native code for all three layers.

### Tasks
- [ ] Update `wam_haskell_target.pl`:
  - Fact tables → `Data.HashMap String [tuple]`
  - Rule dispatch → Haskell function with pattern matching
  - Body code → existing WAM instruction translation
- [ ] Update `wam_rust_target.pl`:
  - Fact tables → `HashMap<String, Vec<tuple>>`
  - Rule dispatch → Rust match expression
  - Body code → existing WAM instruction translation
- [ ] Benchmark Haskell target with indexed architecture
- [ ] Benchmark Rust target with indexed architecture
- [ ] Compare against native SWI-Prolog

### Acceptance Criteria
- Haskell benchmark: <1s at 300 scale, 213/213 tuples
- Rust benchmark: <5s at 300 scale, 213/213 tuples (limited by backtracking)
- All targets produce identical output

## Phase 6: Optimized Prolog Integration

**Goal:** Use the optimized Prolog predicates (seeded accumulation, branch
pruning) through the indexed WAM architecture.

### Tasks
- [ ] Generate WAM code for the optimized `category_ancestor$*` helper predicates
- [ ] Load the generated helpers through the indexed architecture
- [ ] Compare: raw `category_ancestor` WAM vs optimized helper WAM vs native Prolog
- [ ] Identify which optimized helpers can be fully natively lowered (no WAM needed)

### Acceptance Criteria
- Optimized helpers run through WAM with correct results
- Performance is within 5x of native SWI-Prolog for the optimized workload

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Fact table loading too slow for large datasets | Startup cost | Build tables lazily or at compile time |
| Rule head analysis too complex for general predicates | Some predicates can't be separated | Fallback to full WAM compilation |
| Backtracking across fact/rule/body boundary errors | Wrong results | Extensive testing with choice-point-heavy workloads |
| Transpiler output doesn't match three-layer semantics | Target-specific bugs | Use Prolog emulator as test oracle |
| `dict_*` abstraction too leaky for some targets | Performance regression | Allow target-specific overrides |

## Dependencies

- Existing: `wam_target.pl` (WAM compilation), `wam_runtime.pl` (WAM emulator)
- Existing: `haskell_target.pl`, `wam_haskell_target.pl`, `wam_rust_target.pl`
- New: `wam_dict.pl` (dictionary abstraction)
- Testing: WAM E2E tests, effective-distance benchmark
