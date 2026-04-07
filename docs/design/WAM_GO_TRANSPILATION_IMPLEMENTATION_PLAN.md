# WAM-to-Go Transpilation: Implementation Plan

## Phase 0: Go Binding Registry for Prolog Builtins (PREREQUISITE)

**Goal:** Define Go equivalents for the Prolog builtins used by
`wam_runtime.pl`, so the native lowering pipeline can translate them.

**Scope:**
- Register bindings for `library(assoc)` operations → `map[string]Value`
- Register bindings for list operations → `[]Value` slices
- Register bindings for arithmetic → native Go operators
- Register bindings for `format/2` → `fmt.Sprintf()`
- Register bindings for `=../2` → `Compound` construction/destructure

**Changes:**
1. Create `src/unifyweaver/bindings/go_wam_bindings.pl`:
   ```prolog
   :- module(go_wam_bindings, [go_wam_binding/5]).

   go_wam_binding(get_assoc/3, "~map[~key]",
       [key-string, map-gomap], [value-value], [pure]).
   go_wam_binding(put_assoc/4, "~map[~key] = ~val",
       [key-string, map-gomap, val-value], [result-gomap], [mutating]).
   go_wam_binding(empty_assoc/1, "make(map[string]Value)",
       [], [result-gomap], [pure]).
   ```

2. Register these bindings with the Go target's `is_builtin_goal`
   detection so they're recognized during native lowering.

**Effort:** Medium — mechanical mapping work, parallel to Rust Phase 0.

## Phase 1: Mustache Templates for Go WAM Package

**Goal:** Define the Go package skeleton.

**Scope:**
- `templates/targets/go_wam/go.mod.mustache`
- `templates/targets/go_wam/value.go.mustache` — `Value` interface + types
- `templates/targets/go_wam/state.go.mustache` — `WamState` struct
- `templates/targets/go_wam/instructions.go.mustache` — `Instruction` interface + types
- `templates/targets/go_wam/runtime.go.mustache` — `Step` + `Run`

**Changes:**
1. Create template files in `templates/targets/go_wam/`
2. Register templates in `template_system.pl`
3. Add `compile_wam_go_package/3` to `go_target.pl` or a new
   `wam_go_target.pl` module

**Template composition strategy:**
- Larger templates (value.go, state.go) use Mustache sections
- Method bodies are filled by native lowering
- `{{step_switch_cases}}` placeholder filled by compiling `step_wam/3`
  clauses to Go type switch cases

**Go-specific template considerations:**
- Go requires all types in a package to be in `.go` files (no headers)
- Interface methods need marker methods (`valueTag()`, `instrTag()`)
- `go.mod` defines module path and Go version
- No generics needed (interface-based polymorphism sufficient)

**Effort:** Medium — more boilerplate than Rust (Go has no enums).

## Phase 2: WAM Instruction Lowering to Go

**Goal:** Compile WAM instructions from `compile_predicate_to_wam/3`
output into Go struct literals.

**Changes:**
1. Add `wam_to_go_instruction/2` to translate each WAM instruction:
   ```prolog
   wam_to_go_instruction(get_constant(C, Ai), GoCode) :-
       go_value_literal(C, GoVal),
       format(string(GoCode), '&GetConstant{C: ~w, Ai: "~w"}', [GoVal, Ai]).
   ```

2. Add `wam_to_go_code_array/2` to produce a Go slice literal:
   ```go
   var ancestorCode = []Instruction{
       &GetConstant{C: &Atom{"parent"}, Ai: "A1"},
       &Call{Pred: "parent/2", Arity: 2},
       ...
   }
   ```

3. Add `wam_to_go_labels/2` to produce label map:
   ```go
   var ancestorLabels = map[string]int{
       "clause_1": 0,
       "clause_2": 5,
   }
   ```

**Effort:** Low — mechanical translation of instruction terms.

## Phase 3: step_wam/3 Transpilation via Type Switch

**Goal:** Transpile `step_wam/3` from `wam_runtime.pl` to a Go
`Step` method using type switch dispatch.

**Strategy:**
1. Load `wam_runtime.pl` clauses for `step_wam/3`
2. Group by first-argument functor (instruction type)
3. For each group, compile the clause body via `clause_body_analysis`
4. Emit as a `case *InstructionType:` block in the type switch

**Go-specific lowering rules:**
- `get_assoc(Key, Map, Val)` → `val, ok := regs[key]; if !ok { return false }`
- `put_assoc(Key, Map, Val, NewMap)` → `regs[key] = val`
- `Val == C` → `equals(val, c)`
- `is_unbound_var(Val)` → `isUnbound(val)`
- `NPC is PC + 1` → `vm.PC++`
- `nth0(Idx, List, Elem)` → `list[idx]`
- `append(A, B, C)` → `c = append(a, b...)`

**Effort:** High — requires extending Go native lowering to handle
compound first-argument dispatch (type switch pattern). This is the
key engineering challenge.

## Phase 4: Hybrid Module Assembly

**Goal:** Produce a Go package that mixes native and WAM-compiled code.

**Changes:**
1. Add `compile_hybrid_go_package/3`:
   ```prolog
   compile_hybrid_go_package(Predicates, Options, Files) :-
       classify_predicates(Predicates, Native, WamRequired, Builtins),
       compile_native_predicates(go, Native, NativeCode),
       compile_wam_predicates(WamRequired, WamCode),
       render_go_wam_templates(Options, TemplateFiles),
       assemble_package(NativeCode, WamCode, TemplateFiles, Files).
   ```

2. Generate the entry point that routes to native or WAM:
   ```go
   func Query(pred string, args ...Value) ([]Value, bool) {
       switch pred {
       case "factorial/2":
           // Native path
           n := args[0].(*Integer).Val
           return []Value{&Integer{factorial(n)}}, true
       default:
           // WAM fallback
           vm := NewWamState(codeFor(pred), labelsFor(pred))
           for i, arg := range args {
               vm.SetReg(fmt.Sprintf("A%d", i+1), arg)
           }
           return vm.RunAndCollect(), vm.Run()
       }
   }
   ```

**Effort:** Medium — assembly and routing logic.

## Phase 5: Goroutine-Based Parallel Search

**Goal:** Use Go's concurrency for parallel choice point exploration.

**Design:**
```go
func (vm *WamState) RunParallel(maxWorkers int) <-chan []Value {
	results := make(chan []Value, 100)
	sem := make(chan struct{}, maxWorkers)
	var wg sync.WaitGroup

	var explore func(state *WamState, hasToken bool)
	explore = func(state *WamState, hasToken bool) {
		defer wg.Done()
		if !hasToken {
			sem <- struct{}{}
		}
		defer func() { <-sem }()

		for {
			if state.Halted {
				results <- state.CollectResults()
				return
			}
			// ... step and backtrack ...
			if len(state.ChoicePoints) > 0 {
				select {
				case sem <- struct{}{}:
					// Pass token to forked child to avoid race conditions
					if alt := state.ForkAtChoicePoint(); alt != nil {
						wg.Add(1)
						go explore(alt, true)
					} else {
						<-sem
					}
				default:
				}
			}
		}
	}
	// ...
}
```

**Key Features:**
- **Recursive Structural Unification:** Added `WamState.Unify()` to handle complex term unification.
- **Robust Parallelism:** Fixed race conditions in worker allocation by passing semaphore tokens.
- **Halt State:** Replaced PC sentinel with a explicit `Halted` field.

**Effort:** High — requires careful state cloning and synchronization.
**Status:** **Completed**

## Phase 5b: Order-Independent Goal Parallelism

**Goal:** Parallelize body goals of predicates declared or proven
order-independent. This is **separate from WAM choice point
parallelism** — it applies to natively-lowered deterministic predicates
whose subgoals happen to be independent.

**Declarations:**
```prolog
%% User declaration — promise that clause order doesn't matter
:- order_independent(node_score/2).
:- parallel_safe(feature_lookup/2).

%% Compiler can also prove independence via static analysis:
%% - All goals are pure (no I/O, no assert/retract)
%% - Goals bind disjoint variable sets
%% - No shared mutable state
```

**Changes:**
1. Add `order_independent/1` and `parallel_safe/1` as recognized
   directives in the clause body analysis pipeline.

2. Add `classify_parallelism/2` to determine the parallelism strategy:
   ```prolog
   classify_parallelism(Pred/Arity, goal_parallel(Goals)) :-
       single_clause_with_independent_goals(Pred/Arity, Goals), !.
   classify_parallelism(Pred/Arity, clause_parallel) :-
       is_order_independent(Pred/Arity, _), !.
   classify_parallelism(_, sequential).
   ```

3. Add `is_order_independent/2` that checks declarations first, then
   attempts static proof:
   ```prolog
   is_order_independent(Pred/Arity, declared) :-
       order_independent(Pred/Arity), !.
   is_order_independent(Pred/Arity, proven(Reasons)) :-
       all_clauses_pure(Pred/Arity),
       all_goals_bind_disjoint_vars(Pred/Arity),
       Reasons = [pure_goals, disjoint_bindings].
   ```

4. Update Go native lowering to emit `sync.WaitGroup` + goroutines
   when `classify_parallelism` returns `goal_parallel` or
   `clause_parallel`.

**Go code generation:**
```go
// goal_parallel: independent body goals run concurrently
func nodeScore(x Value) Value {
    var a, b Value
    var wg sync.WaitGroup
    wg.Add(2)
    go func() { defer wg.Done(); a = featureA(x) }()
    go func() { defer wg.Done(); b = featureB(x) }()
    wg.Wait()
    return &Integer{a.(*Integer).Val + b.(*Integer).Val}
}
```

**Interaction with Phase 5a (WAM parallel search):**
- Phase 5a: parallelizes WAM **choice point exploration** (non-determinism)
- Phase 5b: parallelizes **independent subgoals** (deterministic concurrency)
- Both use goroutines but for different purposes
- A predicate can use 5b for its body AND 5a for its choice points

**Static analysis rules for proving independence:**
- Goal `G` is **pure** if it doesn't call assert/retract, I/O predicates,
  or global state modifiers
- Goals `G1, G2` have **disjoint bindings** if the output variables of
  `G1` don't appear in the input variables of `G2` and vice versa
- A clause is **order-independent** if all its clauses produce the same
  result set regardless of evaluation order (true for set-producing
  predicates like `findall`, TC, aggregations)

**Effort:** Medium — static analysis is the hard part; code generation
is straightforward (WaitGroup + goroutines).
**Priority:** Medium — valuable without WAM, applies to many predicates.

## Phase Summary

| Phase | Description | Status | Effort | Depends On |
|-------|-------------|--------|--------|------------|
| 0 | Go WAM bindings registry | **Completed** | Medium | — |
| 1 | Mustache templates for Go package | **Completed** | Medium | — |
| 2 | WAM instruction → Go literals | **Completed** | Low | Phase 0 |
| 3 | step_wam/3 → Go type switch | **Completed** | High | Phase 0, 1 |
| 4 | Hybrid module assembly | **Completed** | Medium | Phase 2, 3 |
| 5a | Goroutine WAM parallel search | **Completed** | High | Phase 4 |
| 5b | Order-independent goal parallelism | **Completed** | Medium | Phase 0 (no WAM needed) |

## Differences from Rust Implementation Plan

| Aspect | Rust (PR #1153) | Go (this plan) |
|--------|----------------|----------------|
| Value type | `enum Value` | `interface Value` + structs |
| Dispatch | `match` expression | `switch i := instr.(type)` |
| Memory | Ownership/borrowing | GC, pointer semantics |
| Package | Cargo crate | Go module |
| Template files | `.rs.mustache` | `.go.mustache` |
| Concurrency | Not planned | Phase 5 (goroutines) |
| Bindings file | `rust_wam_bindings.pl` | `go_wam_bindings.pl` |
| Trail simplification | Need lifetime mgmt | GC handles it |
| Build validation | `cargo check` | `go build` |

## Success Criteria

A Prolog program with mixed predicate types:

```prolog
factorial(0, 1).
factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1.
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
unify_complex(f(X, g(Y)), f(a, g(b))) :- X = a, Y = b.
```

Produces a Go package that:

1. `go build` compiles without errors
2. `go test` passes for all three predicates
3. `factorial` uses native Go (direct recursion)
4. `ancestor` uses native Go (TC pattern) or WAM (if not detected)
5. `unify_complex` uses WAM (deep unification)
6. Cross-calls between native and WAM work seamlessly
