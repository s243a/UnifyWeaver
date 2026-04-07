# WAM Indexed Architecture: Philosophy

## Core Insight

The WAM instruction set conflates two fundamentally different operations:

1. **Data lookup** — finding which facts match a query (`category_parent(a, X)`)
2. **Logic execution** — evaluating clause bodies with backtracking, cut, arithmetic

Currently, facts are compiled into WAM instruction chains (`try_me_else` /
`get_constant` / `retry_me_else` / ...) that the VM steps through one instruction
at a time. With 6009 `category_parent` facts, this generates ~18,000 WAM
instructions that must be traversed even with `switch_on_constant` indexing.

**Facts are data. They should be stored in indexed tables and looked up in O(1),
not executed as instruction chains.**

## The WAM Runtime as Transpilation Source

The Prolog WAM emulator (`wam_runtime.pl`) is not just a test oracle — it is the
**reference implementation** that gets transpiled to target languages:

```
wam_runtime.pl → wam_haskell_target.pl → Haskell runtime
               → wam_rust_target.pl    → Rust runtime
               → wam_elixir_target.pl  → Elixir runtime
```

This means:
- Data structure choices in the Prolog emulator **directly determine performance
  in all targets**. An O(log n) assoc in Prolog becomes O(log n) `Data.Map` in
  Haskell. An O(1) hash-based lookup becomes `Data.HashMap` / `HashMap` / `Map`.
- The emulator must use **portable abstractions** that map cleanly to every target
  language's native hash table or dictionary type.
- Implementation details specific to SWI-Prolog (like `nb_setval`, `assert/retract`
  indexing) should be avoided in the transpilable core.

## Three-Layer Architecture

### Layer 1: Indexed Fact Tables

Facts are pure data. They should be stored in hash-indexed tables:

```
category_parent: Map<String, [String]>
  "a"         → ["b"]
  "b"         → ["physics"]
  "Physicists" → ["Physics", "Scientists_by_field"]
```

A WAM `call category_parent/2` becomes a map lookup:
- A1 is bound → `Map.lookup(A1)` → iterate over values, unify with A2
- A1 is unbound → iterate all entries (full scan, rare case)

This replaces 18,000 WAM instructions with a single O(1) hash lookup per query.

### Layer 2: Indexed Rule Heads

Rule heads select which clause body to execute based on argument patterns:

```prolog
category_ancestor(Cat, Parent, 1, Visited) :- body1...
category_ancestor(Cat, Ancestor, Hops, Visited) :- body2...
```

The head `get_constant(1, A3)` in clause 1 distinguishes it from clause 2 (where
A3 is a variable). This dispatch can be done natively:

```
if A3 == 1 → execute body1
else       → execute body2
```

The `try_me_else` / `trust_me` WAM instructions for clause selection become a
native conditional or pattern match. Only the **body** logic (after `:-`) needs
WAM instructions.

### Layer 3: WAM Body Logic

The WAM instruction set handles what can't be natively lowered:
- Backtracking within a clause body
- Cut (`!/0`) and its barrier semantics
- Negation-as-failure (`\+/1`)
- Arithmetic evaluation (`is/2`)
- Structure construction (`put_structure`, `put_list`)

This is typically 5-20 instructions per clause, not thousands.

## Design Principles

### 1. Hash Tables as the Universal Data Structure

Every target language has a native O(1) dictionary:

| Language | Hash Table | Persistent Map |
|----------|-----------|---------------|
| Haskell | `Data.HashMap` | `Data.Map` (O(log n), structural sharing) |
| Rust | `HashMap` | N/A (clone is O(n)) |
| Elixir | `Map` | Built-in (immutable, O(log n)) |
| Prolog | `assoc` (O(log n)) | `assert/retract` (O(1) with indexing) |

The WAM runtime abstraction layer should define:
- `dict_new()` → empty dictionary
- `dict_lookup(Key, Dict)` → value or not_found
- `dict_insert(Key, Value, Dict)` → new dictionary
- `dict_from_list(Pairs)` → dictionary

Each target transpiler maps these to the native type.

### 2. Separation of Concerns

| Concern | Mechanism | WAM Role |
|---------|-----------|----------|
| Fact lookup | Hash table | None (native) |
| Rule dispatch | Pattern match / conditional | Minimal (indexing only) |
| Body execution | WAM instructions | Full |
| Backtracking state | Choice points with dict snapshots | Full |
| Variable binding | Binding dict with trail | Full |

### 3. The WAM Compiler Outputs Three Things

Instead of a single instruction stream, `wam_target.pl` should output:

1. **Fact tables** — `{predicate: {first_arg: [clauses]}}`
2. **Rule index** — which clauses to try for which argument patterns
3. **Body code** — WAM instructions for each clause body only

The target transpiler combines these into native code.

### 4. Portable Binding Table

The binding table maps variable names to values. It must support:
- O(1) lookup (hot path — every `get_reg_val` dereferences)
- O(1) snapshot (for choice points)
- O(k) restore (unwind k bindings on backtrack)

This rules out mutable hash tables (O(n) snapshot) and favors either:
- **Persistent maps** (Haskell `Data.Map`, Elixir `Map`) — O(log n) lookup, O(1) snapshot
- **Hash table + trail** — O(1) lookup, O(k) restore via trail replay

The Prolog emulator should support both strategies via the abstraction layer.

## Lessons from the Rust and Haskell Spikes

### What killed Rust performance (343x slower)
- HashMap clone per choice point (O(n) per `TryMeElse`)
- Vec clone for stack and trail
- Interpreted instruction dispatch loop

### What made Haskell fast (<1s)
- `Data.Map` structural sharing (O(1) snapshot)
- Algebraic data types for instruction dispatch
- Lazy evaluation avoiding unnecessary computation

### What the indexed architecture fixes
- **Fact lookup**: 18K instructions → 1 hash lookup. Eliminates the largest
  source of WAM instructions entirely.
- **Rule dispatch**: `try_me_else` chains → native conditional. Eliminates
  clause-selection overhead.
- **Body execution**: Only the 5-20 instructions per clause body remain.
  These are fast in any implementation.

## Expected Performance

| Component | Current (WAM-only) | Indexed Architecture |
|-----------|-------------------|---------------------|
| Fact lookup | O(n) instruction scan | O(1) hash lookup |
| Rule dispatch | O(k) try/retry chain | O(1) pattern match |
| Body execution | ~50 WAM steps/clause | ~15 WAM steps/clause |
| Total per query | ~1000s of steps | ~20 steps |
| 300-scale benchmark | 116s (Rust), >30s (Prolog) | <1s target |
