# WAM Indexed Architecture: Specification

## Abstraction Layer: Portable Dictionary Interface

### Dictionary Operations

All target languages implement these operations. The Prolog emulator defines
them as predicates; the transpiler maps them to native types.

```prolog
%% dict_new(-Dict)
%  Create an empty dictionary.
dict_new(Dict).

%% dict_lookup(+Key, +Dict, -Value)
%  Look up a key. Fails if not found.
dict_lookup(Key, Dict, Value).

%% dict_insert(+Key, +Value, +DictIn, -DictOut)
%  Insert or update a key. Returns a new dictionary.
dict_insert(Key, Value, DictIn, DictOut).

%% dict_from_list(+Pairs, -Dict)
%  Build a dictionary from Key-Value pairs.
dict_from_list(Pairs, Dict).

%% dict_to_list(+Dict, -Pairs)
%  Convert to a list of Key-Value pairs.
dict_to_list(Dict, Pairs).

%% dict_keys(+Dict, -Keys)
%  Get all keys.
dict_keys(Dict, Keys).
```

### Target Mappings

| Operation | Prolog | Haskell | Rust | Elixir |
|-----------|--------|---------|------|--------|
| `dict_new` | `empty_assoc(D)` | `Map.empty` | `HashMap::new()` | `%{}` |
| `dict_lookup` | `get_assoc(K,D,V)` | `Map.lookup k d` | `d.get(&k)` | `Map.get(d, k)` |
| `dict_insert` | `put_assoc(K,V,D,D1)` | `Map.insert k v d` | `d.insert(k,v)` | `Map.put(d, k, v)` |
| `dict_from_list` | `list_to_assoc(L,D)` | `Map.fromList l` | `HashMap::from_iter(l)` | `Map.new(l)` |

## Fact Table Format

### Schema

A fact table for predicate `P/N` is a dictionary indexed by first argument:

```
FactTable(P/N) = Dict<Value, [Tuple]>
```

Where `Tuple` contains the remaining arguments (A2..AN).

### Example: `category_parent/2`

```
category_parent_table = {
  "a":             [("b",)],
  "b":             [("physics",)],
  "Physicists":    [("Physics",), ("Scientists_by_field",)],
  "Astrophysics":  [("Astronomy",), ("Subfields_of_physics",)],
  ...
}
```

### Lookup Semantics

```
call_fact(P/N, A1, A2, ..., AN, State) →
  if A1 is bound:
    tuples = dict_lookup(A1, fact_table(P/N))
    for each tuple in tuples:
      if unify(A2..AN, tuple):
        yield success  (with choice point for remaining tuples)
  else:
    for each (key, tuples) in fact_table(P/N):
      unify(A1, key)
      for each tuple in tuples:
        if unify(A2..AN, tuple):
          yield success
```

## Rule Index Format

### Schema

Rules for predicate `P/N` are indexed by constant arguments in the head:

```
RuleIndex(P/N) = [RuleEntry]

RuleEntry = {
  head_pattern: [ArgPattern],  -- Constant(val) | Variable
  body_code: [WAMInstruction]  -- only body, no head matching
}
```

### Example: `category_ancestor/4`

```
rules = [
  { head_pattern: [Variable, Variable, Constant(1), Variable],
    body_code: [call category_parent/2, ..., builtin \+/1, proceed]
  },
  { head_pattern: [Variable, Variable, Variable, Variable],
    body_code: [call max_depth/1, builtin length/2, ..., call category_ancestor/4, ..., builtin is/2, proceed]
  }
]
```

### Dispatch

```
call_rule(P/N, A1..AN, State) →
  for each rule in rules:
    if head_pattern matches A1..AN:
      bind variables from head
      execute body_code
      on backtrack: try next matching rule
```

Constant arguments in the head become equality checks (O(1)).
Variable arguments always match.

## WAM Body Instructions

Only these instructions remain for clause bodies:

### Register Operations
- `put_value(Xn, Ai)` — load register
- `put_variable(Xn, Ai)` — create fresh variable
- `put_constant(C, Ai)` — set constant
- `put_structure(F/N, Ai)` — begin structure construction
- `put_list(Ai)` — begin list construction
- `set_value(Xn)` — add arg to structure/list
- `set_constant(C)` — add constant arg

### Control
- `call(P/N)` — call predicate (dispatches to fact table or rule index)
- `proceed` — return to caller

### Builtins
- `builtin_call(!/0)` — cut
- `builtin_call(is/2)` — arithmetic
- `builtin_call(length/2)` — list length
- `builtin_call(</2)` — comparison
- `builtin_call(\+/1)` — negation-as-failure

### Removed from Body Code
These are handled by the fact table and rule index layers:
- ~~`get_constant(C, Ai)`~~ — head matching, done by rule index
- ~~`get_variable(Xn, Ai)`~~ — head binding, done by rule index
- ~~`get_value(Xn, Ai)`~~ — head unification, done by rule index
- ~~`try_me_else(L)`~~ — clause selection, done by rule index
- ~~`retry_me_else(L)`~~ — clause retry, done by rule index
- ~~`trust_me`~~ — last clause, done by rule index
- ~~`switch_on_constant`~~ — first-arg dispatch, done by fact table
- ~~`allocate`~~ — env frame for head, rule index handles this
- ~~`deallocate`~~ — env frame cleanup, rule index handles this

## WamState (Revised)

```
WamState = {
  pc:         Int,                    -- program counter within body code
  regs:       Dict<String, Value>,    -- A/X registers
  stack:      [EnvFrame],             -- for nested calls only
  bindings:   Dict<String, Value>,    -- variable binding table
  trail:      [TrailEntry],           -- for backtracking undo
  cp:         Int,                    -- continuation pointer
  choice_points: [ChoicePoint],       -- backtracking state
  fact_tables: Dict<String, FactTable>,  -- NEW: indexed fact data
  rule_index:  Dict<String, [Rule]>,     -- NEW: indexed rule dispatch
}
```

The `code` and `labels` fields from the old WAM state are replaced by
`fact_tables` and `rule_index`. Body code is stored within each rule entry.

## Binding Table

### Requirements
- O(1) lookup (deref on every register read)
- Snapshot for choice points (save on TryMeElse equivalent)
- Restore on backtrack

### Implementation Strategy

Two modes, selectable per target:

**Mode A: Persistent Map (Haskell, Elixir)**
- Lookup: O(log n)
- Snapshot: O(1) — shared reference
- Restore: O(1) — swap reference
- Best for backtracking-heavy workloads

**Mode B: Hash Map + Trail (Rust, C)**
- Lookup: O(1) amortized
- Snapshot: O(1) — save trail length
- Restore: O(k) — replay k trail entries
- Best for forward-execution-heavy workloads

The Prolog emulator uses Mode A (`assoc` / persistent tree). Target transpilers
choose based on the target language's strengths.
