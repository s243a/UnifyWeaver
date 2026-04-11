# WAM Term Builtins: Specification

This document describes the technical shape of adding the four Group A term
inspection builtins — `functor/3`, `arg/3`, `=../2`, `copy_term/2` — to the
WAM target layer and the three hybrid backends (WAM-WAT, WAM-Rust, WAM-Haskell).

For the *why* of each decision, see `WAM_TERM_BUILTINS_PHILOSOPHY.md`. For
the phased rollout order, see `WAM_TERM_BUILTINS_IMPLEMENTATION_PLAN.md`.

`assert`/`retract` (Group B) are explicitly out of scope; they are tracked
separately because they require runtime-extensible clause storage, which is
an architectural change the three pure term-inspection builtins do not need.

## 1. Semantics

The semantics below mirror ISO Prolog plus SWI-Prolog conventions where the
standard leaves room. Error reporting is deliberately minimal in the v1
implementation (see §1.5) — we aim for correct behavior on well-formed
inputs first and defer full ISO error terms.

### 1.1 `functor/3`

Two modes:

**Read mode** `functor(+T, ?N, ?A)`
- `T` is instantiated (not an unbound variable)
- Binds `N` to the functor name, `A` to the arity
- For compound `T = f(a1,...,an)`: `N = f`, `A = n`
- For atom `T = foo`: `N = foo`, `A = 0`
- For integer `T = 42`: `N = 42`, `A = 0`
- For float `T = 3.14`: `N = 3.14`, `A = 0`

**Construct mode** `functor(-T, +N, +A)`
- `T` is unbound; `N` is an atom (or number, for arity-0); `A` is a
  non-negative integer
- If `A = 0`: `T` is bound to `N` itself
- If `A > 0`: `T` is bound to a fresh compound `N(_1,_2,...,_A)` with
  `A` fresh unbound argument cells

### 1.2 `arg/3`

`arg(+N, +T, ?A)`
- `N` is a positive integer
- `T` is a compound term with arity ≥ N
- Unifies `A` with the Nth argument of `T` (1-indexed)

### 1.3 `=../2` (univ)

**Decompose mode** `=..(+T, ?L)`
- `T` is instantiated
- For compound `T = f(a1,...,an)`: `L = [f, a1, ..., an]`
- For atomic `T = a`: `L = [a]`

**Compose mode** `=..(-T, +L)`
- `L` is a proper list `[F | Args]`
- If `Args = []`: `T` is bound to `F` (where `F` must be atomic)
- Otherwise: `F` must be an atom, and `T` is bound to the fresh compound
  `F(a1,...,an)` where the `ai` unify with the elements of `Args`

### 1.4 `copy_term/2`

`copy_term(+T, -Copy)`
- `T` may be any term (ground or with free variables)
- `Copy` is a structurally identical term in which every unbound variable
  of `T` has been replaced by a fresh unbound variable
- **Sharing is preserved within the copy**: if two positions in `T` share a
  variable, the corresponding positions in `Copy` share the *same* fresh
  variable. `copy_term(f(X,X), C)` gives `C = f(Y,Y)`, not `C = f(Y,Z)`.
- No attributed variables. The current WAM targets do not support them, and
  this spec does not introduce them.

### 1.5 Error handling (v1)

The v1 implementation is permissive:

- Type errors (non-atom in `functor/3` construct mode, non-integer in
  `arg/3`, etc.) produce `fail`, not ISO error terms
- Out-of-range errors (`arg(5, f(a,b), X)`) produce `fail`
- Mode errors (`functor(-T, -N, -A)`) produce `fail`

This matches how the existing `is/2` and arithmetic comparisons currently
handle errors in the WAM targets (silent fail on bad inputs). A follow-up
can add ISO-compliant error terms uniformly across all builtins.

## 2. WAM-level encoding

The four builtins are encoded as `builtin_call` instructions with new
builtin IDs, following the existing pattern in `wam_wat_target.pl:89-106`.
Arguments are passed in fixed registers `A1`, `A2`, `A3`. Output bindings
are written back to the same argument registers (the caller is responsible
for having placed input values there and for reading output values back).

### 2.1 Builtin ID extension

Extend the builtin ID table in each target:

```prolog
%% Existing (wam_wat_target.pl:89-106)
builtin_id('write/1',  0).
builtin_id('nl/0',     1).
builtin_id('is/2',     2).
builtin_id('=:=/2',    3).
builtin_id('=\\=/2',   4).
builtin_id('</2',      5).
builtin_id('>/2',      6).
builtin_id('=</2',     7).
builtin_id('>=/2',     8).
builtin_id('var/1',    9).
builtin_id('nonvar/1', 10).
builtin_id('atom/1',   11).
builtin_id('integer/1', 12).
builtin_id('float/1',  13).
builtin_id('number/1', 14).
builtin_id('true/0',   15).
builtin_id('fail/0',   16).
builtin_id('!/0',      17).

%% New
builtin_id('functor/3',    18).
builtin_id('arg/3',        19).
builtin_id('=../2',        20).
builtin_id('copy_term/2',  21).
```

The same four IDs (18–21) are used in all three backends for consistency.
Each backend's `$execute_builtin` (or equivalent) gets four new cases.

### 2.2 Register conventions

| Builtin       | A1 (in/out)   | A2 (in/out)       | A3 (in/out) |
|---------------|---------------|-------------------|-------------|
| `functor/3`   | `T` in/out    | `N` in/out        | `A` in/out  |
| `arg/3`       | `N` in        | `T` in            | `A` in/out  |
| `=../2`       | `T` in/out    | `L` in/out        | —           |
| `copy_term/2` | `T` in        | `Copy` out        | —           |

The "in/out" distinction is runtime-determined: the builtin implementation
inspects each register's tag to decide mode. An `Unbound` input means
"output position"; anything else means "input position".

### 2.3 Canonical WAM layer change

Add to `wam_target.pl` around line 619:

```prolog
is_builtin_pred(functor, 3).
is_builtin_pred(arg, 3).
is_builtin_pred((=..), 2).
is_builtin_pred(copy_term, 2).
```

After this change, any user predicate that calls `functor/3` etc. will be
compiled to a `builtin_call functor/3, 3` instruction instead of a regular
`call` to a non-existent predicate. Predicates that were previously
unlowerable now emit valid WAM; they still fail at runtime until a backend
implements the new builtin IDs.

## 3. Per-target implementation requirements

### 3.1 WAM-WAT (`src/unifyweaver/targets/wam_wat_target.pl`, `templates/targets/wat_wam/`)

Four new WAT helper functions are added to `compile_wam_helpers_to_wat/2`:

- `$builtin_functor` (no params, returns i32 success/fail)
- `$builtin_arg`
- `$builtin_univ`
- `$builtin_copy_term`

Each helper:

1. Reads `A1`, `A2`, `A3` from the register file via `$get_reg_tag`/`$get_reg_payload`
2. Dereferences via `$deref`
3. Branches on the input mode (unbound output vs instantiated input)
4. Performs the operation against the heap
5. Writes results via `$set_reg` and uses `$trail_binding` for any
   bindings created on existing variables
6. Returns `1` on success, `0` on failure (triggers backtrack via `$run_loop`)

Dispatch in `$execute_builtin`:

```wat
;; ... existing cases 0–17 ...
(i32.eq (local.get $id) (i32.const 18))
  (if (result i32) (then (return (call $builtin_functor))))
(i32.eq (local.get $id) (i32.const 19))
  (if (result i32) (then (return (call $builtin_arg))))
(i32.eq (local.get $id) (i32.const 20))
  (if (result i32) (then (return (call $builtin_univ))))
(i32.eq (local.get $id) (i32.const 21))
  (if (result i32) (then (return (call $builtin_copy_term))))
```

#### 3.1.1 `copy_term` on flat linear memory

This is the only helper with significant complexity. Algorithm:

1. Allocate a small "var map" table on the heap: pairs `(old_var_id, new_var_id)`.
   Size bound: the number of distinct variables in the source term. For v1,
   a fixed 64-entry table is sufficient for the test corpus; overflow
   triggers fail.
2. Push the source term's heap offset onto a work stack.
3. While the work stack is non-empty, pop an offset, read its cell:
   - `Atom`/`Integer`/`Float`: write a matching cell into the destination
     and continue
   - `Unbound(id)`: look up `id` in the var map; if present, emit a `Ref`
     to the mapped new variable; if absent, allocate a fresh unbound
     variable, add the mapping, emit `Ref` to it
   - `Compound(functor, arity, arg_offset)`: allocate a fresh compound
     header at the destination, push each of the `arity` arg offsets
     onto the work stack (in reverse order, for left-to-right copy)
   - `List(head_off, tail_off)`: allocate a fresh list cell, push both
     offsets
   - `Ref(target)`: follow once and retry
4. Bind `A2` to the root of the destination.

The work stack reuses the WAM state area — there is already an env/choice
stack, and `copy_term` can claim a slice of it as scratch.

### 3.2 WAM-Rust (`src/unifyweaver/targets/wam_rust_target.pl`)

The Rust backend is the easiest of the three. It already has a `Value::univ()`
method (`templates/targets/rust_wam/value.rs.mustache`), a `deref_var`
helper, and persistent-enough data structures via `Rc<...>` that `copy_term`
can reuse existing variable-fresh allocation paths.

New builtin handlers go in the instruction interpreter loop where
`BuiltinCall` is handled (around `wam_rust_target.pl:195-209`):

```rust
// Builtin 18: functor/3
18 => {
    let t = vm.get_reg_deref("A1");
    let n = vm.get_reg_deref("A2");
    let a = vm.get_reg_deref("A3");
    match (t, n, a) {
        (Value::Unbound(_), n_val, Value::Integer(arity)) => {
            // Construct mode
            let fresh = vm.build_skeleton(n_val, arity as usize);
            vm.bind_reg("A1", fresh);
            true
        }
        (t_val, _, _) => {
            // Read mode
            let (name, arity) = t_val.functor_and_arity();
            vm.bind_reg("A2", name);
            vm.bind_reg("A3", Value::Integer(arity as i64));
            true
        }
    }
}
// ... similar for 19, 20, 21
```

`copy_term` uses a `HashMap<VarId, VarId>` as the local var map, recursively
walks the source, and builds a fresh term tree. The existing `Value` enum
and heap are sufficient.

### 3.3 WAM-Haskell (`src/unifyweaver/targets/wam_haskell_target.pl`)

Haskell has no template directory — all code is generated as inline format
strings. The four new builtins are added to the builtin dispatch case
expression (around `wam_haskell_target.pl:195-245`).

Haskell's persistent data structures make `copy_term` the simplest of the
three implementations:

```haskell
copyTerm :: Value -> WamM Value
copyTerm = go IntMap.empty
  where
    go m (Unbound i) = case IntMap.lookup i m of
        Just j  -> return (Unbound j)
        Nothing -> do
            j <- freshVar
            return (Unbound j)  -- sharing handled via returned map
    go m (Str f args) = Str f <$> mapM (go m) args
    go _ v = return v  -- Atom, Integer, Float: structural
```

The real implementation threads the `IntMap` properly (the sketch above
drops it between recursive calls; the actual code needs a `StateT` or
explicit map passing).

`functor/3`, `arg/3`, and `=../2` all fall out of Haskell pattern matching
on the existing `Value` type — implementations are a few lines each.

## 4. Heap and state impact

### 4.1 Heap growth

- `functor/3` construct mode: allocates one compound header + A fresh
  unbound cells
- `arg/3`: no heap allocation (just reads)
- `=../2` decompose mode: allocates one list cell per argument (O(arity))
- `=../2` compose mode: allocates one compound header + N arg cells
- `copy_term/2`: O(source-term-heap-cells) for the copy + O(distinct-vars)
  for the scratch var map

All allocations use the existing heap allocator (`$heap_alloc` in WAT,
`vm.heap.alloc()` in Rust, pure value construction in Haskell).

### 4.2 Trail impact

Bindings created on existing registers are trailed normally. The only
interesting case is `copy_term`: the fresh variables in `Copy` are *not*
trailed — they didn't exist before the call, so there's nothing to restore.
The scratch var map is also not trailed (it's scoped to the single call).

### 4.3 Choice point impact

None of the four builtins introduce choice points. They are deterministic.
If a future extension adds nondeterministic `arg/3` (iterate over arguments),
that would be in scope for a separate plan.

## 5. Test plan shape

Three layers of tests per builtin, per target:

### 5.1 Canonical layer (`tests/test_wam_target.pl` extension)

Verify that a predicate using the builtin produces a WAM instruction stream
containing `builtin_call functor/3, 3` (or the equivalent). This is the
cheapest test and catches regressions in `is_builtin_pred/2`.

### 5.2 Per-target codegen test

For each of WAM-WAT, WAM-Rust, WAM-Haskell: compile a predicate using each
builtin and assert that the generated code contains the expected helper
call. For WAT this is `sub_string` assertions on the emitted `.wat` text
(matching the existing `test_wam_wat_target.pl` style). For Rust it's
`sub_string` assertions on the generated `.rs`. For Haskell, same.

### 5.3 Functional execution test

For WAM-Rust (existing `test_wam_rust_runtime.pl` pattern): compile, run
via `cargo test`, assert on bindings.

For WAM-WAT (new harness needed, see the functional-test discussion from
the prior conversation): compile, run via `wat2wasm` + `wasmtime` with a
`record_result` host import, assert on captured bindings.

For WAM-Haskell: matches the existing Haskell test pattern — codegen
assertions only, no execution. (This is a known gap in Haskell testing and
is not this plan's job to fix.)

### 5.4 Specific test predicates

At minimum:

```prolog
% functor/3 read
test_functor_read :- functor(foo(1,2,3), F, A), F == foo, A == 3.

% functor/3 construct
test_functor_construct :- functor(T, bar, 2), T = bar(_, _).

% arg/3
test_arg :- arg(2, foo(a,b,c), X), X == b.

% =../2 decompose
test_univ_decompose :- foo(a,b) =.. L, L == [foo, a, b].

% =../2 compose
test_univ_compose :- T =.. [baz, 1, 2], T == baz(1, 2).

% copy_term: ground input
test_copy_ground :- copy_term(f(1,2), C), C == f(1,2).

% copy_term: variable sharing preserved
test_copy_sharing :-
    copy_term(f(X,X), C),
    C = f(A,B),
    A == B.  % <-- the critical assertion

% copy_term: variables are fresh
test_copy_fresh :-
    X = hello,
    copy_term(f(X,Y), C),
    C = f(hello, Z),
    var(Z).  % Z must be unbound — not aliased to Y
```

The `copy_term` sharing test (`test_copy_sharing`) is the single most
important test in this entire workstream. It is the property most likely
to be silently broken by an implementation bug, and the symptom is
correctness corruption rather than a crash.

## 6. Deferred items

### 6.1 Group B — assert/retract

Not in this spec. Requires:

- Runtime-extensible clause store per target
- Dispatch plumbing that lets `call` find dynamically-added clauses
- A backtracking-semantics decision (logical vs immediate update view)
- An indexing strategy for dynamic predicates

These are tracked for a separate `WAM_DYNAMIC_DATABASE_*` doc set.

### 6.2 ISO error terms

The v1 implementation uses silent-fail on malformed inputs. A follow-up can
introduce a uniform `throw/1`-compatible error path across all builtins.
This is cross-cutting work and should not be bundled with term builtins.

### 6.3 Nondeterministic `arg/3`

Standard Prolog's `arg/3` is deterministic. A nondeterministic variant that
iterates over all arguments is useful but non-ISO and introduces a choice
point. Deferred.
