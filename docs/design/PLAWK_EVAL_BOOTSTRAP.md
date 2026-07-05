<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk eval bootstrap (JIT roadmap item 5)

Compiling a grammar from **source text at runtime** — the endgame of the
dynamic-grammar arc. This is the plan and the milestone sequence; it is a
multi-PR effort, not a single change.

## The goal

```awk
BEGIN { DYNLOAD = "..." }
{ g = compile($1) ; total += dyncall_at(g, $2) }
```

`eval(Source)` / `compile(Source)` = compile a grammar from a source string
**inside the running binary**, then load and call it. Concretely:

1. Ship the WAM compiler itself as a `.wamo` object (`compiler.wamo`).
2. `eval(src)` loads `compiler.wamo` (once, lazily — the pay-for-what-you-use
   invariant: the compiler object only loads when an `eval` surface is used).
3. Run the compiler object on the source string → it returns the compiled
   grammar's **`.wamo` bytes** (a blob — the byte-return bridge from item 2).
4. Load those bytes with the **in-memory loader** `@wam_object_load_bytes`
   (#3463) → a `%WamState*` + entry PC.
5. Call the freshly compiled grammar like any `dyncall`.

Steps 3–5 already have their primitives: `@wam_object_call_bytes` returns a
byte slice, and `@wam_object_load_bytes` + `@wam_object_call_i64` load and run
a buffer. The `mtime` cache-invalidation path (#3465) is the natural home for
"recompile when the source changes." **The hard part is step 1**: getting the
compiler to run *as a loaded object*, which needs the loadable WAM subset
expanded to cover what the compiler leans on.

## Why it is genuinely last

The compiler is ordinary Prolog compiled to WAM, but it uses constructs the
`.wamo` loadable subset does not yet cover. Until the subset reaches the
compiler's needs, the compiler can be compiled *ahead of time* (as the host
does today) but not *loaded and run* from a `.wamo`. So item 5 is really a
**subset-expansion campaign** with the bootstrap as its payoff.

The loadable subset today: `try_me_else`/`retry_me_else` chains,
`get`/`put`/`set`/`unify` variable+value+constant (integer, atom, float),
`get`/`put_list`, `get`/`put_structure`, `allocate`/`deallocate`,
`call`/`execute` (**including `call/N` meta-calls** — milestone 2, landed),
`proceed`, `builtin_call`, `cut`/`get_level`/`cut_ite`, `jump`, **all
clause-indexing dispatch** — every `switch_on_*` variant, matched by prefix
so the `_fallthrough` and `_a2` forms are covered too — as nop-fallthroughs,
and **aggregate control** `begin_aggregate`/`end_aggregate` (milestone 3 —
`findall`/`setof`/`bagof`/`aggregate_all`).

### Why indexing dispatch is a safe nop

The tier-2 compiler emits every indexing instruction **inline at the head of
the predicate**, immediately before the `try_me_else` chain, e.g. for an
atom-keyed `col/2`:

```
switch_on_constant red:default green:L_col_2_2_body blue:L_col_2_3_body
try_me_else L_col_2_2
get_constant red, A1
...
L_col_2_2: retry_me_else L_col_2_3
...
L_col_2_3: trust_me
...
```

The switch is an *optimization*: its entries only point deeper into the same
`try_me_else`/`retry`/`trust` chain that follows it inline. Dropping the
switch and falling through runs every clause in order — correct, just
unindexed. The loader already did this for `switch_on_term`; extending it to
`switch_on_constant`/`_a2` (this PR) means grammars with atom-keyed clauses —
pervasive in the compiler — now load. Cost: unindexed dispatch in loaded
objects (an optimization gap, not a correctness one).

## Milestones (subset expansion → bootstrap)

Rough dependency order. Each is its own PR(s); the earlier ones are
independently useful (richer hand-written grammars load sooner).

1. **Clause indexing** — *LANDED (this PR).* `switch_on_constant`/`_a2` as
   nop-fallthroughs. Atom-keyed multi-clause predicates load.
2. **`call/N` meta-call in objects** — *LANDED.* A meta-call encodes
   `op1 = -1` (`op2` = total arity), exactly as the host AOT path does, and
   dispatches through `@wam_dispatch_meta_call`. The subtlety is that
   dispatch resolves a *goal* (atom/functor + arity) to a *label index*, and
   a loaded object's predicates are not in the host's compile-time dispatch
   table. So the format grew a **per-object meta-call table** (version 2): a
   trailing section listing each predicate as `(atomIdx, funIdx, arity,
   labelIdx)` into the object's own atom/functor tables. The loader
   materializes it into a `%WamMetaRow` array hung off two new `%WamState`
   fields (25/26); `@wam_dispatch_meta_call` consults the VM's table when
   present (via `@wam_meta_find_atom` / `@wam_meta_find_compound`) and falls
   back to the host-global arrays otherwise. Compound-goal matching works
   because a loaded object's runtime functor pointers are its own malloc'd
   copies — the very pointers the table stores — so identity holds within one
   object. The table is only emitted for objects that actually contain a
   meta-call (pay-for-what-you-use). This is the spine of any grammar that
   calls a goal built at runtime — an interpreter or compiler.
3. **The builtin closure the compiler needs** — *audit done; aggregates
   landed.* `builtin_call` is already in the subset, so a builtin works in a
   loaded object *if the host runtime provides it AND the construct is in the
   loadable subset*. The audit sorted the compiler's constructs into four
   buckets by **how the tier-2 compiler lowers them**, which decides what
   "make it loadable" means:

   | Construct | Lowers to | Loadable now? |
   |---|---|---|
   | `is`/`=`/`==`/comparison, `functor`/`arg`/`=..`, `copy_term`, `atom_codes`/`number_codes`/`char_code`, `sub_atom`/`atom_concat`, `sort`/`msort`/`keysort`, `length`/`append`/`nth`/`reverse`, `sum_list` … | `builtin_call <id>` (id in `builtin_op_to_id`, host-implemented) | **yes** — already in the subset and loader-safe (operate on VM regs/heap + libc) |
   | `findall`/`aggregate_all`, `setof`/`bagof` | `begin_aggregate`/`end_aggregate` (tags 28/29) | **yes, this PR** — opcodes lifted into the subset; setof/bagof additionally need `inline_bagof_setof(true)`, now the `.wamo` default |
   | `term_to_atom/2` (write direction) | `builtin_call term_to_atom/2` (id 173) → `@wam_term_to_sb` | **yes, milestone 3b** — a recursive term→text writer into a growable buffer, interned as an atom. Works in loaded objects too: cons detection is by functor *bytes*, not pointer identity. Unquoted (write semantics), so it does not yet round-trip through a reader |
   | `assertz`/`asserta`/`retractall` | `builtin_call <name>` (ids 175/176/177) → `@wam_dyn_assert` / `@wam_dyn_retractall` | **yes, milestone 3b-db (PR 1)** — a process-global, malloc-backed clause store that survives the arena rewind. Ground facts; calling them goes through the `call/1` meta-call, whose meta-table miss consults the store with unification + backtracking (`agg_type = -3` choice point). See PLAWK_DYNAMIC_DB.md |
   | `retract/1` (nondet), direct calls to `:- dynamic` predicates | `call retract/1` / `execute <dyn>/N` | **yes, milestone 3b-db (PR 2)** — direct dynamic calls are rewritten to a `call/1` store consult (`dynamic_store_goal/1`); nondet `retract/1` is an `agg_type = -4` remove+unify+backtrack iterator (op1 = -3 sentinel). See PLAWK_DYNAMIC_DB.md |
   | `read_term`/`read_term_from_atom` | `builtin_call read_term_from_atom/2` (id 174) → the reader | **yes, milestone 3b** — a tokenizer + operator-precedence recursive-descent parser (done: canonical + operator surface, variables, control ops, floats, quoted atoms) |
   | `catch`/`throw` | `call`/`execute catch/3` (op1 = -5) and `throw/1` (op1 = -6) | **yes, milestone 3c** — a process-global side stack of catch frames; catch pushes a frame + meta-calls Goal, throw deep-copies the ball and unwinds to the nearest frame whose catcher unifies, running Recovery. No cross-object linkage needed (the runtime handles both). See PLAWK_DYNAMIC_DB.md |

   **Landed here:** the aggregate opcodes, verified end-to-end — `findall`,
   `setof`, `bagof` over a *user predicate* goal load and run from a `.wamo`
   (the case a compiler actually hits, iterating over clauses). Also note a
   **pre-existing host bug** surfaced by the audit: a *backtracking list
   builtin as a findall goal* (e.g. `findall(X, member(X, L), _)`) segfaults
   even in AOT-compiled host code — orthogonal to the loader, filed
   separately, and not a blocker since compiler-style `findall` iterates over
   predicates, not `member`.

   **term_to_atom/2, write direction (milestone 3b):** a recursive term→text
   writer (`@wam_term_to_sb`) into a growable buffer, interned as an atom.
   Verified in both AOT and loaded objects; the byte-based cons detection
   (`@wam_functor_is_cons`) makes list rendering correct across the loader
   boundary despite each object carrying its own functor copies.

   **Reader (milestone 3b):** `read_term_from_atom/2` is a recursive-descent
   parser (`@wam_parse_term` / `@wam_parse_list` / `@wam_make_atomic`, ported
   in structure from the C++ hybrid target) for canonical terms: integers,
   unquoted atoms, compounds `name(a,b)`, and lists `[a,b|t]` (unbounded, built
   through a tail slot). Verified in loaded objects — `point(3,4)` unifies
   against a source literal `point(X,Y)`, nested `f(g(7),h(5))` decomposes,
   `[10,20,30]` sums to 60.

   The functor-pointer problem it exposed is **solved**, not deferred:
   unification compares compound functors by pointer, and a reader-built
   functor (from the atom table) will not equal the AOT `@.fn_*` global for the
   same name. `@wam_functor_eq` makes the compare **pointer-fast with a strcmp
   fallback only when the pointers differ** — the hot AOT-vs-AOT path is
   unchanged, dynamic compounds unify correctly, and this also retroactively
   fixes the `=..` "did not compare equal to a literal" issue. (List cons cells
   reuse the shared `@.fn__5B_7C_5D` global, so they stay pointer-equal.)

   **Operators (milestone 3b):** a precedence-climbing layer
   (`@wam_parse_expr`) sits over the primary parser, with a char-class
   tokenizer (`a+b` splits without spaces) and a table-driven operator lookup
   (`@wam_infix_op`) covering arithmetic (`+ - * / // mod rem`) and the
   700-level comparisons / `is`. Correct precedence, associativity (yfx/xfx/xfy
   parse the right operand at `prio-1` / `prio`), parentheses, and negative
   numbers. Verified in loaded objects: `1+2*3`→7, `(1+2)*3`→9,
   `100 - 2 * -3`→106 — evaluated by `is/2`, which works because the arithmetic
   evaluator dispatches on the functor bytes (so it reads reader-built
   `+`/`*`/`-` compounds directly).

   **Variables (milestone 3b):** a per-parse var-dictionary (a transient block
   hung off `%WamState` field 27, set by `read_term_from_atom` while parsing).
   A name led by an uppercase letter or `_` is a variable; `@wam_var_ref`
   interns the name and looks it up in the dict so repeated occurrences share
   one fresh heap cell (`X` in `p(X,X)`), while anonymous `_` is always a fresh,
   unshared cell. Reader variables are bound by unifying the parsed *term* (they
   have no connection to the surrounding clause's variables). Verified in loaded
   objects: `p(X,X)` binds both from one unify (→9), shared var through
   arithmetic (→42), anonymous `q(_,_) = q(3,4)` succeeds (distinct).

   **Control operators (milestone 3b):** `:-` (1200 xfx), `,` (1000 xfy), `;`
   (1100 xfy), `->` (1050 xfy) — added as `@wam_infix_op` table entries over the
   precedence-climbing machinery. `,` and `;` are single punctuation chars
   (neither symbol nor alnum), so `parse_expr` gains explicit solo-token
   handling for them; `:-`/`->` tokenize as symbol runs. With variables +
   these, **a whole clause parses**: `read_term_from_atom("foo(X) :- bar(X),
   baz(X)", T)` yields `:-(foo(X), ,(bar(X),baz(X)))` with `X` shared across
   head and body (verified in a loaded object: binding X once is visible in the
   body goal → 7; right-assoc conjunction `1,2,3` → 123; disjunction → 33).
   The reader now covers the term shapes a clause is made of.

   **Floats + quoted atoms (milestone 3b) — the reader is now complete for the
   canonical + operator surface.** A digit- or `-`digit-led token routes to a
   `strtod`-based number scan: the endptr gives the token end, and a scan for
   `.`/`e`/`E` picks Float (via `@value_float`) vs Integer (exact, via
   `@wam_make_atomic`). A `'`-led token reads a single-quoted atom (content to
   the next quote; no escape handling yet). Verified in loaded objects:
   `3.5 + 1.5` → 5, negative/compound floats, and `'hello world'` → length 11.

   **Dynamic clause store (milestone 3b-db) — PR 1 + PR 2 landed.** A
   process-global, malloc-backed clause store (survives the arena rewind) with
   `assertz` / `asserta` / `retractall` builtins. Calling a dynamic fact goes
   through the `call/1` meta-call (its meta-table miss consults the store, with
   unification and backtracking through an `agg_type = -3` choice point), and
   PR 2 makes **direct** calls (`counter(N)`, not just `call(counter(N))`) reach
   it by rewriting them to `call/1` at compile time, plus **nondet `retract/1`**
   as an `agg_type = -4` remove+unify+backtrack iterator. Works in loaded
   objects — the store is process-global. See **PLAWK_DYNAMIC_DB.md**.
   Remaining there: rule bodies (`assertz((H :- B))`, PR 3) and `call/N`
   partial-application consult.

   **`catch`/`throw` (milestone 3c) — landed.** A process-global side stack of
   catch frames (`@wam_catch_setup` / `@wam_throw`, reset per top-level query in
   `@wam_prepare_call`). No cross-object linkage was needed after all: `catch/3`
   and `throw/1` lower to op1 sentinels (-5 / -6) the runtime handles directly,
   and Goal/Recovery are run through the existing meta-call dispatch. Verified
   in loaded objects (catch + recover, goal-succeeds, nested/propagating throw,
   recovery using the ball value, uncaught). Documented scoping limitation:
   catch protects Goal, but a frame can linger within a query until backtracking
   or the next query resets it, so a throw sequenced *after* a catch in the same
   clause may be caught by it (real ISO restricts catch to Goal).

   **Remaining:** a minor loadable-subset gap — `arg/3` with a constant index
   compiles to a specialised `arg` opcode outside the `.wamo` subset (so loaded
   objects decompose reader terms via unification / `functor/3`, not `arg/3`) —
   a candidate subset lift; and PR 3 for the dynamic store (rule bodies). The
   **reader is done**, the **dynamic store** gives a grammar mutable state, and
   **catch/throw** gives it error handling — the runtime-primitive layer for the
   eval surface (milestone 5) is now essentially complete.

   (Aside: `findall(X, call(G), L)` — an aggregate over a `call/1` meta-call
   goal — is now fixed. It used to collect nothing: the tier-2 compiler
   re-initialised the aggregate template variable with a fresh cell, breaking
   the sharing with the copy embedded in the pre-built goal `G`, so the goal
   bound one cell while the aggregate collected another. The fix skips that
   re-initialisation when the template var already has a register. Works for
   both object-compiled predicates and dynamic-store facts.)
4. **Byte-buffer output from a grammar. — landed.** The compiler object must
   *emit* `.wamo` bytes. It returns them as an Atom/byte string (the item-2
   blob bridge already carries bytes out); building that byte string inside the
   grammar needs the string/codes builtins from milestone 3.

   **What landed:** this milestone needed *no new target IR* — it composes
   primitives already in place. A loaded grammar assembles a byte string with
   the milestone-3 string/codes builtins (`number_codes`, `atom_codes`,
   `atom_concat`, arithmetic) and returns it as an Atom; the host reads it back
   through `@wam_object_call_bytes`, which returns `{ptr, len, ok}` — the
   pointer is into the persistent atom table, so it survives the arena rewind,
   and the length lets the caller print it with `%.*s` (embedded NULs and the
   absence of a trailing newline are both fine). Verified in loaded objects
   (`tests/test_wam_object.pl:emit_bytes_from_object`): a computed decimal
   (`6*7` → `"42"`), a synthesized header line (`atom_concat('WAMO ', V, S)` →
   `"WAMO 2"`, the shape of a real `.wamo` header), and a literal code list
   (`atom_codes(S,[104,105])` → `"hi"`). The byte-return path is the same one
   the eventual `eval`/`compile` surface (milestone 5) hands assembled `.wamo`
   text back across.
5. **The `eval` / `compile` surface + pipeline.** Wire the plawk surface:
   `compile(src)` → run `compiler.wamo` on `src` (blob out) →
   `@wam_object_load_bytes` → a handle usable by `dyncall_at`. Lazy-load the
   compiler object; cache per `DYNCACHE`.
6. **Self-host.** Compile the actual WAM compiler to a `.wamo` and run the
   whole pipeline from source text end to end. The capstone.

## Cross-cutting notes

- **Pay-for-what-you-use holds:** the compiler object (large) loads only when
  an `eval`/`compile` surface is compiled into the program, riding
  `emit_wamo_loader(true)` like every other dynamic capability.
- **Correctness-first, speed-later:** loaded objects run unindexed (milestone
  1). If the eval loop becomes hot, a real switch-table in the loader is a
  later optimization — the format already carries the switch operands we
  currently drop.
- **Milestones 1–4 have landed**; 5–6 are plumbing over primitives that
  already exist (`@wam_object_load_bytes`, `@wam_object_call_bytes`, the
  `mtime` cache). The reader (3b) and byte-buffer output (4) together mean a
  loaded object can now both parse source text into terms *and* emit assembled
  bytes back out — the two halves the eval loop threads together.
