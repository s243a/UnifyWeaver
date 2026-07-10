<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# plawk eval bootstrap (JIT roadmap item 5)

Compiling a grammar from **source text at runtime** — the endgame of the
dynamic-grammar arc. This was the plan and the milestone sequence; **the
payoff has landed** (see "The landed surface" below).

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

## The landed surface

```awk
{ total += dyncall_at(compile("[(sq(X2, R2) :- atom_number(X2, N2), R2 is N2 * N2)]"), $1) }
END { print total }
```

`compile(field-or-string)` in the **dyncall_at source position** compiles
the Prolog source text at runtime and yields a grammar HANDLE. It is the
goal snippet minus the intermediate variable: `compile` **deduplicates by
source text** (interned atom id), so a per-record `compile(...)` compiles
each distinct grammar once and reuses the loaded VM thereafter — the
two-step `g = compile(...)` form adds only variable plumbing (plawk
scalars are numeric; a handle-in-scalar surface is a possible follow-up).

How it works:

- The CLI ships the **bootstrap compiler** — the self-hosted `cgfull`
  (see PLAWK_SELFHOST.md; it lives in
  `src/unifyweaver/targets/wam_bootstrap_compiler.pl`) — as
  `<bin>.evalc.wamo` next to the output binary, ONLY when the program has
  compile sites (pay-for-what-you-use). `BEGIN { EVALC = "path.wamo" }`
  points at an existing compiler object instead and skips the emission.
- `@plawk_compile(src, len)` interns the source, dedups against the
  dyncall_at cache registry, and on miss runs
  `@wam_object_load_cached(evalc)` → `@wam_object_eval(cvm, cpc, src,
  len)` (compiler entry `cgfull(Src, Wamo)`; the emitted `.wamo` bytes
  load into a fresh VM) and records the grammar under the source id.
  The handle is the 1-based registry index; 0 means compile failure.
- The handle reaches the existing `@plawk_dyncall_at_N` shims as
  `(null path, handle)` — a null path is the discriminator
  `@plawk_dyncall_at_get` uses to resolve from the registry instead of
  the filesystem. All three shim families (i64 / float / blob) accept
  handles for free.
- `DYNCACHE "off"` cannot carry a handle (no registry), so compile sites
  under it are a **build error**, not a silent miscompile.
- Text-mode fields marshal as ATOMS (awk strings), so a numeric grammar
  converts with `atom_number/2` before arithmetic — the loaded runtime,
  like SWI, fails (rather than coerces) arithmetic on atoms.

End-to-end tests: `tests/test_plawk_eval_compile.pl` — a grammar compiled
from source text inside the binary sums `sq(x)` over input records; two
distinct runtime-compiled grammars coexist with per-source dedup; the
cache-off build error.

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
unindexed. The loader did this for the whole family through the eval
campaign — the "optimization gap, not a correctness one" that let
atom-keyed grammars (pervasive in the compiler) load early.

**The gap is now closed for `switch_on_constant` / `_a2`** (loaded clause
indexing): the writer encodes them as REAL dispatch rows (tags 25/27 —
the same step cases the AOT dispatcher uses) with their key→label tables
in a trailing `.wamo` section (emitted only when non-empty, so
switch-free objects — including everything the bootstrap compiler's own
serializer emits — stay byte-identical to the pre-section format, and
old objects load with the new loader unchanged). At load time each table
materializes as the `%SwitchEntry` array `@wam_switch_on_constant`
already consumes, with atom keys resolved through the object's relocated
atom ids and label indices resolved against the loaded VM's own label
table via `@wam_label_pc` — no new dispatch code at all. A strict switch
miss backtracks (exactly the AOT semantics), an unbound argument skips
into the chain. On a 200-clause atom-keyed fact table probed 20 000
times at its last key, the indexed object runs **7.6× faster** than the
same object with its tables stripped (25 ms vs 189 ms end to end),
byte-identical results. `switch_on_term`/`_structure` and the
`*_fallthrough` variants stay nops — matching the AOT dispatcher, which
nops them too.

The lift also fixed a latent AOT bug the loaded path inherited: the
switch-entry parser split `key:label` at the FIRST colon and never
unquoted writeq-style keys, so entries for atoms like `'=:='` or `'\=='`
sheared in half or interned their quote characters — either way the
table entry never matched at runtime and a strict switch silently
FAILED the call. Both parsers now split at the last colon and unquote
(`switch_entry_split/3`, `switch_entry_unquote/2`); regression
`loaded_switch_on_constant_dispatch` pins quoted-key dispatch, the
clean miss-into-else, and the unbound skip on the loaded path, and
`test_quoted_key_dispatch_executes` (tests/core/test_wam_llvm_switch.pl)
pins it on the AOT path with an executed native binary — verified to
fail against the pre-fix parser and pass with it.

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
   as an `agg_type = -4` remove+unify+backtrack iterator. PR 3 adds **rule
   bodies** (`assertz((H :- B))`): a var-preserving clause copy (head↔body
   variable sharing, fresh vars per call) plus a deterministic body interpreter
   handling `,`/2, builtins, and predicate calls (including nested rules).
   Works in loaded objects — the store is process-global. Body `;`/`->`/`\+`
   (deterministic if-then-else / disjunction / negation) are handled too. See
   **PLAWK_DYNAMIC_DB.md**. Remaining there: `!` (cut) and cross-goal
   backtracking in bodies, retract of rules, and `call/N` partial application.

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
   a candidate subset lift. The **reader is done**, the **dynamic store** (facts
   AND rule bodies) gives a grammar mutable state, **catch/throw** gives it
   error handling, and the **eval pipeline** loads+runs emitted bytes — the
   runtime-primitive layer for the eval surface is complete; only milestone 6
   (a real compiler grammar, self-host) remains.

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
5. **The `eval` / `compile` surface + pipeline. — landed (runtime).**
   `compile(src)` → run `compiler.wamo` on `src` (blob out) →
   `@wam_object_load_bytes` → a handle usable by `dyncall_at`. Lazy-load the
   compiler object; cache per `DYNCACHE`.

   **What landed:** two runtime primitives that compose the existing pieces
   into the eval loop, in `wam_object_support_ir` (emitted with
   `emit_wamo_loader(true)`):
   - `@wam_object_eval(cvm, cpc, src, srclen)` — interns `src` as an atom, runs
     the compiler entry `compile(Src, Wamo)` via `@wam_object_call_bytes`
     (`Src` in A1, emitted bytes read from A2), then feeds those bytes straight
     to `@wam_object_load_bytes`, returning the fresh `{vm, entry_pc}`. The
     emitted bytes live in the persistent atom table, so they survive the
     compiler VM's arena rewind and need no buffer management.
   - `@wam_object_load_cached(path)` — lazy-loads a compiler object once and
     memoizes it in a small path-keyed cache, so repeated `compile` calls reuse
     one compiler VM (the `DYNCACHE` role).

   Verified end to end (`tests/test_wam_object.pl:eval_compile_pipeline`): a
   loaded compiler object runs on source text, emits `.wamo` bytes,
   `@wam_object_eval` loads them into a fresh VM in the same process, and
   running its entry yields `42`. The stand-in compiler echoes its source (so
   the "source" is itself a valid `.wamo`); a real source-to-bytecode compiler
   is milestone 6. This closes the eval loop at the runtime layer: **emit bytes
   → load → run**, all in one process.
6. **Self-host. — design landed.** Not the full host compiler: a **minimal**
   Prolog→`.wamo` compiler written in the loadable subset, run through the
   existing `@wam_object_eval` pipeline. The enabler is that `.wamo` is a
   **text** format (`wamo_serialize/8`), so its back end is string assembly —
   loadable since milestone 4 — while its front end (`read_term_from_atom/2`)
   and middle (`functor/3`/`arg/3`/`=..` + list builtins) are already loadable
   too. The one real subset gap is constant-index `arg/3` (handled by a coding
   constraint or a small opcode lift). Staged A→D from a diffable serializer to
   the self-host fixpoint. Full design: **PLAWK_SELFHOST.md**.

## Cross-cutting notes

- **Pay-for-what-you-use holds:** the compiler object (large) loads only when
  an `eval`/`compile` surface is compiled into the program, riding
  `emit_wamo_loader(true)` like every other dynamic capability.
- **Correctness-first, speed-later — the "later" arrived:** loaded objects
  ran unindexed through the whole campaign (milestone 1's nop lift); with
  the fixpoint closed, `switch_on_constant`/`_a2` are now REAL dispatch in
  loaded objects (see "Why indexing dispatch is a safe nop" above for the
  landed design: a trailing table section, loader-built `%SwitchEntry`
  arrays, shared step cases, 7.6× on a wide-fact-table probe).
- **Milestones 1–5 have landed, and the dynamic store is complete through rule
  bodies (3b-db PR 3);** milestone 6 is self-host. The reader (3b), byte-buffer
  output (4), dynamic store incl. rule bodies + catch/throw (3b-db/3c), and the
  eval pipeline (5) mean a loaded object can parse source text into terms, keep
  mutable state (facts and rules), handle errors, emit assembled bytes, and
  load+run those bytes in the same process — the whole eval loop, wanting only
  a real compiler grammar to fill it (6).
