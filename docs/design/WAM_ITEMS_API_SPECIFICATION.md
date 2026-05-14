# WAM Items API — Specification

The concrete shape of the structured-items API we're adding to
`wam_target.pl`. For *why* each decision, see
`WAM_ITEMS_API_PHILOSOPHY.md`. For per-target migration plans, see
the per-target section in §6 below.

## 1. Public API

Two new exports from `wam_target.pl`. The existing
`compile_predicate_to_wam/3` keeps its semantics — it's now
implemented as `wam_items_to_text(Items, Text)` driven by the new
items predicate.

```prolog
%% compile_predicate_to_wam_items(+PredIndicator, +Options, -Items)
%
%  Returns a list of structured WAM items for the predicate. Each
%  item is either label(NameStr) or an instruction term whose
%  shape matches the existing parser-output convention (so
%  consumers that have been parsing the text format can switch
%  with minimal code changes).
%
%  Options accepts the same flags as compile_predicate_to_wam/3
%  (module/1, inline_bagof_setof/1, etc.).
compile_predicate_to_wam_items(PredIndicator, Options, Items).

%% wam_items_to_text(+Items, -Text)
%
%  Pretty-prints an items list as the canonical multi-line WAM
%  text format that compile_predicate_to_wam/3 has produced
%  historically. The output is byte-identical to the existing text
%  emission for any items list that compile_predicate_to_wam_items
%  could produce.
wam_items_to_text(Items, Text).
```

The existing predicate becomes a one-liner:

```prolog
compile_predicate_to_wam(PI, Options, Text) :-
    compile_predicate_to_wam_items(PI, Options, Items),
    wam_items_to_text(Items, Text).
```

Tests that pin the exact text output continue to pass without
modification.

## 2. Item term shapes

The shapes match what every target's existing tokenizer + parser
already produces, so consumers that switch from text-parsing to
items don't have to relearn the term language. Strings are used for
register / functor names (matching the current parser output);
labels are bare atoms wrapping a name string.

```prolog
% Labels
label("foo/2")
label("L_foo_2_3")

% Get-arg-register instructions
get_constant(Value, "A1")          % Value is Integer/Atom/etc.
get_variable("X1", "A1")
get_value("X1", "A1")
get_structure("foo/2", "A1")
get_list("A1")
get_nil("A1")
get_integer(Int, "A1")

% Put-arg-register instructions
put_constant(Value, "A1")
put_variable("X1", "A1")
put_value("X1", "A1")
put_structure("foo/2", "A1")
put_list("A1")

% Unify instructions (in get_structure / get_list mode)
unify_variable("X1")
unify_value("X1")
unify_constant(Value)

% Set instructions (in put_structure / put_list mode)
set_variable("X1")
set_value("X1")
set_constant(Value)

% Control flow
call("bar/3", "3")
execute("bar/3")
proceed
allocate
deallocate
fail
cut_ite
jump("L_label_name")

% Choice point management
try_me_else("L_clause_2")
retry_me_else("L_clause_3")
trust_me

% Builtin / foreign
builtin_call("is/2", "2")
call_foreign("c_func", "3")

% Aggregate scope
begin_aggregate("count", "X1", "X2")
end_aggregate("X1")

% Indexing dispatch tables
switch_on_constant([Const-Label, ...])
switch_on_structure([FunctorAtom-Label, ...])
switch_on_term(VarLabel, ConstEntries, ListLabel, DefaultLabel)
```

The spec is canonical — these are the SHAPES the items predicate
returns. Targets MUST handle every shape they intend to support
(or fall back to the catch-all "unknown item" handling each one
already has).

## 3. Shared helpers

Three small predicates are exported alongside the API to handle
the operations that every target ends up writing on top of items:

```prolog
%% wam_items_walk(+Items, -Labels, -FlatInstrs)
%  Walks an items list, separating label declarations from
%  instructions. Returns Labels as a list of NameStr-PC pairs and
%  FlatInstrs as the instruction list (no labels). Equivalent to
%  the per-target "walk_blocks" helpers most targets already have.
wam_items_walk(Items, Labels, FlatInstrs).

%% wam_items_resolve_label(+Name, +Labels, -PC)
%  Looks up a label name in the Labels list from wam_items_walk/3.
%  Throws existence_error if not found.
wam_items_resolve_label(Name, Labels, PC).

%% wam_items_pretty_print(+Items)
%  Prints the items list in a human-readable diagnostic form to
%  current_output. Useful for debugging the generator pipeline.
wam_items_pretty_print(Items).
```

## 4. Implementation strategy

The wam_target internal pipeline currently has three layers:

1. **Per-clause compilation** (`compile_single_clause_wam`,
   `compile_multi_clause_wam`) — builds head/body code as text.
2. **Per-instruction emission** (~128 `format(string(...))` sites)
   — formats individual instruction lines.
3. **Aggregation** — concatenates clause text with `\n` between.

The refactor replaces these with an items-first pipeline:

1. **Per-clause compilation** returns a list of items.
2. **Per-instruction emission** returns a single item term (one
   helper per instruction shape:
   `mk_put_variable(Xn, Ai, put_variable(Xn, Ai))`).
3. **Aggregation** is `append/3` instead of string concatenation.

Then `wam_items_to_text/2` walks the items and emits the canonical
text form. The bodies of the existing per-instruction format calls
become the bodies of `wam_items_to_text`'s clauses — same output,
just driven from items instead of inline format calls.

### 4.1 Internal helper convention

To keep the refactor mechanical, each instruction-emission site
gets a corresponding constructor:

```prolog
% Old:
format(string(Code), "    put_variable ~w, ~w", [Xn, Ai]).

% New:
mk_put_variable(Xn, Ai, put_variable(Xn, Ai)).

% wam_items_to_text walks the item list:
item_to_text(put_variable(Xn, Ai), "    put_variable Xn, Ai") :-
    format(string(Out), "    put_variable ~w, ~w", [Xn, Ai]).
```

The format string moves from the emission site to the printer, but
its content is byte-identical.

## 5. Test coverage strategy

Two regression layers gate the refactor.

### 5.1 Text round-trip tests

For every existing test that pins specific text output, add an
assertion that:

```prolog
compile_predicate_to_wam_items(PI, Options, Items),
wam_items_to_text(Items, Text),
compile_predicate_to_wam(PI, Options, ExpectedText),
assertion(Text == ExpectedText).
```

If the items pipeline produces text that differs from the legacy
pipeline by even one byte, this fires. All existing tests that
match `compile_predicate_to_wam/3` output continue to pass — they
go through the new pipeline transparently.

### 5.2 Items-shape tests

A new test module pins the items shapes for representative
predicates: a single-clause fact, a multi-clause predicate with
indexing, a recursive predicate with try_me_else / trust_me, a
predicate with `is/2` and arithmetic compares (so the future ISO
sweep has a fixture to point at), one with `catch/3` so the
goal-as-data path is covered.

Each test compares the items list against an expected term list.
Forces us to keep item shapes stable.

## 6. Migration plan

### Phase 1 — items pipeline lands (this design's PR)

Sole change: refactor `wam_target.pl` to be items-first. Existing
text API keeps working byte-identically. No target migrates yet.

Expected diff: ~500-700 LOC in `wam_target.pl` (extracts each
emission site into a constructor + items-list build), ~200 LOC for
`wam_items_to_text/2` (the displaced format strings), ~100 LOC of
new tests (round-trip + items shape).

### Phase 2 — per-target migration

One PR per target, in roughly increasing order of parser
complexity (start with the smallest so the migration pattern
stabilises before tackling the heavyweights):

1. `wam_clojure_target.pl` (7 format calls) — proof of concept.
2. `wam_python_target.pl` (19).
3. `wam_cpp_target.pl` (22).
4. `wam_jvm_target.pl` (35).
5. `wam_fsharp_target.pl` (52).
6. `wam_elixir_target.pl` (56).
7. `wam_rust_target.pl` (74).
8. `wam_scala_target.pl` (80).
9. `wam_lua_target.pl` (83).
10. `wam_haskell_target.pl` (88).
11. `wam_r_target.pl` (120) — heaviest parsing, may split into 2
     PRs.

Each PR:

- Switches the target's `parse_wam_text/2` (or equivalent) to
  consume `compile_predicate_to_wam_items/3` directly.
- Deletes the target's tokenizer / parser code.
- All target-specific tests continue to pass (no observable change
  in generated source).

### Phase 3 — special-case targets

Four targets have low or zero `format(string(...))` counts and may
already be on a different pipeline:

- `wam_c_target.pl` (0 parses)
- `wam_ilasm_target.pl` (0 parses)
- `wam_go_target.pl` (2 parses)
- `wam_llvm_target.pl` (2 parses, uses `parse_wam_to_pass1`)
- `wam_wat_target.pl` (2 parses, uses `pass1_parse_predicates`)

Audit each: are they already structured? Do they bypass
`compile_predicate_to_wam/3` entirely? Document the answer; if
they'd benefit from items, migrate; if not, leave them alone.

## 7. Files touched (Phase 1 estimate)

- `src/unifyweaver/targets/wam_target.pl` — items refactor +
  printer + walk helpers (~700 LOC changed/added).
- `tests/test_wam_target.pl` (or new `test_wam_items.pl`) — round-
  trip + items-shape tests (~150 LOC).
- `docs/design/WAM_ITEMS_API_PHILOSOPHY.md` — this design.
- `docs/design/WAM_ITEMS_API_SPECIFICATION.md` — this design.

## 8. Phasing rationale

Why phase 1 alone before any target migration:

- The items pipeline + text printer are the **load-bearing
  refactor**. Text round-trip tests prove it works.
- Once landed, target migrations are independent and small. They
  can stack with other PRs (e.g. resume the ISO sweep on the C++
  target after Phase 2 PR #3 migrates wam_cpp_target).
- Each migration PR is small enough that bisecting any regression
  is trivial.

Why migrate small targets first:

- The migration pattern (delete tokenizer, switch import, run
  tests) needs to stabilise. Smallest target = simplest
  trial-by-fire.
- Heavyweight targets like wam_r and wam_haskell will surface
  edge cases. Better to handle those after the pattern is proven
  on three or four lighter targets.

## 9. ISO-sweep interaction

PR #3 of the ISO sweep (arith compares + `succ_iso/2`) is on hold
behind this refactor in the C++ target. After Phase 1 + Phase 2 PR
for wam_cpp_target lands, the ISO sweep ships against the items
API, with a single rewrite rule per ISO-aware key (instead of four
shape-specific rules per key). Net: the ISO sweep gets ~70 LOC
smaller.

If items-API migration takes too long, the ISO sweep can land
against text-parsing as designed. Doc updated to reflect whichever
order ships first.
