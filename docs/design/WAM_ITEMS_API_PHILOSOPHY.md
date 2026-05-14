# WAM Items API — Philosophy

How WAM target generators should consume the WAM compiler's output.
The current text-based interface forces every target to ship its own
parser, which is wasteful, error-prone, and slow. This doc lays out
why we want a structured-items API instead, what we'd lose by
moving away from text-as-canonical, and how the migration plays out
across the 15 (and counting) WAM targets in the tree.

Companion doc:
- `WAM_ITEMS_API_SPECIFICATION.md` — the API shape we ship and the
  per-target migration plan.

## 1. The current pipeline

```
user Prolog
   │
   ▼
wam_target.pl (introspection-driven WAM compiler)
   │  ─┐
   │   │  format(string(...), "...", [...]) emits TEXT one fragment
   │   │  at a time, concatenates, returns a single multi-line string.
   ▼  ─┘
"foo/2:\n    allocate\n    put_variable Y1, A1\n    call bar/1, 1\n..."
   │
   ▼  (per target)
target-specific tokenizer + parser
   │  ─┐
   │   │  Splits on whitespace + commas, recognises label vs
   │   │  instruction lines, builds compound terms like
   │   │  put_variable("Y1", "A1") for each instruction.
   ▼  ─┘
[ label("foo/2"), allocate, put_variable("Y1", "A1"),
  call("bar/1", "1"), proceed, ... ]
   │
   ▼  (per target)
target language emitter
   │  ─┐
   │   │  Walks the items list, emits target-language source
   │   │  (C++ struct literals, Rust match arms, Lua table
   │   │  entries, Haskell data ctors, ...).
   ▼  ─┘
target-language source code
```

The middle two steps cancel out. The compiler internally builds
structured data (it has to — it's making bind decisions, register
allocation, etc.), then `format`s it into text. Every target then
parses that text back into structured data. **The structure exists,
gets serialized, then gets re-parsed** — and the round-trip is
where the bugs hide.

## 2. The problem isn't theoretical

The C++ target shipped a quote-aware tokenizer in PR #2076 (the
`format/2` PR) because the original tokenizer corrupted format
strings containing spaces. That fix landed cleanly. Then in PR
#2084 (the first ISO builtin) a *second* tokenizer bug surfaced:
the conjunction functor `,/2` was being split into `/2` because
bare comma was treated as a separator. **That bug had been there
since the C++ target's first commit.** It only mattered when a
predicate's WAM contained `put_structure ,/2, ...` — uncommon
enough to escape every existing test, common enough that the first
real `catch((A, B), ...)` use would have hit it.

Both bugs are signs of the wrong abstraction boundary. A serialized
text format is the wrong contract between a compiler and 15
consumers.

Other costs:

- **Speed.** `format(string(X), "...", [...])` followed by per-line
  `split_string`, character classification, regex-style pattern
  matching, atom→string conversions. None of it is needed if we
  hand structured terms across.
- **Code size.** ~80 lines of tokenizer in wam_cpp_target. ~120 in
  wam_r_target. Multiplied by 15 targets, that's roughly 1k lines
  of redundant parsing logic.
- **Drift.** Every target has its own parser, so each has slightly
  different bugs. The existing C, Rust and Haskell targets have
  *also* hit format-string bugs in the past. Nothing in the design
  prevents new ones in tomorrow's target.
- **Coupling to print format.** When the WAM compiler changes how
  it prints something — say, adding a new operand format —
  every consumer needs to update its parser. With a typed item, the
  same change is one edit to the term shape definition.

## 3. Three options considered

### 3.1 Move parsing to wam_target (rejected)

Centralise the existing tokenizer + parser inside `wam_target.pl`,
expose `parse_wam_text/2` as a public predicate, have every target
call it. One parser instead of 15.

**Pro:** No invasive changes to the WAM compiler internals. Targets
get a clean items list. Bug fixes happen in one place.

**Con:** The text serialization round-trip is still there. Speed
doesn't improve. The "wrong abstraction" diagnosis from §2 isn't
addressed — we're just moving the wrong abstraction from the leaves
to the root. And we'd be locking in the text format as part of the
public API forever.

### 3.2 Refactor wam_target to emit items, derive text (chosen)

The WAM compiler's internals already build structured data. Refactor
the leaf emission functions so they return items, have aggregating
functions concatenate item lists, and expose a new public predicate
`compile_predicate_to_wam_items/3` that returns the structure
directly. Add a separate **printer** `wam_items_to_text/2` so the
existing text format stays available for debug dumps and any
external tool that consumes it.

**Pro:** Eliminates the round-trip entirely. Each target consumes
items directly with no parser. Speed wins, code shrinks, an entire
class of bugs disappears. The text printer is one well-defined
predicate that the existing tests can pin down — text behavior is
guaranteed unchanged.

**Con:** It's an actual refactor of `wam_target.pl` (~1975 lines,
128 emission sites). Not trivial. Requires careful test coverage
to make sure the text printer produces byte-identical output to the
current text emission.

### 3.3 Parallel pipeline (rejected)

Add `compile_predicate_to_wam_items/3` as a *separate*
implementation that walks the same input and produces items
directly, without touching the existing text path. Two pipelines
live side by side; they need to stay in sync manually.

**Pro:** No risk to existing text consumers.

**Con:** Code duplication is the worst of both worlds. Bugs in one
path don't surface in the other; new WAM features land twice.

## 4. Why this matters for ISO errors specifically

The ISO-errors PR series (#2079 docs, #2081 plumbing, #2084
`is_iso/2`) had to extend the C++ target's per-predicate rewrite
across **four item shapes** — `builtin_call`, `put_structure`,
`call`, `execute` — because `is/2` can appear in any of them
depending on context. With items as the contract, that rewrite is
a one-line `swap_key_in_item/3` walk over a typed list. With text,
each new shape is "what does the WAM printer write here? hope I
didn't miss a case."

PR #3 of the ISO sweep (arith compares + `succ_iso/2`) needs the
same multi-shape rewrite for `>/2`, `</2`, `>=/2`, `=</2`,
`=:=/2`, `=\=/2`, `succ/2`. That's seven more keys × four shapes
= 28 rewrite rules to write defensively in text-land, vs 14 table
entries in items-land.

The items API isn't a prerequisite for shipping the ISO sweep, but
**doing the items refactor first** would make the sweep half the
size and its rewrite rules trivially auditable.

## 5. Migration scope

15 WAM targets, all on the text API today. Migration is opt-in per
target — each one switches independently when there's appetite,
since the text API stays available throughout.

Estimated cost per target:

| Target | Format/parse calls | Migration effort |
|---|---:|---|
| wam_cpp_target.pl | 22 | ~1 PR (~80 LOC removed) |
| wam_clojure_target.pl | 7 | ~1 PR (small) |
| wam_python_target.pl | 19 | ~1 PR |
| wam_jvm_target.pl | 35 | ~1 PR |
| wam_elixir_target.pl | 56 | ~1 PR |
| wam_fsharp_target.pl | 52 | ~1 PR |
| wam_rust_target.pl | 74 | ~1 PR |
| wam_scala_target.pl | 80 | ~1 PR |
| wam_lua_target.pl | 83 | ~1 PR |
| wam_haskell_target.pl | 88 | ~1 PR |
| wam_r_target.pl | 120 | ~1-2 PRs (heaviest parsing) |
| wam_c_target.pl | 0 (no parser) | already structured? — audit |
| wam_ilasm_target.pl | 0 | already structured? — audit |
| wam_go_target.pl | 2 | already structured? — audit |
| wam_llvm_target.pl | 2 | uses `parse_wam_to_pass1` — own path |
| wam_wat_target.pl | 2 | uses `pass1_parse_predicates` — own path |

The "0-or-2-format-calls" group is interesting — it suggests some
targets already have an alternative pipeline. Worth auditing to
understand what they're doing and whether the items API helps or
duplicates their work.

## 6. Backwards compatibility

The text *path* stays. The text *default* changes.

The bare-name `compile_predicate_to_wam/3` becomes a generic
dispatch wrapper (see SPECIFICATION §1) selecting items or text
via an `output/1` option. **Default is items**, because items is
the new primary path and the dominant caller (every WAM target)
is going to migrate to it anyway. Forcing every text consumer to
say so explicitly keeps the cheap path the easy one.

Migration is mechanical:

- Tests that pin specific text output (a small handful) switch to
  `compile_predicate_to_wam_text/3` or pass `output(text)` —
  one-line edit per call site.
- Existing target generators that treat the result as text
  continue working in Phase 1 because their migration PR (Phase
  2) deletes both the parser AND the bare-name call together,
  switching to `compile_predicate_to_wam_items/3` directly.
- The text format itself is byte-identical — `wam_items_to_text`
  preserves the existing emission down to whitespace.

The text API as a *capability* is permanent. The text API as a
*default* is replaced by the items default.

## 6.1 Common parser as a complement

The items API removes the **need** to parse for in-process target
generation: the compiler and the target emitter both speak items
natively, so text never enters the pipeline. But parsing isn't
gone from the world — three cases keep it relevant:

- **External WAM input.** If a user hand-writes WAM text or
  imports a dump from another tool, a target needs to parse it
  before emitting.
- **Debug dump round-tripping.** Tooling that consumes
  `compile_predicate_to_wam_text/3` output (CI snapshot tests,
  WAM diff tools) may want to re-parse for analysis.
- **Target-specific WAM extensions.** A target that introduces
  custom instructions still needs a parser for the standard subset
  + special handling for its extensions.

Today every target that touches text ships its own parser — the
same redundancy the items API removes for in-process flows. The
fix for the parsing case is **one shared parser**, exposed from
`wam_target.pl` alongside the items API:

```prolog
%% wam_text_to_items(+Text, -Items)
%  Inverse of wam_items_to_text/2. Parses canonical WAM text into
%  the structured items list. Used by any target that needs to
%  consume text it didn't generate itself.
wam_text_to_items(Text, Items).
```

Symmetry:

| Source | Predicate | Output |
|---|---|---|
| Prolog clauses (compile) | `compile_predicate_to_wam_items/3` | Items |
| Prolog clauses (compile) | `compile_predicate_to_wam_text/3` | Text |
| Existing text (parse) | `wam_text_to_items/2` | Items |
| Existing items (print) | `wam_items_to_text/2` | Text |

Targets always consume **items** — the difference is whether items
came from compilation or from parsing. The 15 per-target parsers
collapse into one shared `wam_text_to_items/2` that every target
imports. Per-target parsers shrink to extension-recognising shims
(or disappear entirely for targets that don't extend the
instruction set).

For *finer-grained sharing* — targets with custom instructions
that need to compose their own parser around the standard set —
we also expose the building blocks:

```prolog
%% wam_tokenize_line(+Line, -Tokens)
%  The quote-aware whitespace+selective-comma tokenizer. Reusable
%  primitive for any custom parser.
wam_tokenize_line(Line, Tokens).

%% wam_recognise_instruction(+Tokens, -Item)
%  Given a token list, recognise a standard WAM instruction and
%  return the corresponding item term. Fails (no exception) if the
%  tokens don't match a standard instruction — caller can fall
%  back to its own recogniser for extensions.
wam_recognise_instruction(Tokens, Item).

%% wam_recognise_label(+Tokens, -LabelName)
%  Companion to wam_recognise_instruction/2 for label lines.
wam_recognise_label(Tokens, LabelName).
```

Targets with extensions compose:

```prolog
my_target_recognise(Tokens, Item) :-
    (   wam_recognise_label(Tokens, Name)
    ->  Item = label(Name)
    ;   wam_recognise_instruction(Tokens, Item)
    ->  true
    ;   my_target_extension_recognise(Tokens, Item)
    ).
```

This is layered: the simple case (no extensions) gets a one-line
import of `wam_text_to_items/2`. The advanced case (extensions)
opts into the building blocks and adds its own clauses. Either
way, the bug-prone tokenizer and the bulk of instruction-shape
recognition lives in **one** place.

## 7. What's intentionally out of scope

- **Removing the text API.** Even after every target migrates, the
  text dump is useful for debugging, education, and any third-party
  tool that wants to consume WAM. Keep it.
- **Renaming the existing text predicates.**
  `compile_predicate_to_wam/3` keeps its name and semantics. The
  new predicate gets a parallel name (suggested:
  `compile_predicate_to_wam_items/3`).
- **Forcing all targets to migrate at once.** Each PR moves one
  target. The unmoved targets keep working.
- **Re-designing the item shapes.** The current de-facto shapes
  (`put_variable("Y1", "A1")`, etc.) used by every target's parser
  ARE the item shapes. The spec writes them down; they don't
  change.

## 8. Open questions

These are flagged for review during the implementation PR, not now.

- **Should the items list also carry source-map info** (Prolog
  source line / column for each item)? Useful for debugger
  integration. Not needed for v1.
- **Should we ship a JSON/IR dump** alongside the text format for
  external tooling consumption? Probably not — items API is for
  in-process consumers; external tools can use the text dump
  forever.
- **Do we want a `validate_items/1` predicate** that checks an
  items list is well-formed (label-targets resolve, register names
  are syntactically valid, etc.)? Useful as a debugging aid for
  target authors. Not strictly necessary.
