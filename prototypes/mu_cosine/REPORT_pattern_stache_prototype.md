# Report: pattern_stache structural template dispatcher — prototype

*Prototype answering the six open questions in
[`docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`](../../docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md)
("Is this document enough to build from?") with running code rather than guesses. The motivating
consumer is goals → typed AST nodes from
[`DESIGN_desugaring_to_prolog_goals.md`](DESIGN_desugaring_to_prolog_goals.md) §4.1.*

**Status: prototype. Nothing in `src/unifyweaver/core/` or `templates/` was touched.** The
dispatcher lives entirely in [`pattern_stache/`](pattern_stache/) and shares no code path with
core `case_matches/2`; the block scanner was *copied* from `template_system.pl`, not imported,
precisely so nothing here can reach the 67 targets.

## What was built

| file | contents |
|---|---|
| `pattern_stache/pattern_stache.pl` | the dispatcher: loader (`load_stache_file/2`), load-time checks, renderer (`render_stache/3`), worked-example driver (`goals_to_ast/3`, `demo/0`) |
| `pattern_stache/goal_to_ast.stache` | the worked example — the goals→AST template from the philosophy doc's "What it would look like" section, verbatim plus one `judge` and one `mu_bounded` case |
| `pattern_stache/test_pattern_stache.pl` | 38 plunit tests, one block per question, plus a characterization suite run against the real core renderer (read-only) |

Run (SWI-Prolog, from `prototypes/mu_cosine/pattern_stache/`):

```bash
swipl -g run_tests -t halt test_pattern_stache.pl   # 38 tests
swipl -g pattern_stache:demo -t halt pattern_stache.pl
```

Demo output, end to end:

```text
has_type(x,substrate(pearltrees))
    => TypeNode { term: x, kind: "substrate", corpus: pearltrees }
has_type(j1,judge(sonnet))
    => TypeNode { term: j1, kind: "judge", judge: sonnet }
non_amplifying(min)
    => ConstraintNode { kind: "non_amplifying", op: min }
mu_bounded(path,0.5)
    => ConstraintNode { kind: "mu_bounded", op: path, bound: 0.5 }
in_support(decay,d)
    => /* unhandled goal: in_support(decay,d) */
```

The last line is the block that is unreachable with string matching — the whole point of the
exercise — and it interpolates the *entire dict value as a term* (`{{goal}}`).

## The patterns actually needed

This is the finding the spec should be built from. The real consumer template needed exactly
**four pattern shapes**, all of one kind:

1. `has_type(X, substrate(C))` — compound, nesting depth 2, two variables;
2. `has_type(X, judge(J))` — same shape, sibling functor (disjoint by functor alone);
3. `non_amplifying(Op)` — compound, depth 1, one variable;
4. `mu_bounded(Op, B)` — depth 1, two variables, one binding a *numeric* leaf (`0.5`).

Plus one interpolation form the doc's example already contained: `{{goal}}`, the whole matched
value rendered as a term in the `{{default}}` body.

**Everything needed was first-order, linear, guard-free unification against a ground term.** The
consumer did *not* need: guards, arithmetic in patterns, non-linear patterns (repeated
variables — supported by construction via unification, never exercised), list patterns, string
patterns, anonymous `_` (works, tests only), quoted-atom literal cases (works, tests only),
operator patterns like `A-B` (works, tests only), nested `{{match}}` (works, tests only),
sections `{{#}}`/`{{^}}` (not implemented, never missed), or partials. A specification that
stops at "case value = readable Prolog term, matched by unification, bindings scoped to body"
covers the entire observed demand.

One pattern the prototype needed that no design doc mentioned: **whitespace discipline**. Case
bodies carry the newlines around their markers, so the raw render of each goal arrives padded;
the driver strips with `normalize_space/2`. A real spec has to say who owns surrounding
whitespace (mustache's "standalone lines" rule, a trim flag, or caller-trims-as-here).

## Answers to the six questions

### 1. Binding propagation — scoped child dict by prepending; shadowing is lexical and silent

The pattern is read once with `variable_names(VarNames)`, giving `Name=Var` pairs. At dispatch,
`copy_term(Pattern-VarNames, P-Vs)` then `P = Value` instantiates the copies; `Vs` — now a list
of `'C'=pearltrees`-style pairs — is **prepended** to the incoming dict:
`append(VarNames, Dict, ChildDict)`. Every lookup in the renderer is `member/2`, so a prepended
binding shadows an outer key of the same name *for the body only*; the caller's dict is never
modified, and text after `{{/match}}` sees the outer value again.

Evidence (`q1_bindings`): with dict `[g=f(inner_val), 'C'=outer_val]` and case `f(C)`, the body
renders `in:inner_val` and the text after the block renders `out:outer_val`. Outer keys remain
visible inside the body (child scope extends, does not replace), and a nested `{{match}}` can
dispatch on an outer binding.

Shadowing is **silent**, as in every host language's lexical scoping. A load-time "pattern
variable collides with a conventional dict key" lint would be cheap to add but was not needed.

### 2. Term reading — `read_term_from_atom/3`, `variable_names/1`, module-pinned operator table

```prolog
read_term_from_atom(CaseText, Pattern,
                    [variable_names(VarNames), module(pattern_stache)])
```

- **Variable naming**: names come from the source text via `variable_names/1`; that list *is*
  the binding channel of Q1, so no separate naming scheme exists to design. `_` produces no
  pair — it is a true wildcard that matches without binding (`anonymous_var_binds_nothing`).
- **Operator table**: `module(pattern_stache)` pins reading to this module's table — the
  standard one — so an `op/3` declared anywhere else in the process cannot change how a
  template parses. Standard operators work (`{{case A-B}}` dispatches on `3-4`).
- **Errors**: the reader throws on syntax errors; the prototype wraps them as
  `error(pattern_stache(bad_case_pattern(Text, syntax_error(Why))), _)` so the message names the
  offending case text. Verified: `{{case f(}}` raises exactly that.
- Quoting restores literal semantics for migration-checklist shapes:
  `{{case 'wam-fsharp'}}` matches the atom `'wam-fsharp'` (tested), confirming the philosophy
  doc's measurement table.

### 3. Overlap — detected at load, and it is a trichotomy, not the doc's binary

The philosophy doc offered two options: forbid overlap at load, or version the order. The
prototype found that `subsumes_term/2` splits "overlap" into three cases that deserve different
treatment:

| relation between earlier case Pᵢ and later case Pⱼ | meaning | prototype behaviour |
|---|---|---|
| Pᵢ subsumes Pⱼ (e.g. `f(X)` before `f(a)`) | Pⱼ can **never** fire | load-time **error** `unreachable_case(case_index(J), subsumed_by(case_index(I)), …)` |
| Pⱼ subsumes Pᵢ (e.g. `f(a)` before `f(X)`) | specific-before-general refinement — the idiom of every match construct | **silent**, allowed |
| unifiable, neither subsumes (e.g. `p(a,Y)` then `p(X,b)`) | genuinely order-dependent | **warning** on stderr; first match wins |

A flat "forbid overlapping cases" would outlaw the refinement idiom, which is exactly how a
specific corpus case would be written above a general one; a flat "accept ordering" would let
dead cases ship silently. Detection runs in `check_stache_template/1` at **load**, walks nested
blocks, and needs no dict — a broken template fails when loaded, not when the dead case is first
wanted. Variant duplicates (`f(X)` then `f(Y)`) are caught by the same rule (mutual subsumption
⇒ first row fires). Cost: O(cases²) unifiability checks per block at load; irrelevant at the
observed 2–5 cases per file.

### 4. Determinism — commit at unification; confirmed, and justified

The dispatch is:

```prolog
(   member(case(Pattern0, VarNames0, CaseBody), Cases),
    copy_term(Pattern0-VarNames0, Pattern-VarNames),
    Pattern = Value
->  Body = CaseBody, append(VarNames, Dict, ChildDict)
;   Body = Default, ChildDict = Dict
)
```

The if-then-else condition contains **only** the case enumeration and the unification test;
`->` prunes the `member/2` choice points the instant one pattern unifies, and body expansion
happens after the commit. Two tests pin the consequences:

- a committed body whose nested match produces nothing (no case matches, no default) renders
  empty — the dispatcher does **not** reconsider a later case that also matches
  (`no_backtracking_into_case_selection`: value `p(a,b)` against `p(a,Y)` then `p(X,b)` yields
  the first body, empty inner and all);
- an **error** raised while rendering a committed body (a malformed nested case, parsed lazily)
  propagates; there is no fallthrough to the next case (`body_error_propagates_no_fallthrough`).

Justification, not just confirmation: if body success could influence selection, (a) case
selection would depend on arbitrarily deep rendering success, making output unpredictable from
the patterns on the page; (b) the Q3 overlap analysis would be meaningless, because reachability
would no longer be decidable from patterns; (c) errors in bodies would be silently converted
into different dispatch results. It also mirrors core `resolve_match/5` (`member/2` under `->`)
by construction, so string-mode and pattern-mode dispatch share one determinism story.

### 5. Loader dispatch — neither option as posed; the dialect needs its own load step, then a render branch

The question offered "a new `try_source/4` strategy, or a branch in `render_named_template/4`".
Building it showed the first is the wrong layer: `try_source/4` strategies answer *where the
template text comes from* (file / cache / generated), while the dialect answers *how the text is
parsed* — and cached and generated templates have no file extension at all, so a
`try_source(Name, stache, …)` clause would conflate provenance with dialect and leave non-file
sources dialect-less.

What the prototype actually needed:

- `load_stache_file/2` — enforces the `.stache` extension, requires
  `{{! dialect(pattern_stache, 1) }}` as the **first non-empty line** (a pragma later in the
  file is not a header — tested), reads the pragma with `term_string/2` (no bespoke parser),
  **fails closed** on a headerless file and on an unknown version
  (`unsupported_dialect_version`), and runs the Q3 load-time checks;
- the loaded template is `stache(Version, Body)` — a tagged term, so the renderer branches on
  the *tag*, not on any config option.

Mapped onto core, that means: `find_template_file/3` (which knows the path) records the dialect
at load; `render_named_template/4` branches on the loaded template's tag. No `structural(true)`
config option exists anywhere — the file marks itself, which is the by-construction guarantee
the philosophy doc asked for. Loading a `.mustache` path through the stache loader is an error
(`not_a_stache_file`), keeping the two parsers unreachable from each other's files.

### 6. Dict contract — values become arbitrary *ground* terms; existing callers are safe because the old path fails loudly

Under `pattern_stache`, dict values are terms and must be **ground at dispatch time**. The
prototype enforces this with `error(pattern_stache(nonground_dispatch(Key, Value)), _)` — a
nonground value could be bound *by* the pattern, i.e. dispatch would decide the data instead of
the data deciding the dispatch, which is precisely the residual-goal hazard §4.1's rule
("dispatch only on goals guaranteed discharged") exists to prevent. The prototype makes that
rule mechanical.

Would term values break existing callers? Characterized against the real core module
(read-only, `core_characterization` suite):

- `render_template/3` with a compound dict value **throws `type_error(atom, …)`** — in the match
  path (`resolve_match/5` stringifies with `atom_string/2`) and even in plain substitution
  (`render_template_string/3` does the same). So nothing about the existing atom/string contract
  changes for existing callers, and a term accidentally passed to the *old* renderer fails
  immediately and loudly rather than mis-rendering.
- Legacy-shaped calls work unchanged in the new dialect: `[mode=cached]` against
  `{{case cached}}` matches, because atoms are terms and read as themselves.

Interpolation renders bound terms with `~w` (plain write): atoms appear as today, compounds
print readably (`in_support(decay,d)`). This is **not re-readable** for atoms needing quotes
(`'hello world'` loses its quotes); the worked example never needed quoted output, so a
`{{q:X}}`-style quoted-interpolation sigil is recorded as future work rather than designed now.

## What the philosophy doc got wrong

1. **"The mustache rule for a missing key is to render the empty string."** Not in this
   codebase: the core renderer leaves unknown placeholders **verbatim** (test:
   `render_template("corpus is {{C}}", [], R)` gives `"corpus is {{C}}"`). So the misparse
   failure mode is visible template text in the output, not invisible holes — more detectable
   than the doc feared. The dialect-marking argument survives, but its premise needed
   correcting.
2. **"A structural template fed to the string parser does not fail."** Only half true. With a
   *term-valued* dict the string parser fails **loudly** — `atom_string/2` throws
   `type_error(atom, …)` at the first value it touches (tested). The silent case is real but
   narrower: an *atom*-valued dict against a structural template silently takes the default.
   The hazard is confined to the atom-dict corner, not the general case.
3. **The overlap options were posed as binary** (forbid at load / version the order). Subsumption
   analysis yields a finer, strictly better trichotomy — hard error only for unreachable cases,
   silence for the specific-before-general idiom, warning for genuine order dependence — that
   neither of the doc's options captures (§Q3 above).
4. Everything the doc *measured* held up: hyphenated values read as `-/2` compounds,
   uppercase-initial values read as variables, quoting restores both (all re-verified in tests),
   and `member/2`-under-`->` is indeed already the commit behaviour.

## Does structural dispatch earn its keep?

**For this consumer: yes, narrowly — and the earning is entirely where the doc predicted.** The
worked example's value comes from exactly one mechanism: a case head binding variables that the
body interpolates. Matching alone (without binding) would have added nothing over strings here,
confirming the doc's central claim. The pattern language required is the *minimum imaginable* —
first-order unification against ground terms — so the spec, when written, should resist every
extension (guards, arithmetic, regex-in-patterns) until a consumer produces a pattern that needs
it; this prototype produced none.

Costs found in practice: the copied block scanner is the bulk of the code (the dispatcher itself
is small, as the doc guessed — "the predicate is the small part" was right, though it's the
*reading and scoping*, not the scanner, that carries the design weight); whitespace ownership
needs a rule; term interpolation needs a quoting story before templates can emit *re-readable*
terms rather than display text.

---

*Addendum: the second consumer (constraints as dispatch keys) has since been prototyped against
this same dispatcher — see
[`REPORT_pattern_stache_second_consumer.md`](REPORT_pattern_stache_second_consumer.md). The
quoting story deferred above became that consumer's one real demand and is now implemented as
`{{q:Key}}`; the pattern shapes found here held.*
