# SPECIFICATION: the `pattern_stache` template dialect, version 1

**Status: specification.** This document is normative for dialect version 1, implemented by
`src/unifyweaver/core/pattern_stache.pl`. It specifies **only what two witnessed consumers
needed** — goals → typed AST nodes and constraints-as-dispatch-keys, both prototyped in
`prototypes/mu_cosine/pattern_stache/` (kept intact as the historical witness). Everything the
consumers did not need is a **deliberate exclusion** with a revisit condition, listed at the end.
Rationale lives in [`STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`](STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md);
evidence lives in the two prototype reports
([one](../../prototypes/mu_cosine/REPORT_pattern_stache_prototype.md),
[two](../../prototypes/mu_cosine/REPORT_pattern_stache_second_consumer.md)).

## Purpose, input, output

**Purpose.** `pattern_stache` is a template dialect that dispatches on the *shape* of a Prolog
term instead of on the spelling of a string. A template file holds `{{match}}`/`{{case}}` blocks;
each `{{case}}` value is a Prolog term pattern; the first pattern that unifies with the
dispatched value is selected, and the variables it bound become available to that case's body.
It exists so that one case such as `has_type(X, substrate(C))` can serve every corpus, instead of
one hand-written case per concrete spelling.

**Input.** A `.stache` template file (text, mustache-shaped syntax, dialect header required) and
a dict — a list of `Key=Value` pairs where `Value` is a **ground Prolog term** (atoms and numbers
included; they are terms).

**Output.** A string: the template with the selected case bodies rendered and placeholders
substituted. Rendering never modifies the caller's dict.

**What it does not do.** It selects; it does not prove. A closed-fact constraint remains a
lookup, and a numeric constraint remains uncheckable by unification, whatever the matcher.
Structural matching improves *dispatch*, nothing else.

## The one table: `pattern_stache` is not mustache

The predictable conflation is between this dialect and the plain-mustache subset implemented by
`template_system.pl`, because both use `{{...}}` and both have `{{match}}`/`{{case}}` blocks.
They are different languages selected by file extension:

| | `.mustache` (string dialect, `template_system.pl`) | `.stache` (`pattern_stache`, this spec) |
|---|---|---|
| `{{case v}}` value | literal text, string-compared | **Prolog term**, read and matched by unification |
| `{{case Helpers}}` | matches the literal string `Helpers` | reads as a **variable — matches anything** (and is usually a load error, see overlap) |
| `{{case wam-fsharp}}` | matches the literal string `wam-fsharp` | reads as the compound `-(wam, fsharp)` — write `{{case 'wam-fsharp'}}` for the atom |
| dict values | atoms/strings; a compound throws `type_error(atom, _)` | ground terms, compounds welcome |
| variables bound by a match | no such concept | visible to the case body as `{{Var}}`/`{{q:Var}}` |
| header | none | `{{! dialect(pattern_stache, 1) }}` **required** |
| overlap between cases | impossible (exact strings) | checked at load (trichotomy below) |

No `.mustache` file can reach this dialect's parser and no `.stache` file can silently fall back
to the string parser: the extension selects the parser family, and a `.stache` file without the
header is an **error**, not a mustache file (PROJECT_PHILOSOPHY §4: the decision is in the
artifact, not in configuration).

*Acceptance test for this document* (PROJECT_PHILOSOPHY §9): a fresh reader — run the test with
the least capable reader that must succeed, a Haiku-tier model — should be able to explain the
dialect back from this document alone: what a case value is, what happens on a match, what the
header does, and what the table above distinguishes.

## Dialect selection: extension and header

1. A `pattern_stache` file has the extension **`.stache`**. The loader refuses any other
   extension (`not_a_stache_file`).
2. The **first non-empty line** must be the header pragma, a mustache comment whose content is a
   Prolog term:

   ```text
   {{! dialect(pattern_stache, 1) }}
   ```

   - Fixed position: a pragma appearing anywhere later is body text, never a header.
   - A `.stache` file without the header is an error (`missing_dialect_header`).
   - A version other than `1` is an error (`unsupported_dialect_version`) — the loader never
     parses an unknown version with the newest parser and hopes. Fail closed.
   - Blank lines before the header are permitted.

## Template syntax

The block syntax of the mustache subset, unchanged:

```text
{{match key}}
{{case Pattern}}body
{{case Pattern2}}body2
{{default}}fallback
{{/match}}
```

plus two interpolation forms usable anywhere placeholders are substituted:

| form | rendering | use |
|---|---|---|
| `{{Key}}` | `~w` — plain write | display text (witnessed by the AST-emission consumer) |
| `{{q:Key}}` | `~q` — quoted write | **re-readable** text, for output that is read back as a term (witnessed by the constraint-dispatch consumer); identical to `{{Key}}` on plain atoms and numbers |

A placeholder whose key is not in scope is left **verbatim** in the output (both forms) — the
same behaviour as `template_system.pl`, kept deliberately (see exclusions).

## Case patterns

A `{{case ...}}` value is read as a Prolog term with:

```prolog
read_term_from_atom(CaseText, Pattern,
                    [variable_names(VarNames), module(pattern_stache)])
```

- **Variables are named by their source text**; the `variable_names/1` list is the binding
  channel into the body. `_` is a true wildcard: it matches and binds nothing.
- **Operator table**: the standard table, pinned to the `pattern_stache` module — an `op/3`
  declared elsewhere in the process cannot change how a template parses.
- **Syntax errors are load errors**, wrapped as `bad_case_pattern(Text, syntax_error(Why))`.
- **Linearity is required**: a variable may occur at most once in a pattern. A repeated variable
  (`f(X, X)`) is a load error (`nonlinear_pattern`). This is a contract boundary, not a
  capability limit — see exclusions.

The witnessed pattern grammar is: **any first-order, linear, guard-free term**. The five shapes
the two consumers actually used, recorded here as the evidence base for that rule:

| shape | example | consumer |
|---|---|---|
| compound, depth 2, two variables | `has_type(X, substrate(C))` | both |
| sibling functor of the above | `has_type(X, judge(J))` | both |
| compound, depth 1, one variable | `non_amplifying(Op)` | both |
| compound, depth 1, two variables, numeric leaf | `mu_bounded(Op, B)` | both |
| mixed ground-leaf/variable | `has_type(X, substrate(pearltrees))` | constraint dispatch |

Quoted atoms restore literal semantics anywhere term reading would misparse an intended tag:
`{{case 'wam-fsharp'}}`, `{{case 'Helpers'}}`. The migration checklist in
`STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md` is normative here and enforced by tests: hyphenated
and uppercase-initial values misparse **silently** when unquoted, which is exactly why the
checklist exists.

## Matching semantics

Given `{{match key}}` and a dict:

1. **Lookup.** If `key` is absent from the dict, the `{{default}}` body is rendered with no new
   bindings (empty output if there is no default).
2. **Ground check.** The dict value must be ground. A nonground value is an error
   (`nonground_dispatch`) — dispatch may not bind the data. Consumers that carry residual
   (nonground) goals must route them away from dispatch upstream, which is where the mode
   discipline belongs; this check is the engine's backstop, not the mode system.
3. **Selection.** Cases are tried in file order. Each case's pattern (with fresh variables) is
   unified with the value; the **first** case whose pattern unifies is selected and
   **committed** — nothing that happens while rendering the body can reconsider the selection.
   A body whose nested content renders empty does not fall through to a later case; an error
   raised while rendering the body propagates.
4. **Binding scope.** The selected pattern's `Name=Value` bindings are prepended to the dict,
   forming a child scope for that case body only. A binding **shadows** an outer dict key of the
   same name, silently and lexically; after `{{/match}}` the outer key is visible again,
   untouched. Outer keys not shadowed remain visible inside the body.
5. **No match.** If no case unifies, the `{{default}}` body is rendered (with no new bindings);
   with no default, the block renders as the empty string.

## Overlap: checked at load, a trichotomy

For every ordered pair of cases (Pᵢ before Pⱼ) in a match block, at load time:

| relation | meaning | treatment |
|---|---|---|
| Pᵢ subsumes Pⱼ | Pⱼ can never fire | **error** `unreachable_case` |
| Pⱼ subsumes Pᵢ | specific-before-general refinement | allowed, silent |
| unifiable, neither subsumes | genuinely order-dependent | **warning**, first-match-wins |

The refinement row is load-bearing in a witnessed consumer (`substrate(pearltrees)` above
`substrate(C)`); a flat overlap ban would have rejected a real template. Identical patterns are
mutual subsumption and land in the first row. The check runs when the file is loaded — a
template with a dead case fails before any dict exists. Order-dependent overlap is *never*
silent: the warning names both cases.

## Errors (all fail closed)

| error term | raised when |
|---|---|
| `not_a_stache_file(Path)` | loader given a non-`.stache` path |
| `missing_dialect_header(Path)` | `.stache` file whose first non-empty line is not the pragma |
| `unsupported_dialect_version(Path, V)` | header version ≠ 1 |
| `bad_case_pattern(Text, syntax_error(W))` | case value does not read as a term |
| `nonlinear_pattern(Text)` | a variable occurs twice in one pattern |
| `unreachable_case(...)` | an earlier case subsumes a later one |
| `nonground_dispatch(Key, Value)` | dict value for a match key is nonground |

All are wrapped as `error(pattern_stache(...), _)`.

## Whitespace

Case bodies include the literal text between markers, newlines included. The engine does not
trim; the caller owns whitespace policy (both witnessed consumers trimmed per-rendering with
`normalize_space/2`). Mustache "standalone line" semantics are an exclusion, below.

## Deliberate exclusions

Each exclusion records a **revisit condition** (PROJECT_PHILOSOPHY §3: a condition to check, not
a date). None of these was needed by either witnessed consumer; per the graduation rule, a third
consumer shape the grammar cannot express stops the work and reopens the spec rather than
extending the grammar in place.

| excluded | v1 behaviour | revisit when |
|---|---|---|
| **non-linear patterns** (`f(X, X)`) | load error `nonlinear_pattern` | a consumer needs equality constraints between argument positions |
| **guards** (conditions on a case beyond its shape) | not expressible; the one guard-shaped question — groundness — is owned by the caller's discharge ordering | a consumer's dispatch decision provably cannot be expressed as a term shape plus caller-side routing |
| **nested `{{match}}` blocks** | outside the v1 contract; the implementation's scanner is depth-aware and current behaviour is characterized by tests, but v1 does not promise nesting semantics | a witnessed consumer template needs a second dispatch key |
| **missing-key / unbound-placeholder changes** | left verbatim, matching `template_system.pl` | a consumer needs fail-on-unbound rendering; would then be a new dialect version, since it changes output |
| **pattern arithmetic** | not expressible | never expected: numeric constraints are property-test obligations by design, not matcher work |
| **list patterns / store iteration** | the caller iterates; the template dispatches one term | a consumer that cannot iterate outside the template |
| **regex / glob cases** | absent | orthogonal to structure; would be its own extension per `template_system.pl`'s notes |
| **multi-key dispatch** | absent | a consumer that cannot state the second key in the case body (both witnessed consumers could) |
| **partials / delegation** (`{{> name}}`) | absent | the two-live-files divergence hazard in the philosophy doc materializes |
| **`.mustache` → `.stache` converter** | not built | per [`docs/TODO_STACHE_CONVERTER.md`](../TODO_STACHE_CONVERTER.md): a match-using library grows to the low tens of cases |

## Relationship to the string dialect

`template_system.pl` and its 67 dependent targets are untouched and remain the default for every
`.mustache` file. This dialect is **additive**: a new extension selects the new parser, so no
existing file can reach it by construction, and no configuration option exists that could route
a `.mustache` file here. The two dialects share no code path; the production module carries its
own scanner.
