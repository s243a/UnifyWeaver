# Template System Match/Case: Testing Plan

**Status**: Feature implemented, happy-path tested. Edge cases pending.
**Date**: 2026-05-25
**Feature location**: `src/unifyweaver/core/template_system.pl`
**Predicates**: `expand_match_blocks/3`, `case_matches/2`,
`find_match_block/5`, `parse_match_cases/3`, `resolve_match/5`

## 1. What was implemented

The template system was extended with `{{match key}}...{{/match}}`
blocks for multi-valued option dispatch:

```
{{match mode}}
{{case eager}}eager code
{{case cached}}cached code
{{default}}fallback
{{/match}}
```

Processing order in `render_template/3`:
1. `expand_match_blocks` (new) -- resolve match/case blocks
2. `expand_sections` (existing) -- resolve `{{#flag}}`/`{{^flag}}`
3. `render_template_string` (existing) -- substitute `{{name}}`

The `case_matches/2` predicate currently does exact string match.
It is factored out for future extension to glob/regex patterns.

## 2. What was tested (happy path)

Seven ad-hoc tests run via `swipl -q -g "..."`:

1. Basic exact match: `mode=cached` matches `{{case cached}}`
2. Default fallback: `mode=lazy` with no lazy case -> default
3. No match, no default: produces empty string
4. Variable substitution inside case body: `{{name}}` works
5. Key not in dict: falls through to default
6. Match inside section: `{{#enabled}}...{{match}}...{{/enabled}}`
7. Multiple match blocks in one template

## 3. What needs testing (edge cases)

### 3.1 Nesting: section inside case body

```
{{match mode}}
{{case cached}}
{{#has_l2}}L2 enabled{{/has_l2}}
{{^has_l2}}L2 disabled{{/has_l2}}
{{case eager}}
no cache
{{/match}}
```

Expected: when `mode=cached, has_l2=true`, renders "L2 enabled".
This tests that match expansion happens before section expansion
(the current processing order).

### 3.2 Nested match blocks

```
{{match outer}}
{{case a}}
  {{match inner}}
  {{case x}}AX{{case y}}AY
  {{/match}}
{{case b}}B
{{/match}}
```

Expected: when `outer=a, inner=y`, renders "AY".
Risk: the `find_match_block` parser finds the first `{{match`
and the first `{{/match}}` -- nested matches would close at the
wrong `{{/match}}`. This likely FAILS with the current
implementation and needs a fix (balanced-bracket parsing).

### 3.3 Case values with special characters

```
{{match backend}}
{{case lmdb_offset}}LMDB{{case sorted_array}}SORTED{{/match}}
```

Expected: `backend=lmdb_offset` renders "LMDB".
The `valid_tag_name` check requires alphanumeric + underscore for
section tags, but match keys go through `atom_string` -- verify
underscores work in case values.

### 3.4 Case values with hyphens/dots

```
{{match target}}
{{case wam-fsharp}}F#{{case wam-haskell}}Haskell{{/match}}
```

Expected: `target=wam-fsharp` renders "F#".
Risk: hyphens in case values may fail `atom_string` or the
`find_case_at` parser if `}}` is not found correctly.

### 3.5 Empty case body

```
{{match mode}}{{case skip}}{{case keep}}KEPT{{/match}}
```

Expected: `mode=skip` renders empty string; `mode=keep` renders "KEPT".

### 3.6 Match with only default

```
{{match anything}}{{default}}ALWAYS{{/match}}
```

Expected: always renders "ALWAYS" regardless of dict.

### 3.7 Whitespace handling

```
{{match mode}}
  {{case eager}}
    EAGER
  {{case lazy}}
    LAZY
{{/match}}
```

Expected: rendered output includes the whitespace/newlines around
the case body. Verify no spurious trimming.

### 3.8 Malformed blocks

- `{{match key}}` without `{{/match}}` -- should not crash
  (ideally passes through unchanged or produces empty)
- `{{match}}` without a key -- should not match
- `{{case}}` without a value -- should not match
- `{{/match}}` without opening -- should pass through

### 3.9 Real template file

Load a `.fs.mustache` or `.hs.mustache` file from disk that
contains match blocks. Verify `render_template/3` works on
file-loaded content (not just inline strings).

### 3.10 Integration with codegen

Generate an actual F# or Haskell project where the template
uses `{{match register_mode}}` to select between `run` and
`runMutableRegs`. Build with GHC/dotnet to verify the rendered
code compiles.

## 4. Where to add tests

Add test cases to `test_template_system/0` in
`src/unifyweaver/core/template_system.pl` (around line 850+).
Follow the existing pattern:

```prolog
write('Test N - Match/case description: '),
render_template(Template, Dict, Result),
(   expected_check(Result)
->  writeln('PASS')
;   format('FAIL: got ~w~n', [Result])
),
```

Also add a separate test file:
`tests/test_template_match_case.pl` for the edge cases that
might fail (nested matches, malformed blocks) -- these should
be isolated so failures don't block the main test suite.

## 5. Known limitations

- **Nested match blocks**: likely broken (first `{{/match}}`
  closes the outer block). Needs balanced-bracket parsing if
  we want to support this. Low priority -- codegen rarely
  needs nested match.
- **Same-key multiple matches**: if two `{{match mode}}` blocks
  appear in the same template, both are expanded independently.
  This works correctly (tested as "multiple match blocks").
- **Case ordering**: first matching case wins. No fall-through.
- **No pattern matching**: only exact string equality. Future
  extension via `case_matches/2` (documented in code).

## 6. References

- Implementation: `src/unifyweaver/core/template_system.pl`
  lines 400-555 (expand_match_blocks and helpers)
- Existing tests: `test_template_system/0` in same file
- Philosophy: comments in `case_matches/2` (line 550+)
  document future glob/regex extension points
