# Prompt: Template System Match/Case Testing

## Context

The template system (`src/unifyweaver/core/template_system.pl`) was
extended with `{{match key}}{{case val}}...{{default}}...{{/match}}`
blocks for multi-valued option dispatch. The feature is implemented
and happy-path tested (7 cases) but edge cases are untested.

## Task

Add comprehensive tests for the match/case feature. Follow the
testing plan in `docs/design/WAM_TEMPLATE_MATCH_CASE_TESTING.md`.

### Specific deliverables

1. **Add tests to `test_template_system/0`** in
   `src/unifyweaver/core/template_system.pl` (around line 850+).
   Cover sections 3.1 through 3.8 of the testing plan.

2. **Create `tests/test_template_match_case.pl`** for isolated
   edge case tests (nested matches, malformed blocks) that might
   fail and need fixes.

3. **Fix any bugs found** -- particularly:
   - Nested match blocks (section 3.2) likely fail because
     `find_match_block` finds the first `{{/match}}` instead of
     the balanced closing one. Fix with depth-counting or
     recursive parsing if needed.
   - Case values with hyphens (section 3.4) may fail atom parsing.
   - Malformed blocks (section 3.8) should not crash.

4. **Run `swipl -q -g test_template_system -t halt`** to verify
   all existing + new tests pass.

### Key files

- `src/unifyweaver/core/template_system.pl` -- implementation +
  existing tests (search for `expand_match_blocks`, `case_matches`,
  `test_template_system`)
- `docs/design/WAM_TEMPLATE_MATCH_CASE_TESTING.md` -- full testing
  plan with expected results for each case

### Architecture notes

- `render_template/3` pipeline: match blocks -> sections -> substitution
- Match expansion runs first so case bodies can contain `{{#flag}}`
  sections and `{{name}}` variables
- `case_matches/2` is the extensibility point for future glob/regex
  support (currently exact string match only)
- The template system is used by both Haskell and F# WAM targets
  for kernel templates, and will soon be used for Program.fs/Main.hs

### What NOT to change

- Don't change the `render_template/3` signature or pipeline order
- Don't change existing section (`{{#flag}}`) behavior
- Don't add glob/regex to `case_matches/2` yet (document as future)
- Don't modify any WAM target files -- this is template-system only
