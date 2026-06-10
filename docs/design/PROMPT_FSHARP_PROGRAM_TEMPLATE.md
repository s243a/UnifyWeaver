# Prompt: Migrate F# Program.fs to Mustache Template

## Context

The F# WAM target generates `Program.fs` as a ~200-line Prolog
format string embedded in `generate_program_fs/4` in
`src/unifyweaver/targets/wam_fsharp_target.pl` (lines 4843-5049).

The template system (`src/unifyweaver/core/template_system.pl`)
now supports `{{match key}}{{case val}}...{{/match}}` blocks
for multi-valued dispatch, plus the existing `{{#flag}}`/`{{^flag}}`
sections and `{{name}}` substitution.

## Task

Extract Program.fs from the inline Prolog format string into
`templates/targets/fsharp_wam/program.fs.mustache`, using template
constructs for conditional logic.

### Prerequisite check

Before starting, verify template match/case tests pass:
```
swipl -q -g test_template_system -t halt src/unifyweaver/core/template_system.pl
```

If the match/case edge case tests have not been added yet, see
`docs/design/PROMPT_TEMPLATE_MATCH_TESTING.md` and do that first.

### Specific deliverables

1. **Create `templates/targets/fsharp_wam/program.fs.mustache`**
   with the Program.fs content, replacing:
   - `~w` format variables with `{{name}}` template variables
   - Prolog conditional logic with `{{#flag}}` and `{{match key}}`
   - See design doc `WAM_FSHARP_PROGRAM_TEMPLATE_MIGRATION.md`
     for the full variable table and match block examples

2. **Update `generate_program_fs/4`** in `wam_fsharp_target.pl` to:
   - Compute a Dict of template variables
   - Load the template via a new `fsharp_program_template_source/1`
   - Call `render_template/3`
   - The predicate should be ~20 lines, not ~200

3. **Add `fsharp_program_template_source/1`** following the pattern
   of `fsharp_lmdb_template_source/1` (line 4690+)

4. **Verify all existing tests pass**:
   ```
   swipl -g main tests/core/test_wam_fsharp_csr_smoke.pl
   swipl -g main tests/core/test_wam_fsharp_csr_bench.pl
   ```

5. **Verify generated output is equivalent** to pre-migration by
   diff-comparing generated Program.fs with both old and new paths

### Key files

- `src/unifyweaver/targets/wam_fsharp_target.pl` -- current
  `generate_program_fs/4` (lines 4843-5049), template loaders
  (lines 4690-4710)
- `src/unifyweaver/core/template_system.pl` -- `render_template/3`
- `templates/targets/fsharp_wam/*.mustache` -- existing templates
- `docs/design/WAM_FSHARP_PROGRAM_TEMPLATE_MIGRATION.md` -- design

### Watch out for

- Single quotes in F# code (e.g., `'\t'`) -- these are fine in
  template files but were `''\\t''` in the Prolog format string
- The template's `{{` delimiters don't conflict with F# syntax
  (F# uses `{ }` for records but not `{{ }}`)
- The format string has exactly 4 `~w` placeholders matching
  `[LmdbOpen, CsrOpen, ForeignPredsStr, LookupSourcesExpr]`
