# F# Program.fs Template Migration: Design

**Status**: Implemented. Template created, generate_program_fs/4 refactored.
**Date**: 2026-05-25
**Prerequisite**: `WAM_TEMPLATE_MATCH_CASE_TESTING.md` edge cases pass

## 1. Problem

The F# WAM target generates `Program.fs` as a ~200-line Prolog
format string inside `generate_program_fs/4` in
`src/unifyweaver/targets/wam_fsharp_target.pl` (lines 4843-5049).

Issues:
- Hard to read and edit (F# code inside single-quoted Prolog atom)
- Quoting hell (`''` for literal single quotes, `~w` for variables)
- No syntax highlighting in editors
- Conditional code uses Prolog `format` arguments (`~w`) computed
  before the format call, not template-level conditionals

## 2. Proposed solution

Move Program.fs to `templates/targets/fsharp_wam/program.fs.mustache`
using the template system's `{{name}}`, `{{#flag}}`, and
`{{match key}}` constructs.

### Template variables

| Variable | Source | Example |
|---|---|---|
| `{{foreign_preds}}` | `format_foreign_preds_fs/2` | `"category_ancestor/4"` |
| `{{lookup_sources_expr}}` | `generate_lookup_sources_expr_fs/2` | `Map.ofList [...]` |
| `{{module_opens}}` | conditional on options | `open CsrReader` |

### Template conditionals

```
{{#has_csr}}
open CsrReader
{{/has_csr}}

{{#has_lmdb}}
open LmdbFactSource
{{/has_lmdb}}
```

### Template match blocks

```
{{match materialisation}}
{{case eager}}
    let parentsInterned = loadCategoryParent env |> ...
{{case cached}}
    let cachedSrc = TwoLevelCachedLookupSource(...)
{{case lazy}}
    let lazySrc = LmdbCursorLookup(env, "category_parent")
{{/match}}
```

## 3. Migration steps

1. Extract the format string content to a new file:
   `templates/targets/fsharp_wam/program.fs.mustache`

2. Replace `~w` format variables with `{{name}}` template variables

3. Replace conditional Prolog logic with `{{#flag}}` sections
   and `{{match key}}` blocks

4. Update `generate_program_fs/4` to:
   - Compute template variables into a Dict
   - Load the template file
   - Call `render_template/3`

5. Follow the pattern of `fsharp_lmdb_template_source/1` and
   `fsharp_csr_template_source/1` for loading templates

## 4. Template variable computation

```prolog
generate_program_fs(Predicates, DetectedKernels, Options, Code) :-
    pairs_keys(DetectedKernels, ForeignKeys),
    format_foreign_preds_fs(ForeignKeys, ForeignPredsStr),
    generate_lookup_sources_expr_fs(Options, LookupSourcesExpr),
    (option(csr_path(_), Options) -> HasCsr = true ; HasCsr = false),
    (option(lmdb_path(_), Options) -> HasLmdb = true ; HasLmdb = false),
    option(lmdb_materialisation(Materialisation), Options, cached),
    Dict = [
        foreign_preds = ForeignPredsStr,
        lookup_sources_expr = LookupSourcesExpr,
        has_csr = HasCsr,
        has_lmdb = HasLmdb,
        materialisation = Materialisation
    ],
    fsharp_program_template_source(Template),
    render_template(Template, Dict, Code).
```

## 5. Risk

- The template system's match/case has untested edge cases
  (nested matches, special chars in values). Run the testing
  plan in `WAM_TEMPLATE_MATCH_CASE_TESTING.md` first.
- The existing format string works. Migration is a refactor,
  not a feature. Don't break working tests.
- Some F# code in the template contains `'` characters (e.g.,
  `foldl'`) which might interact with template parsing.

## 6. Testing

After migration:
- `swipl -g main tests/core/test_wam_fsharp_csr_smoke.pl` must pass
- `swipl -g main tests/core/test_wam_fsharp_lmdb_smoke.pl` must pass
- Generate a project with various option combinations and verify
  the Program.fs content is identical to the pre-migration output

## 7. References

- Current implementation: `wam_fsharp_target.pl` lines 4843-5049
- Template system: `src/unifyweaver/core/template_system.pl`
- Existing template files: `templates/targets/fsharp_wam/*.mustache`
- Template match/case testing: `WAM_TEMPLATE_MATCH_CASE_TESTING.md`
