<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# G-P9 Integration Patch (for INT-0)

**Task:** G-P9 — make the TypeScript pattern target (and, by inheritance,
annotated_js / vanilla_js) a **data-source consumer** for JSON and CSV,
emitting a self-contained Node script (no npm deps: `fs` + `JSON.parse`),
mirroring the PowerShell-pure path exactly (**Option A**). Option B (the
C#-style query-plan/relation model) was explicitly out of scope.
**Worktree:** `agent-ad249e888ea95cd7a`
**Shared-file rule:** this agent did **NOT** edit any `wam_*` file,
`clojure*`, `core/target_registry.pl`, `docs/BINDING_MATRIX.md`,
`tests/test_advanced.pl`, `glue/js_glue.pl`, or `powershell_compiler.pl`.

## Files changed (all inside the allowed set — no central wiring required)

| File | Change |
|------|--------|
| `src/unifyweaver/sources/json_source.pl` | **Additive templates only.** Added `json_file_source_typescript` and `json_stdin_source_typescript` (`template_system:template/2` facts) alongside the existing bash / `_powershell_pure` variants. They use only the variables `generate_json_bash/12` already passes (`pred`, `json_file`, `arity`), so the bash and PowerShell paths are byte-for-byte unchanged. |
| `src/unifyweaver/sources/csv_source.pl` | **Additive templates only.** Added `csv_source_unary_typescript` and `csv_source_binary_plus_typescript`. They use only the variables `generate_csv_bash/11` already passes (`pred`, `file`, `delimiter`, `skip_lines`, `arity`, `columns`). Existing bash / `_powershell_pure` behaviour unchanged. |
| `src/unifyweaver/targets/typescript_source_compiler.pl` | **New wrapper module** modeled on `powershell_compiler:compile_to_pure_powershell/3`. Exports `compile_to_typescript_source/3`; looks up the registered `dynamic_source_def/3`, sets `template_suffix('_typescript')`, and dispatches by source type to `csv_source:compile_source/4` / `json_source:compile_source/4`. |
| `src/unifyweaver/targets/typescript_target.pl` | **One routing clause + two `use_module`.** Added a clause of `compile_predicate_to_typescript/3` **before the fallback** that fires only when `dynamic_source_compiler:is_dynamic_source(Pred/Arity)` holds, delegating to the wrapper. Parallel, independent path — zero changes to `native_ts_clause_body`, the guard/recursion machinery, or any existing clause. |
| `tests/core/test_typescript_source.pl` | **New plunit suite** (`test_typescript_source/0`): JSON + CSV compile-shape checks and, gated on `node_available`, node executions over fixtures asserting the emitted rows. |

**No central wiring is required.** All edited files are in the allowed set, and
`annotated_js` / `vanilla_js` pick the new path up automatically because both
delegate to `typescript_target:compile_predicate_to_typescript/3`.

## How the TS path differs from PowerShell-pure

`compile_to_pure_powershell/3` reads `source_type` + config from its **Options**
(its own top-level entry). The TypeScript path is instead reached from
`typescript_target` via `is_dynamic_source/1`, so `compile_to_typescript_source/3`
recovers the type and config from the registered `dynamic_source_def/3`. Both
then converge on the same plugin `compile_source/4` with a `template_suffix`.

The pure-Node JSON template ignores `jq_filter` (it reads the array with
`JSON.parse`), exactly as the `_powershell_pure` JSON template ignores it. Since
`json_source:validate_config/1` still requires `jq_filter` + an input mode, the
wrapper injects harmless defaults (`jq_filter('.[]')`, and `json_stdin(true)`
when no `json_file` is present) so a Node-targeted JSON source needs no jq
semantics in its declaration.

## Emitted shape (mirrors PowerShell-pure `":"`-joined rows)

- **JSON:** read the file, `JSON.parse`, for each item join `Object.values(item)`
  with `":"` (arity 1 → first value). Cross-checks against
  `jq -r '.[] | [.[]] | map(tostring) | join(":")'`.
- **CSV:** read the file, split on `\r?\n`, skip `{{skip_lines}}` lines, split each
  row on the delimiter; arity 1 → first field, arity 2+ → first `arity` fields
  joined with `":"`; an optional first CLI arg does a first-column key lookup.
  Cross-checks against `awk -F, 'NR>1{print $1":"$2":"$3}'`.

Both run under `node --experimental-strip-types file.ts` (the test harness) and
plain `node file.js` — the emitted code is CommonJS (`require`), which works in
both.

## Follow-ups (not blocking)

- **CSV non-comma delimiters.** The minimal slice embeds the delimiter literally,
  which is correct for the comma default. Tab/pipe delimiters (whose awk-escaped
  form is `\t` / `\|`) need a JS-side unescape before `split` — deferred.
- **Column projection / typing.** `generate_json_bash/generate_csv_bash` do not
  pass `columns` to the JSON template (and pass it only as a comment to CSV), so
  the Node output is positional `":"`-joined values, matching PowerShell-pure.
  Honoring `columns`/`schema` projection would require passing those variables
  through the generators (a change to their existing behaviour) and is left for a
  larger follow-up.
