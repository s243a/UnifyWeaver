# C++ WAM template refactor plan

Status: in progress. Phase 1 (runtime header) and Phase 2 (lowered
`get_constant`) landed in #4252 and #4253. The first Phase 3 slice, including
project preflight and `main.cpp`, landed in #4255. The program and runtime
extraction landed in #4256, and the first three runtime sections in #4257.
The first Phase 4 reusable lowered family landed in #4261; the current slice
extracts the repeated lowered function shell from `b8a54ee`.
The Pattern Stache literal-substitution fix landed in #4254.
The original planning baseline was `55796489a9750388f4fcf3cac602e4d525b14c28`
(2026-09-11).

## Phase 3 progress (2026-09-12)

The optional `main.cpp` is now a whole-file asset. Its old emitted UTF-8 bytes
are pinned by SHA-256 `0c0f30bcf02ebd2adf39e546e02cbdf53970bcd07f778be2f95c327`
(3,252 bytes, 111 lines). The template has no placeholders; adapter validation
rejects template tags in this asset and in the runtime header.

The `generated_program.cpp.mustache` shell has two ordered, exactly-once
markers: `predicates_code` and `setup_code`. The adapter splices only source
template spans, so a predicate or setup fragment containing literal
`{{setup_code}}` or `{{predicates_code}}` stays unchanged. Pre-extraction
program hashes remain identical for interpreter, functions, and LMDB-enabled
fixtures, including both marker-shaped literals in generated C++ values.

At its initial extraction, `runtime.cpp` was a single 311,009-byte, 7,027-line asset,
with frozen SHA-256 `fa0317a11da5069d0fa18a36bc1937419bcf187e7f6ec4c5d278fd0270cfd300`.
`compile_wam_runtime_to_cpp/2` was option-invariant at extraction: LMDB code
is already guarded by the header's macro, and the same runtime bytes are
emitted with or without LMDB options. The adapter treats no-variable assets
as literal source after rejecting any template tags. This avoids a needless
generic-renderer scan of the 311 KB file. Later Phase 3 slices divided that
asset into ordered sections, as recorded below; no semantic logic moved out
of Prolog.

Project generation now validates the required Pattern Stache cases and renders
all requested artifacts before creating the output directory. A failure in a
required asset or in template-backed predicate rendering leaves a new project
directory absent.
This does not make filesystem writes transactional if a later write itself fails.
Isolated fixture tests cover missing and malformed `.stache` and `.mustache`
files, plus the frozen main bytes. The next slice adds missing/malformed
program-shell and runtime-source fixtures; each leaves the project directory
absent. The 32 focused template checks, six generator checks (including native
compilation and fact/choice/caller queries), and three lowered T6 checks pass.

The existing adapter caches the loaded Stache body by full path and SHA-256,
while checking the file on every render so same-size and same-time edits cannot
reuse stale content. A 10-fact `bench_pair/2` project in functions mode emitted
10 `get_constant` fragments; 20 generations averaged 18.9 ms per project on the
local host. A separate 10,000-render synthetic loop took 4.38 s (0.438 ms per
render). The realistic project invokes this fragment roughly 12 times including
the two preflight case renders, so another parsed-case cache is deferred until
larger real projects show material generation time in this path.

After both new assets were extracted, a 10-fact `bench_pair/2` functions-mode
project with 10 emitted `get_constant` fragments averaged 29.7 ms over 20
generations on the same local host. Passing the validated static runtime
through the generic Mustache scanner instead averaged 41.8 ms on the same
fixture; direct literal return avoids that scan while preserving bytes.

### First bounded runtime split (2026-09-12)

The runtime shell now inserts three contiguous sections at their original
definition positions: `runtime/cell_helpers.cpp.mustache` (323 lines),
`runtime/arithmetic_eval.cpp.mustache` (261 lines), and
`runtime/lmdb_fact_source.cpp.mustache` (356 lines). The remaining shell is
6,087 lines. The adapter requires each of its three markers exactly once and
in order, rejects any other source tags, validates each static section, and
concatenates the original source spans and section text once. It does not
rescan inserted text. Missing or malformed sections fail before project
directory creation.

The assembled runtime keeps the frozen 311,009-byte SHA-256 above for plain and
LMDB-enabled options. Native fact, choice, caller, lowered T6, and LMDB
runtime/codegen cases pass. This split improves file ownership and review size;
it is **relocation and composition, not fragment reuse**. Each section appears
once in one runtime artifact, with no shared body replacing duplicate source.
The same 10-fact functions-mode project averaged 27.4 ms before and 39.7 ms
after the split over separate isolated 20-generation runs on the local host.
The extra file reads, marker validation, and assembly add measurable generation
time; later splits should weigh that cost against concrete reuse or maintenance
benefit. The C++ executable source and behavior remain unchanged.

### Builtin and Step runtime split (2026-09-12)

The next two contiguous sections follow existing headings:
`runtime/builtin_dispatch.cpp.mustache` contains the complete 2,894-line
`WamState::builtin` dispatch, and `runtime/step_execution.cpp.mustache` contains
1,664 lines of Step/execution code up to the term parser heading. The shell is
now 1,529 lines. No definitions moved across their original order or into a
different translation unit. The adapter uses five explicit ordered slots,
requires each marker exactly once, validates static shell spans and child
files, then joins them without rescanning inserted text. It remains a bounded
C++-specific assembler rather than a general template engine.

The frozen runtime SHA-256 above still matches under plain and LMDB options.
Missing and malformed fixtures and duplicate markers for both new sections,
plus an out-of-order shell, all fail before any project directory is created.
The 38 template checks, six focused generator checks, three lowered
T6 checks, and two native LMDB checks pass.

This is still **relocation and single-artifact composition, not reuse**:
Builtin dispatch and Step each appear once in `wam_runtime.cpp`, and no shared
code was deduplicated. A 10-fact `bench_pair/2` functions-mode project averaged
41.3 ms before this split and 34.0 ms after it over separate isolated
20-generation runs on the local host; a further 50-generation post-split run
averaged 34.2 ms. Shrinking the validated shell appears to offset the added
section reads on this fixture, but that timing is not C++ compiler time.
Each additional explicit slot grows the adapter contract, so further splits
should have a concrete maintenance or reuse payoff.

## Phase 4 progress: shared head bind/match body (2026-09-12)

The existing `lowered/head_constant.cpp.stache` atom and value cases now serve
three lowered instruction handlers: `get_constant`, `get_integer`, and `get_nil`.
This is actual source reuse: the bind-or-match C++ bodies previously duplicated
in the integer and nil Prolog emitters now live in one pair of cases.
Prolog still chooses the case, normalizes the register, classifies constant
tokens, escapes atom names, and constructs literal values and exact comments.
In particular, `get_integer` retains its raw numeric token spelling rather
than passing through `cpp_val_literal/2`. The adapter validates the four-key
render contract and both case bodies before project creation, and still checks
the source file hash on every render so edits cannot use a stale parsed body.

Frozen SHA-256/length checks cover the existing seven atom, integer, float,
escaped, and placeholder-shaped `get_constant` fragments, plus three
`get_integer` fragments (including raw `0007`) and two `get_nil` fragments.
All bytes match the previous output. A native C++17 harness generated from
manual WAM instruction lists exercises each case's equal/mismatch paths,
binding, alias visibility, trail unwind, and a later-match failure. The
rollback fixture initializes an unbound A1 cell because the existing runtime
`trail_binding(name)` records only cells already present; this slice does not
change runtime semantics.

A 30-fact functions-mode project with atom, numeric and nil heads contains
30 `get_constant` instructions and zero `get_integer`/`get_nil` instructions:
the current source WAM compiler lowers all those heads through `get_constant`.
Thus the template has three supported lowered call sites, but only one is used
by this ordinary project. Twenty-generation runs after the change averaged
45.9–46.9 ms/project on the local host (an earlier separate pre-change run
was 56.5 ms/project); these separate runs do not establish a speedup. A direct
100-iteration loop rendering all three operations measured 0.36–0.37 ms per
fragment. Each call still hashes the template file to detect mutation, so
expanding this family does add that cost when the integer/nil opcodes arrive
through WAM text or manual lowering. No additional cache layer is justified by
the ordinary project count.

## Phase 4 progress: lowered function shell (2026-09-12)

`lowered/function.cpp.mustache` now owns the common comment/signature/body/
closing-brace shape used by the plain, T4, T5, and T6 lowered paths. Prolog
continues to select the lowering strategy, render instruction bodies, and
construct the mode-specific comment. The adapter validates three ordered,
exactly-once source slots (`comment`, `name`, `body`) and returns the existing
`[Header, Body, Footer]` shape. It inserts already-rendered C++ once, without
rescanning marker-shaped text inside body or comment fragments. Project
preflight validates the shell even for an interpreter-mode project, so missing,
unknown, duplicate, or reordered slots fail before creating output files.
If it changes after preflight, asset errors still propagate through the
per-predicate `warn` policy instead of silently omitting a lowered predicate.

Frozen full-function hashes match the old output for plain, T4, T5, T6, and a
structured ITE predicate. Native C++17 ITE, T4, T5, and T6 execution suites
pass. This is reuse across four emitter paths, replacing their repeated outer
shells; it does not move clause dispatch or ITE semantics into the template.
A 30-fact functions-mode project emits 30 uses of this shell. Three separate
20-generation runs measured 46.25–47.75 ms/project, compared with 45.9–46.9
ms/project in separate runs before this shell extraction. The small difference
is within run variation and is generation time, not C++ compile time.

## Outcome and scope

Move C++ source text out of large Prolog literals and repeated `format/2` calls,
using the existing Mustache subset and Pattern Stache engines. Keep semantic
analysis, lowering eligibility, register normalization, C++ escaping, label
resolution, and artifact assembly order in Prolog. Preserve public predicates,
options, generated bytes, and runtime behavior during the initial migration.

The primary target is `src/unifyweaver/targets/wam_cpp_target.pl` (11,171 lines
at this baseline), together with `wam_cpp_lowered_emitter.pl` (781 lines).
`src/unifyweaver/targets/cpp_target.pl` is a separate 1,778-line streaming/JSON
backend using nlohmann/json; do not accidentally treat its pipeline emitter as
the WAM emitter. Its templates can migrate later using the proven adapter.
No new template language, broad runtime redesign, or cross-target semantic
unification is required.

## Current source map

Paths below are relative to the project root; line numbers describe the baseline.

| Existing location | Responsibility | Refactor boundary |
| --- | --- | --- |
| `src/unifyweaver/targets/wam_cpp_target.pl:157` and `:210` | C++ value literals and instruction factory expressions | Retain classification and escaping; tiny final expressions may be atom templates. |
| `src/unifyweaver/targets/wam_cpp_target.pl:361` | `compile_wam_predicate_to_cpp/4` selects interpreter or lowered emission | Retain selection and supported-instruction policy in Prolog. |
| `src/unifyweaver/targets/wam_cpp_target.pl:432` | `emit_setup_function/3` builds instruction arrays, PCs, registration and configuration | Render an outer shell from precomputed fragments; retain ordering and calculations. |
| `src/unifyweaver/targets/wam_cpp_target.pl:2454` | `write_wam_cpp_project/3` creates runtime header/source, generated program, optional main | Whole-file templates and explicit child-fragment composition. |
| `src/unifyweaver/targets/wam_cpp_target.pl:2511` | `emit_main_shim/1` embeds CLI driver | Whole-file `main.cpp.mustache`. |
| `src/unifyweaver/targets/wam_cpp_target.pl:2983` and `:3004` | Header option wrapper and embedded header body | Extract body first; preserve LMDB prefix behavior. |
| `src/unifyweaver/targets/wam_cpp_target.pl:4143` | `compile_wam_runtime_to_cpp/2` embeds the runtime | Extract intact first, then divide by coherent runtime responsibility. |
| `src/unifyweaver/targets/wam_cpp_lowered_emitter.pl:281` | `lower_predicate_to_cpp/4` and clause-chain/multi-clause emitters | Keep lowerability and control-flow traversal; render bodies from normalized data. |
| `src/unifyweaver/targets/wam_cpp_lowered_emitter.pl:460`–`:495` | Foreign call and structured ITE emission | Preserve semantic dispatch and recursive emission; later compose rendered children. |
| `src/unifyweaver/targets/wam_cpp_lowered_emitter.pl:519` | `emit_one(get_constant(...), Indent)` branches on atom vs numeric value | First useful structural template dispatch slice. |

## Exemplars and what to reuse

Haskell provides the clearest kernel-to-template boundary.
`src/unifyweaver/targets/wam_haskell_target.pl:550`–`:590` contains
`render_kernel_function/2`, `config_ops_to_template_vars/2`, and
`read_kernel_template/2`: derive a data dictionary, select a template, render,
then assemble output. `generate_kernel_haskell/3` also deduplicates kernel kinds
before rendering reusable handler bodies. Its template assets are under
`templates/targets/haskell_wam/`, including `kernel_category_ancestor.hs.mustache`,
`execute_foreign.hs.mustache`, and `main.hs.mustache`.

Reuse that separation and module-relative path resolution. Do **not** copy the
Haskell fallback that substitutes a comment for a missing template or unsupported
kernel: a required C++ artifact must fail generation with a useful error.

Rust illustrates grouping related fragments in one file:
`src/unifyweaver/targets/wam_rust_target.pl:1545` onwards renders
`templates/targets/rust_wam/process_builtin.rs.mustache` twice with `part=helpers`
and `part=arms`. That is an appropriate use of existing string-valued match cases.
F# illustrates whole-file assets and assembly:
`src/unifyweaver/targets/wam_fsharp_target.pl:5241` onwards loads
`templates/targets/fsharp_wam/lmdb_fact_source.fs.mustache`,
`csr_reader.fs.mustache`, and `program.fs.mustache`; its program generator renders
a dictionary later in the file. Neither exemplar implies converting the lowered
emitter's semantic instruction matching into a template algorithm.

## Two engines, three useful template sizes

1. **Small atom templates:** one-line expressions or stable statement snippets
   can stay in a C++-local predicate as atoms and pass through
   `template_system:render_template/3`. Avoid a file per instruction. Leave a
   simple existing `format/3` alone when replacement does not improve composition.
2. **Related cases:** `.mustache` supports `{{match part}}` with exact string
   cases for helpers/arms or named variants. Use `.stache` only when term shape
   and bound fields make the representation simpler, such as
   `head_constant(atom(EscapedName))` versus `head_constant(value(CppLiteral))`.
3. **Larger source units:** store whole runtime files and substantial functions in
   `templates/targets/cpp_wam/`. Split the runtime further only after extraction
   is byte-equivalent; avoid exchanging one giant Prolog literal for hundreds
   of trivial templates.

`src/unifyweaver/core/template_system.pl` exports `render_template/3`, file
loading, named rendering, and `compose_templates/3`. Its match cases are exact
string comparisons, not structural unification. It accepts `Key=Value` lists;
normalize output values to atoms/strings. It does not provide a general partials
system or list iteration, and same-name sections cannot nest. Render repeated
children in Prolog and join them explicitly before filling an outer template.
`compose_templates/3` loads named templates with generated-before-file lookup;
it is not a mixed-engine composer and should not be the C++ adapter's default.

`src/unifyweaver/core/pattern_stache.pl` exports `load_stache_file/2`,
`render_stache/3`, and `render_stache_file/3`. A file must have extension `.stache`
and first nonempty line `{{! dialect(pattern_stache, 1) }}`. Dispatch values must
be ground; case patterns are linear first-order terms; the first matching case
commits. Bound variables shadow outer keys only within that case. Unreachable
cases are load errors; order-dependent overlap warns. `{{q:Key}}` quotes Prolog
terms, **not C++ strings**. Escape C++ values with the existing emitter helpers.
Never route `.mustache` through this engine or modify shared engine semantics.

At the pinned baseline, no other module under `src/` imports or calls Pattern
Stache; its current consumers are prototypes/tests, so C++ would be its first
production target caller. The stronger load-structure validator landed in
PR #4250 after the pinned baseline. Before Phase 2, verify its malformed-block
checks in the branch used for implementation. Keep that engine-hardening change
separate from C++ extraction.

An illustrative structural fragment (not a complete migration) is:

```text
{{! dialect(pattern_stache, 1) }}
{{match op}}{{case head_constant(atom(Name))}}{{indent}}int _m = vm->match_reg_atom("{{reg}}", "{{Name}}");
{{case head_constant(value(Literal))}}{{indent}}Value _a = vm->get_reg("{{reg}}");
{{/match}}
```

The real cases must contain the entire existing atom/value bodies, including
trail updates, unbound handling and failure returns. Prolog still calls
`wam_classify_constant_token/2`, normalizes the register, and escapes the atom.
Template matching selects already-decided source shapes; it must not decide
lowerability, choose runtime semantics, execute guards, or infer instruction
support. Keep unsupported-instruction fallback before rendering. A render error
must propagate instead of silently retrying interpreter emission.

## Proposed adapter and composition contract

Add `src/unifyweaver/targets/wam_cpp_templates.pl` as a small C++-specific adapter,
imported by both WAM C++ modules without introducing their existing potential
circular dependency. Proposed internal API:

- `cpp_render_template(+Id, +Vars, -Text)` for an allowlisted `.mustache` file or
  small local atom template.
- `cpp_render_stache(+Id, +Vars, -Text)` for an allowlisted `.stache` file.
- A registry of IDs, relative asset paths, required keys and dispatch contracts.

These names are proposals, not existing predicates. Resolve paths from the
adapter's callable `source_file/2` anchor to the repository template directory.
Do not depend on caller cwd, mutate global template configuration, or accept an
arbitrary path in an emitter option. Initially load once per generation call;
introduce caching only with evidence, keyed by full path/content identity rather
than a bare template name, and test cold and warm behavior.

Children render first, with only their declared input keys. Join in deterministic
order with explicit separators; inject resulting fragments into an outer shell
once. Preserve indentation, LF policy, trailing newlines, output atom/string
contracts and all existing filename conventions. Avoid repeated rendering of
inserted text: placeholder-like text in a Prolog atom or C++ string literal is
program data. Add a test containing `{{name}}` in an input literal; if naive
substitution would reprocess it, isolate fragment insertion from parameter
substitution at the adapter boundary instead of changing the shared engine.

Both current engines can preserve unknown placeholders. Missing dispatch keys or
unmatched cases may choose a default or produce empty output. Therefore the
adapter must validate required keys/types and supported dispatch shapes before
rendering, require nonempty output for mandatory fragments, and lint template
placeholder references against each case's declared outer/bound keys. Do not
reject arbitrary `{{...}}` sequences in final C++ indiscriminately: they can be
literal user data. Template syntax/contract validation must distinguish those
from template-origin unresolved references. Reject any unresolved template-origin
placeholder before returning a generated artifact.

Missing/unreadable files, invalid dialect/version, nonground data, invalid case
patterns, and required-fragment failures must throw a contextual error containing
template ID/path and emitting predicate or artifact. Treat order-dependent
case overlap as a migration review/test failure even though the engine only
warns. Do not emit a diagnostic comment as success, choose another engine, or
fall back to embedded copies after a selected template fails.

Preflight all required assets and render all required artifact contents before
writing the project files. This prevents a missing late template from leaving a
plausible partial project. Keep filesystem write errors visible; full directory
transactionality is a separate change if needed.

## Migration sequence and acceptance gates

### Phase 0 — freeze behavior and inventory

Capture generated header, runtime, main and program bytes at the pinned baseline,
plus representative lowered fragments. Record options and a manifest/hash for
each fixture. Count embedded source lines and repeated format blocks so review
can distinguish source relocation from real composition improvements. List all
runtime options affecting output; include LMDB header enablement and optional
main. Use isolated output directories for old/new generation.

### Phase 1 — first complete vertical slice: runtime header

Create `templates/targets/cpp_wam/runtime.h.mustache` by extracting the **rendered**
body returned by `compile_wam_runtime_header_body_to_cpp/2`, preserving its exact
bytes. Copying Prolog source spelling directly would retain doubled quotes or
escape sequences incorrectly. Add the adapter and replace only that body's
source with a load/render call. Keep `compile_wam_runtime_header_to_cpp/2` and
its LMDB prefix unchanged for this slice.

Add `tests/test_wam_cpp_templates.pl` with byte-equivalence assertions against
the sealed old header, module-relative loading from a changed cwd, and explicit
missing-file errors using an isolated template fixture/root in tests. Check both
plain header and the LMDB-enabled header without requiring an LMDB runtime merely
to compare text. Generate a minimal full project and compile/run it with a C++17
compiler. Acceptance: public header output identical, executable behavior
identical, and no embedded fallback copy remaining.

### Phase 2 — structural dispatch slice: lowered get_constant

Add `templates/targets/cpp_wam/lowered/head_constant.cpp.stache`. Replace the
large atom/value `format` branches in `emit_one/2` with normalization followed by
structural rendering. Use atom templates for `proceed`/`fail` only if they improve
this local composition; do not make their migration a prerequisite. Preserve
comment spelling and register/value conversions as well as executable code.

Cover atom, integer and float constants, escaped quotes/backslashes, bound and
unbound registers, success, failure, aliasing and trail rollback. Compare bytes
against frozen old fragments, then run generated projects in interpreter and
functions modes. Existing lowerability rules and clause ordering remain intact.
Acceptance: no changed generated bytes, no changed results, and no selection
fallback on a broken template.

### Phase 3 — extract large files, then compose

Extract `runtime.cpp.mustache`, `main.cpp.mustache`, and
`generated_program.cpp.mustache` from the corresponding existing outputs.
The program template receives `predicates_code` and `setup_code`. Keep assembly
and setup calculations in Prolog. Once each extraction passes equivalence,
partition runtime code into coherent modules such as value helpers, parser,
builtins, and interpreter step dispatch. Related helpers/arms can use string
match cases as in Rust; render them as separate fragments into the same outer
runtime. Choose boundaries from actual dependencies and preserve definition order.
This phase provides most of the Prolog line reduction.

### Phase 4 — expand only where composition pays

Migrate repeated lowered instruction bodies, function shells and structured ITE
shells while keeping their semantic traversals in Prolog. Add explicit contracts
for pre-rendered condition/then/else blocks. Keep whole large cases in files when
splitting would obscure control flow. Evaluate `cpp_target.pl` separately after
WAM stability; its includes, JSON/pipeline behavior and tests are different.
Remove migration-only dual-render code after fixtures and execution gates pass.
Each phase is independently reviewable/revertible; commit associated templates,
adapter changes and call-site changes together.

## Validation matrix

| Gate | Required evidence |
| --- | --- |
| Engine compatibility | Existing `tests/test_template_match_case.pl`, `tests/core/test_template_sections.pl`, `tests/core/test_pattern_stache.pl`, and `tests/core/test_pattern_stache_whitespace.pl` still pass through their documented runners. |
| Template contracts | New `tests/test_wam_cpp_templates.pl`: missing file/key, unknown shape, nonground dispatch, malformed/nonlinear/unreachable cases, overlap diagnostics, valid empty optional fragment, required empty rejection, foreign cwd, repeated calls and literal placeholder text. |
| Byte equivalence | Old/new header, runtime, main, generated program and migrated lowered cases match exactly for pinned input/options; no whitespace normalization hiding drift. |
| Generator behavior | `tests/test_wam_cpp_generator.pl`, `tests/test_wam_cpp_lowered_emit.pl`, `tests/test_wam_cpp_lowered_emit_ctrl.pl`, `tests/test_wam_cpp_lowered_head_unify.pl`, `tests/test_wam_cpp_lowered_set.pl`, `tests/test_wam_cpp_lowered_func_name.pl`. |
| Control flow and rollback | `tests/test_wam_cpp_lowered_multi_clause_1.pl`, `tests/test_wam_cpp_lowered_t4.pl`, `tests/test_wam_cpp_lowered_t5.pl`, `tests/test_wam_cpp_lowered_t6.pl`, `tests/test_wam_cpp_lowered_ite_exec.pl`, `tests/test_wam_cpp_frameless_ite_level.pl`, `tests/test_wam_cpp_cut_semantics.pl`, `tests/test_wam_cpp_var_alias.pl`. |
| Native integration | `tests/core/test_cpp_native_lowering.pl`; compile and run generated C++17 projects in interpreter, functions and representative mixed modes, including failure/retry and parser/foreign call paths affected by later extraction. |
| Options | No-main/with-main, LMDB header text plus linked LMDB execution where available, runtime parser modes, foreign predicate registration, lowered clause strategies. Preserve existing option acceptance/rejection. |

The existing `tests/test_template_match_case.pl` is not a green baseline at this
commit: 29 of 31 checks pass. Its eager-LMDB assertions still expect
`DictLookupSource` and `loadDupsortRelationDict`, while the current F# template
uses a demand-pruned dictionary path. Reconcile those two assertions with the
intended F# behavior in a separate change before using this suite as a migration
gate; do not count its current failures as evidence of C++ template drift.

Follow root `AGENTS.md` and `docs/TESTING.md` for setup before executing tests:
SWI-Prolog 9.x, `scripts/setup_local.sh`, and `mkdir -p output/advanced`.
For a focused existing plunit run from the repository root:

```bash
swipl -q -s tests/test_wam_cpp_generator.pl -g run_tests -t halt
swipl -q -s tests/test_wam_cpp_lowered_head_unify.pl -g run_tests -t halt
```

Inspect each suite's runner rather than assuming every file uses plunit. A run
that skips C++ execution because the compiler is absent is not native-equivalence
evidence. Record generated project compiler command, exit status and observable
query results; missing optional toolchains are an explicit incomplete gate.
A documentation-only plan does not establish any of these implementation results.

Track before/after Prolog source-text lines, total asset lines, fragment reuse,
generation time, and generated artifact hashes. Accept line movement as an
extraction milestone, but claim improved composition only when common shells or
cases actually replace duplicated source. Measure rendering overhead before
adding a cache. Runtime performance should remain unchanged when bytes match;
any later output change needs a separate semantic/performance review.
