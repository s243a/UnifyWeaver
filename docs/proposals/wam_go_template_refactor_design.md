# Go WAM template refactor design: atom-embedded Go out of `wam_go_target.pl`

Status: **design only, nothing executed.** Written 2026-09-20 against `main` at
`a102514d0` ("docs: note Go-generator template-hygiene backlog item"). Resolves
`docs/proposals/JS_TARGETS_PARITY_PUNCHLIST.md` §8 (D119 hygiene item). The
worked in-repo precedent is the C++ WAM refactor
(`docs/design/CPP_WAM_TEMPLATE_REFACTOR_PLAN.md`, PRs #4252–#4264); this design
brings the Go target to the same shape and adopts the same acceptance bar:
**generated Go byte-identical before and after every phase.**

How this document was produced: the recon (C++ template system, Go generator,
both template engines, harness) and the analysis were done directly by the lead
designer with repo tools, including one empirical regeneration of the
`pkg_resolver` Go lane into a scratch directory to capture baseline digests.
No nested subagents were available in the session, so no Sonnet/Opus subagent
work is reported here.

Revision 2026-09-20 (same day, after Phase 0/1 landed in `9f583c946` /
`087a738bb`): §4.3, new §4.4, §5, §8 Phase 4 and §11 Q5 were rewritten to
close the Phase-4 `.stache` gap — the first draft left a `.stache`
instruction library as a *conditional* item with no concrete deliverable.
The revision audits every per-instruction body in
`wam_go_lowered_emitter.pl` (unchanged between `a102514d0` and `087a738bb`)
against the C++ precedent's criterion and reaches a decision rather than a
condition. That audit was also done directly with repo tools; no nested
subagents were available.

---

## 1. Problem and goals

`src/unifyweaver/targets/wam_go_target.pl` (5,311 lines) embeds two bodies of
**static** Go source as single-quoted Prolog atoms:

| Body | Where | Size | Interpolation | Consumed by |
|---|---|---|---|---|
| (A) helper blob | `compile_wam_helpers_to_go/2`, lines 3153–5096 | 1,942 lines, 61,313 chars rendered, 90 top-level `func`s plus types/vars | none (`format(atom(GoCode), '...', [])`) | `write_wam_go_project/3` line 119 → `{{helper_methods}}` in `templates/targets/go_wam/runtime.go.mustache` |
| (B) 50 `wam_go_case(GoType, Snippet)` facts | lines 2368–3146 | 782 lines, 27,593 chars once assembled | none (bodies are passed as `~w` *arguments*) | `compile_go_step_case/1` line 2361 → `compile_step_wam_to_go/2` line 2348 → `{{step_method}}` |

Because they are quoted atoms, every apostrophe in Go must be doubled
(`''`: 4 in the blob, 8 inside the snippet bodies plus one in a Prolog
comment at line 2938), `~` is format-significant in the
blob (it is the control string), and `\` sequences are read by the Prolog
reader (5 blob lines carry `\\n`/`\\t`/`\\d`). An apostrophe added to a Go
comment broke Prolog load twice during D117. By contrast
`templates/targets/go_wam/state.go.mustache` (147 KB) holds the real VM and is
edited freely.

Goals, in priority order:

1. **Byte-identical generated Go** after each phase (hard requirement, §7).
2. Go text lives in template files; Prolog keeps only **assembly** (ordering,
   iteration over instruction types, interleaving per-predicate code).
3. **Generality and composability**: smaller templates composed as partials
   (mirroring `templates/targets/cpp_wam/runtime/` and `lowered/`), and template
   files acting as **libraries** through the engines' `{{match}}` features,
   choosing per piece between the literal (`.mustache`) and pattern
   (`.stache`) match (§4–§5). Outcome of that choice: every Go piece is
   literal; §4.4 records why no piece qualifies for pattern match and what
   would change that.
4. Fail-closed loading with contextual errors, preflight before output files
   exist, and a test file that pins bytes (as `tests/test_wam_cpp_templates.pl` does).

Non-goals: no change to the Go VM semantics, no change to `state.go.mustache`,
no new template language, no change to either shared engine.

---

## 2. Current-state inventory (file:line)

### 2.1 Project assembly (`wam_go_target.pl`)

| Lines | What | Notes |
|---|---|---|
| 76–255 | `write_wam_go_project/3` | Creates dir first (line 91), then writes go.mod, value.go, instructions.go, state.go, runtime.go, lib.go, atoms.go, lowered.go, optional main.go. |
| 94–115 | `read_template_file/2` + `render_template/3` (from `core/template_system`) for go.mod/value/instructions/state | Dict keys `package_name`, `date` (no template uses `date` today; `state.go.mustache` has exactly one tag, `{{package_name}}` on line 1). |
| 117–127 | runtime.go: `compile_step_wam_to_go/2` + `compile_wam_helpers_to_go/2` rendered into `runtime.go.mustache` with keys `package_name`, `step_method`, `helper_methods` | `render_template/3` substitutes keys **sequentially over the whole string** (`template_system.pl:280–287`), so inserted Go is rescanned for later keys. Harmless today; the adapter removes the hazard by splicing source spans. |
| 144–148, 162–200, 204–216, 229–249 | Other atom-embedded Go with `~w` interpolation: lib.go header, two atoms.go variants (one ~35-line static Go fallback), lowered.go header, parallel main.go | Out of the §8 scope; listed as Phase 4 candidates (§8). |
| 417–423 | `read_template_file/2` | **Silently substitutes** `// Template not found: <path>` on a missing file. The new adapter must throw instead. |
| 431–435 | `resolve_template_path/2` | Module-relative root via `source_file/2` — same idea as `cpp_template_root/1`. |
| 710–713 | `compile_wam_runtime_to_go/2` | `Step ++ "\n\n" ++ Helpers`; used by `tests/test_go_lmdb_materialisation.pl:118`, `tests/test_wam_go_iso_smoke.pl:221`, `tests/test_wam_go_foreign_lowering.pl:141` (content greps). Keep exported. |
| 716–721 | `write_file/2` | `format(Stream, "~w", [Content])`, locale-default encoding (harness runs `LANG=C.utf8`). |

### 2.2 `templates/targets/go_wam/runtime.go.mustache` (29 lines)

```
package {{package_name}}
import ( fmt math os os/exec runtime sort strconv strings sync )
var ( _ = fmt.Sprintf ... _ sync.Locker = (*sync.Mutex)(nil) )

{{step_method}}

{{helper_methods}}
```
(one trailing LF after `{{helper_methods}}`). The rendered `Step` text has **no**
trailing LF; the helper blob **ends with one LF**; the file therefore ends
`}\n\n`. These two facts drive the whitespace rules in §7.3.

### 2.3 Body (A): the helper blob, lines 3155–5096

Top-level declarations in file order (blob-relative line = file line − 3154):

| Group (proposed partial, §5) | Declarations | File lines (approx.) |
|---|---|---|
| run loop | `Run`, `indexedClauseBodyStart`, `isChoiceChainHead`, `enterIndexedAlternatives`, `enterIndexedClause`, `runIsolatedGoal`, `CollectResults`, `fetch`, `resolveInstructions` | 3155–3395 |
| aggregates | `executeAggregate`, `findMatchingAggregateEnd`, `runUntilPC`, `backtrackAbove`, `bindOrUnifyReg`, `aggregateResultValue`, `uniqueAggregateValues`, `aggregateNumericValue` | 3396–3595 |
| foreign registry | `registerForeignNativeKind` … `foreignUsizeConfig`, `parseForeignTupleLayout` | 3596–3710 |
| atom-fact2 sources | `staticAtomFact2Source`, `lmdbAtomFact2Source` (+ eager/lazy/cached comment block), `parseAtomFact2Rows` | 3711–3920 |
| foreign results | `applyForeignResult`, `finishForeignResults`, `finishStreamResults`, `executeIndexedAtomFact2` | 3921–4076 |
| native kernels | `valueAsAtomString` … `tupleValue`, adjacency builders, `collectNative*` (closure, distances, weighted shortest path, A\*, category-ancestor hops), `pickShortestCandidate`, `heuristicLookup`, `listAsAtomStrings` | 4077–4543 |
| execute foreign | `executeForeignPredicate` (the native-kind dispatcher) | 4544–4692 |
| seek fact source | `factIO*` counters/getters, `seekFactSource` + `leU16/leU32/bytesCompareGo/factIORead`, text classifiers (`isIntText`, `isFloatText`), `parseFactSourceValue`, `encodeStoreKey`, open/lookup/read/scan/rows, `registerIndexedSeekFact2`, `registerLmdbSeekFact2`, `lmdbSeekMissingError`, `executeCallFactStream` | 4693–5096 (two `// ----` rules at 4679 and 4691 mark this boundary) |

Hazard census of the blob text (Prolog spelling): `{{` ×4 (lines 4171, 4204,
4258 `queue := []qd{{source, 0}}`, 4266 `cands = [][2]string{{next, source}}`);
`}}` ×8 (the four above plus `[]Value{&Float{Val: 0}}`-style literals at
4417, 4448, 4557, 4577); `~` ×0; `''` ×4 (3284, 3289, 3291, 4469); `\\`
escape lines ×5; 18 lines with non-ASCII (em dashes, `×`) — the file is
`:- encoding(utf8)`.

Rendered baseline (from `compile_wam_helpers_to_go([], H)` at `a102514d0`):
**61,313 chars, SHA-256 `dab3b8c24103f54c863b0c89ecc4b1b7cb27cd8ebbb9c29b83dbe9000cd06552`, ends with LF.**

### 2.4 Body (B): the 50 `wam_go_case/2` snippets, lines 2366–3147

Family headings already present in the Prolog source, in **findall order**
(= clause order; this order is the byte-identity contract for the switch):

| Family (source heading) | Lines | Cases (in order) |
|---|---|---|
| Head Unification | 2366–2571 | GetConstant, GetVariable, GetValue, GetStructure, GetList, UnifyVariable, UnifyValue, UnifyConstant |
| Body Construction | 2572–2716 | PutConstant, PutVariable, PutValue, PutStructure, PutList, SetVariable, SetValue, SetConstant |
| Control | 2717–2952 | Allocate, Deallocate, Call, GetArgInto, CallForeign, CallIndexedAtomFact2, CallFactStream, CallPc, Execute, ExecutePc, Jump, JumpPc, CutIte, GetLevel, Cut, BeginAggregate, EndAggregate, Proceed, BuiltinCall, BuiltinExecute |
| Choice Point | 2953–3011 | TryMeElse, TryMeElsePc, RetryMeElse, RetryMeElsePc, TrustMe, Try, Retry, Trust |
| Indexing | 3012–3146 | SwitchOnConstant, SwitchOnConstantPc, SwitchOnStructure, SwitchOnStructurePc, SwitchOnConstantA2, SwitchOnConstantA2Pc |

(8 + 8 + 20 + 8 + 6 = 50.) Each body starts with eight spaces and ends with
its last statement, **no trailing LF**. Assembly (`2348–2364`):

```
'// Step executes a single WAM instruction.\nfunc (vm *WamState) Step(instr Instruction) bool {\n    switch i := instr.(type) {\n' ++
join("\n", [ "    case *" ++ Type ++ ":\n" ++ Body | wam_go_case(Type, Body) ]) ++
'\n    default:\n        return false\n    }\n}'
```

Hazard census: `{{` ×0; `}}` ×2 (2473, 2478 — `[]Value{h, t}}`); `~` ×3
(2732–2733, inside a comment — safe only because bodies are `~w` arguments, not
control strings); `''` ×9; 7 non-ASCII lines.

Rendered baseline: **27,593 chars, SHA-256
`433eea3c43c09a55633885b9978c534a9a0b1b8930a33f4edcf3ba79a031416e`, no trailing LF.**

### 2.5 Full-project baseline (pkg_resolver `go` lane, `package wam`)

Regenerating with `examples/pkg_resolver/go/build.pl` at `a102514d0` into a
scratch directory gives deterministic output (all files depend only on the
predicate set and `package_name`; `date` is passed but unused):

| File | Bytes | SHA-256 |
|---|---|---|
| runtime.go | 89,244 | `6b0e707ca7a76f6daad8ce28b96e72a25193344e14eceae3c9c1e25980fdabbd` |
| state.go | — | `f5a9b98f07415b7e6c1a0bd5434fad1aa8336cae180a5734bb364d7e610a97f0` |
| value.go | — | `dcad39587c21e561344c83e28e59265135e8105fa7d8e4d249bba933ddbf0c27` |
| instructions.go | — | `377b318e33337177000ab6810721ccf262aef07d812f7a5dc82cd9cae09ece7a` |
| lib.go | — | `8c525934b70227b7807eb91eea15834cc87395dc9ead564dbc213e0338e315de` |
| lowered.go | — | `8811b3c216fca26285c1c48af8189662f80551d5f99b4fbd2c36bd709f23a099` |
| atoms.go | — | `b3997d6ef37692335a4a23509a9b99ab64bc2d0e1af72ce67836f17cda18c347` |
| go.mod | — | `c7731d5b4dec2b8e16a23d15dfe1d3b6de4e944eb9833f405a5d26d65438ab47` |

**Caution for the gold diff:** the checked-in `examples/pkg_resolver/go/*.go`
at `a102514d0` are *stale* relative to the generator (runtime.go there is
86,579 bytes, SHA `5d53a31e…`, last regenerated in `4fe2b178e`, 2026-09-06);
the `go_store` lane's checked-in `runtime.go` is current (`6b0e70…`, identical
to a fresh `go` lane build). The gold must therefore be **generator-vs-generator**
(base commit build vs refactor build), never "diff against the checked-in file"
unless that lane is regenerated at the base commit first (§9).

### 2.6 The lowered emitter (`wam_go_lowered_emitter.pl`, 824 lines)

Not part of §8, but in the "same shape" remit. It contains no large static
Go: five `format(atom/string …)` shells (function header/footer at 348–352,
363–372, 407–410) and line-by-line `format/2` emission per instruction
(`emit_one/2` from 540; ITE block 506–527; T5/T6 dispatch 390–458). The
`get_constant` / `get_integer` / `get_nil` bodies (548–590) are the same
twelve lines differing only in the Go value expression and comment — the
Go analogue of the C++ head-match family, but *without* the atom-vs-value
runtime split C++ needed. That missing split is the whole `.stache`
question for Go; §4.4 audits it. See also §5.4 and §8 Phase 4.

---

## 3. The C++ reference pattern, distilled

### 3.1 Files

```
templates/targets/cpp_wam/
  runtime.h.mustache            whole-file, no tags (100 KB)
  runtime.cpp.mustache          SHELL with 5 ordered slots: {{cell_helpers}} {{arithmetic_eval}}
                                {{builtin_dispatch}} {{step_execution}} {{lmdb_fact_source}}
  runtime/*.cpp.mustache        5 static SECTIONS spliced into the shell (12–135 KB each)
  main.cpp.mustache             whole-file, no tags
  generated_program.cpp.mustache SHELL: {{predicates_code}} {{setup_code}} (exactly once, ordered)
  lowered/function.cpp.mustache  SHELL: {{comment}} {{name}} {{body}}
  lowered/ite.cpp.mustache       SHELL: 17 ordered slots (repeated {{indent}}/{{counter}})
  lowered/head_constant.cpp.stache  PATTERN library: {{match op}}{{case head_constant(atom(Esc))}} … {{case head_constant(value(CppVal))}}
src/unifyweaver/targets/wam_cpp_templates.pl   (405 lines) the adapter
tests/test_wam_cpp_templates.pl                  (667 lines) bytes + fault fixtures
```

### 3.2 Adapter principles (all carried over to Go)

1. **Registry of IDs → relative paths** (`cpp_template_path/2`,
   `cpp_stache_path/2`); callers never pass paths. Root is module-relative via
   `source_file/2`, with a thread-local test override
   (`cpp_with_template_root_for_test/2`). No project option can redirect it.
2. **Shells are spliced, never re-rendered.** The adapter splits the shell on
   each declared marker (`split_single_source_marker/4`: exactly once, in
   order), lints the static spans (`source_span_without_tags/1`), then
   concatenates spans and already-rendered children once. Inserted text is
   **never rescanned** — a predicate fragment containing literal
   `{{setup_code}}` stays literal (`test program_shell_does_not_rescan_fragments`).
3. **Whole-file / section assets with no variables are returned literally**
   after a tag lint (`render_cpp_template/5` last clause) — no generic-engine
   scan of a 300 KB file per build (measured 41.8 ms vs 29.7 ms).
4. **Fail closed** with contextual errors: `cpp_wam_template_load/3`,
   `_empty/2`, `_tags/2`, `domain_error(cpp_wam_template_variables, …)`,
   `cpp_wam_stache_failure/3`. Never emit a diagnostic comment as success.
5. **Preflight** (`cpp_preflight_*`) renders every required asset **before**
   the output directory is created (`wam_cpp_target.pl:2585–2592`); a broken
   template leaves no partial project.
6. **`.stache` cache keyed by full path + SHA-256**, rechecked on every render
   so a same-size same-mtime edit can never serve stale parse
   (`load_head_constant_stache/2`).
7. **Bytes pinned in tests**: length + SHA-256 of header, runtime, main,
   program, and each lowered fragment; foreign-cwd test; repeated-render test;
   fault fixtures for missing/malformed/duplicate/reordered markers.
8. **Extraction by execution**: the C++ plan warns "copying Prolog source
   spelling directly would retain doubled quotes or escape sequences
   incorrectly" — the template file is written from the *rendered* output.

### 3.3 `.mustache` vs `.stache` — what the extension selects

| | `.mustache` — `core/template_system.pl` (`render_template/3`) | `.stache` — `core/pattern_stache.pl` (`load_stache_file/2`, `render_stache/3`) |
|---|---|---|
| Loader | any text file; no header | extension **must** be `.stache`; first non-empty line **must** be `{{! dialect(pattern_stache, 1) }}`; load-time structural + case checks |
| `{{case v}}` value | literal text, compared as a string (`case_matches/2` after `atom_string` on both sides) | a **Prolog term** read with `read_term_from_atom/3` (module `pattern_stache`, standard op table); matched by **unification**, first match commits |
| Bindings from a match | none | pattern variables (`variable_names/1`) are **prepended to the dict** as a child scope for that case body; `{{Var}}` (`~w`) and `{{q:Var}}` (`~q`) |
| Dict values | atoms/strings (a compound → `type_error`) | **ground** terms (nonground → `nonground_dispatch` error) |
| Case overlap | impossible | checked at load: earlier subsumes later → **error** `unreachable_case`; refinement (specific before general) silent; unifiable-neither-subsumes → warning |
| Other constructs | `{{#flag}}…{{/flag}}`, `{{^flag}}…{{/flag}}` sections; `{{default}}`; nested match by depth counting | match/case/default only; nested match tolerated but unpromised |
| Substitution order | match blocks → sections → **sequential key replacement over the whole (already substituted) string** | each source segment substituted **once in its own scope**; a rendered value containing `{{K}}` is never rescanned (PR #4254) |
| Unknown `{{key}}` | left verbatim | left verbatim |
| Text between `{{match k}}` and first `{{case}}` | discarded | discarded (the only byte the engine drops) |
| Whitespace | none trimmed | none trimmed; caller owns policy; no "standalone line" semantics |

Normative text: `docs/design/SPEC_pattern_stache.md`; rationale:
`docs/design/STRUCTURAL_TEMPLATE_MATCHING_PHILOSOPHY.md`.

How C++ chose: every shell and every static section is `.mustache` (string
slots; no dispatch). The one `.stache` file exists because *two differently
shaped* C++ bodies must be selected by the **shape** of a normalized term
(`head_constant(atom(Esc))` vs `head_constant(value(CppVal))`) and the
selected body needs the **field bound by the match** (`Esc` / `CppVal`).

---

## 4. Literal match vs pattern match — decision rule for Go

### 4.1 What each match can express

- **`.mustache` `{{match key}}{{case Name}}`** is a keyed lookup of a text body
  by an exact string. It is the right tool when the caller already holds a
  *token* (an identifier) and needs the text filed under it, with no data
  flowing from the key into the body. That is precisely a **library of named
  fragments**: `{{match instr}}{{case GetConstant}}…{{case GetVariable}}…`.
  Rust already uses this shape (`process_builtin.rs.mustache`, rendered twice
  with `part=helpers` / `part=arms`).
- **`.stache` `{{match key}}{{case f(X, g(Y))}}`** selects by the *shape* of a
  ground term and hands the bound sub-terms to the body. It is the right tool
  when (a) the discriminator is structural (functor/arity/nested constant) and
  (b) the body needs fields carried by the term. If neither holds, the pattern
  engine adds a header, load-time overlap checks, and term-reading rules
  (`{{case wam-fsharp}}` reads as `-(wam, fsharp)`; an uppercase-initial
  case is a variable that matches everything) for no benefit.

### 4.2 Hazards specific to moving Go text into either engine

| Hazard | `.mustache` | `.stache` | Consequence for this design |
|---|---|---|---|
| Go composite literals `[]qd{{source, 0}}` (4 in the blob) | `expand_sections` only reacts to `{{#`/`{{^` + `[A-Za-z0-9_]+` + `}}` (`valid_tag_name/1`); `render_template_string` only replaces declared keys; `{{source, 0}}` survives verbatim | scanner reads `source, 0` as a tag; `tag_structure/3` ignores it; renderer leaves an unknown placeholder verbatim | Safe in both engines **today**, by luck of content. The adapter must not rely on it: static Go is spliced as source spans and never handed to either engine (§6.3), and the C++ lint "no `{{`/`}}` at all" **cannot** be copied — it would reject valid Go. Use the Go-aware lint of §6.4. |
| Stray `}}` (`[]Value{&Float{Val: 0}}`, 8 in the blob, 2 in cases) | ignored unless preceded by a recognised opener | same | no action; lint ignores bare `}}` |
| `{{case `, `{{default}}`, `{{match `, `{{/match}}`, `{{#x}}` text inside a Go body | would be parsed as structure | same (and a load error if misplaced) | impossible in valid Go except inside string literals/comments; lint rejects, error names the line |
| Whitespace around case tags | body = literal text between markers, including the LF before the next `{{case` | same | case library needs an explicit, adapter-owned standalone-line rule (§7.3) |
| Text before the first `{{case` | discarded | discarded | header comments for a case library go there (or as Go `//` comments inside bodies) |
| `{{! comment }}` | **not** a comment in `template_system` — emitted verbatim | header pragma only | do not use `{{!` in `.mustache` Go templates |
| Performance | match parse is a string scan per render; C++ measured ~0.4 ms per small render | load hashed + cached per path; render cheap | 50 renders of ~6–8 KB family files per project is negligible; measure once (C++ plan style) before adding any cache |

### 4.3 Per-piece choice (the user's question, answered)

| Piece | Dispatch needed? | Data flows key → body? | Engine | Match style |
|---|---|---|---|---|
| (A) helper blob (§2.3) | none — one fixed sequence of Go declarations | no | `.mustache` **static sections** spliced into the runtime shell | none (ordered slots in the shell; §5.1) |
| (B) 50 step-case snippets (§2.4) | yes — by Go instruction **type name** (`GetConstant`, …), a plain identifier the Prolog assembler already holds | no (bodies have zero interpolation; `i.Ai`, `i.C` are Go field accesses, not template variables) | `.mustache` **library** files, one `{{match instr}}` block per family | **literal**: `{{case GetConstant}}` |
| runtime.go / Step shells | slots only | rendered children | `.mustache` shells | none (exactly-once ordered markers) |
| lowered function / ITE shells (Phase 4) | slots only | name, comment, body, indent | `.mustache` shells | none |
| lowered head-match family `get_constant`/`get_integer`/`get_nil` (Phase 4) | **no** — one body; the three Prolog clauses differ only in how they compute the value expression, and all three land on the same twelve Go lines | yes, but as four pre-rendered strings (`I`, `Comment`, `Ai`, `GoVal`); nothing in the body varies by the *shape* of the value | `.mustache` fragment, **spliced by the adapter on ordered repeated markers** (the `ite.cpp.mustache` mechanism), never through `render_template/3` — a `Comment` for the atom `'{{Ai}}'` would otherwise be rescanned (the C++ tests pin exactly this case) | none (repeated ordered slots: `{{I}}`×12, `{{Comment}}`, `{{Ai}}`, `{{GoVal}}`×2) |
| lowered per-instruction library (Phase 4) | the discriminator is the instruction **functor**, which `emit_one/2`'s clause heads already dispatch on; no instruction's Go *statements* vary with the shape of a bound sub-term (audit in §4.4) | after normalisation (register → index, token → Go literal, functor → escaped, atom → interned var) the fields are already strings | **none — dropped.** A `.stache` here would be chosen for its binding syntax, not for structural dispatch: the "for no benefit" case of §4.1. The bodies stay as `format/2` emission patterns, exactly as the C++ closeout audit decided for the same families | — |

Why not `.stache` for (B): the snippets need no bindings, the discriminator is
a token, and pattern reading would turn `{{case GetConstant}}` into a
**variable** pattern (uppercase initial) that matches everything — the exact
trap the SPEC's migration checklist warns about; every case would need quoting
(`{{case 'GetConstant'}}`) for zero gain. Literal match is the honest fit.

Why not a match library for (A): there is nothing to select; the blob is a
fixed sequence. Splitting it into ordered static sections (composability by
file ownership and review size) is the C++ "relocation and composition"
pattern; a match block would add a dispatch that is always the same.

### 4.4 Audit: is there a `.stache` home in the Go lowered emitter? (decision: no)

The first draft of this document kept a `lowered/instruction.go.stache`
"only if the emitter is restructured to dispatch on normalized instruction
terms". That is a condition, not a design, and it left Phase 4 with no
concrete `.stache` deliverable. This subsection replaces the condition with a
decision, reached by applying the precedent's own criterion to every
per-instruction body the Go emitter has.

**The criterion (from the precedent).** The one `.stache` in the repo,
`templates/targets/cpp_wam/lowered/head_constant.cpp.stache`, earns the
pattern engine because a single WAM instruction has **two differently shaped
statement bodies** — `vm->match_reg_atom(...)` with its `_m < 0 / == 0`
ladder for atoms, `vm->get_reg(...)`/`is_unbound()`/`operator==` for
integers and floats — and the choice between them is made by the **shape of
a normalised term** (`head_constant(atom(Esc))` vs `head_constant(value(CppVal))`)
whose bound field the chosen body needs (`wam_cpp_lowered_emitter.pl:506–527`,
`739–742`). Note what does *not* dispatch there: `get_integer` and `get_nil`
hard-pick a case in Prolog; only `get_constant` dispatches, on
`wam_classify_constant_token/2`'s class. The C++ plan's rule is the same test
in one sentence: "Use `.stache` only when term shape and bound fields make
the representation simpler" (`CPP_WAM_TEMPLATE_REFACTOR_PLAN.md`, "Two
engines, three useful template sizes", item 2), and its Phase-4 closeout
audit applied it negatively to the `get_*`/`put_*`/`unify_*`/`set_*` groups:
"repeat short comments and one-line `vm->step(...)` calls, but their argument
normalization and runtime operations differ … They stay in Prolog as small
emission patterns."

**Census of `emit_one/2` at `wam_go_lowered_emitter.pl:540–755`** (26 emitting
clauses, 4 silent consumers `try_me_else`/`trust_me`/`cut_ite`/`jump`, one
`// TODO` fallback; 116 `format/2` calls in the region). Every emitting clause
has exactly **one** Go body; the shapes are:

| Body shape | Clauses | Lines emitted | What varies between clauses | Varies by *shape* of a bound sub-term? |
|---|---|---|---|---|
| head bind-or-match block | `get_constant` 548, `get_integer` 563, `get_nil` 577 | 12 (identical text) | the comment and one Go expression `GoVal` (`go_val_literal/2` result, `&Integer{Val: N}` with the raw token, or the interned `[]` variable) | **no** — `valueEquals(vm.deref(_a), GoVal)` is the same statement for atom, integer and float; atoms are package-level `*Atom` pointers, so Go has no allocation to avoid and no fast path to select |
| `if !EXPR { return false }` guard | `get_value`, `get_structure`, `get_list`, `put_structure`, `put_list`, `unify_variable`, `unify_value`, `unify_constant`, `set_variable`, `set_value`, `set_constant`, `builtin_call`, `call_foreign` (13) | 3 + comment | one Go expression: `vm.Unify(a, b)`, `vm.Step(&GoType{Fields})`, `vm.executeBuiltin("op", n)`, `vm.executeForeignPredicate("p", n)` | **no** — the ten `vm.Step` delegates differ only in the struct literal (`Functor+Ai`, `Ai`, `Xn`, `C`); that is an *expression*, not a statement shape. A `.stache` with four field-shape cases would trade 10×4 `format/2` lines for 4 cases plus 10 normalisation clauses: same contract count, one more file, and the exact family the C++ closeout audit kept in Prolog |
| one-liner | `proceed`, `fail`, `get_variable`, `put_constant`, `put_value`, `allocate`, `execute` (7) | 1 + comment | register indices / one expression | **no** (the C++ plan's "leave a simple existing `format/3` alone" size class) |
| distinct blocks | `put_variable` (4 lines), `deallocate` (3), `call` (7) | fixed | register indices / call expression | **no** — one body each, nothing to select |

Four places *look* like structural dispatch and were checked individually:

1. **`go_val_literal/2` (792–805)** dispatches on `integer(N)` / `float(F)` /
   `atom(Name)` to three different Go expressions — the only real
   shape-dispatch in the file. It is not a template consumer for two
   independent reasons: the bodies are one-line expressions (size class 1 of
   the C++ rule), and the atom arm calls `intern_atom_go/2`, which is
   **side-effecting** (`retract`/`assertz` of `go_atom_intern_next/1` and
   `go_atom_intern_id/2` at `wam_go_target.pl:3214–3224`; the table it builds
   is what `atoms.go` is generated from). A template cannot perform that
   call, so the Go value must be computed *before* rendering — at which point
   the shape is gone and only a string remains.
2. **T5 if-cascade vs T6 string switch (390–450).** Two genuinely different
   bodies, but the selection is per predicate on a *count threshold* plus an
   all-atoms check (`go_t6_applicable/2`), not on any term's shape, and both
   bodies iterate over the guard list — list iteration is a v1 `.stache`
   exclusion ("the caller iterates; the template dispatches one term").
3. **A whole-instruction library** (`{{case get_variable(Xn, Ai)}}`, the
   first draft's candidate). Its "bound fields" are already-rendered strings
   after normalisation, and its discriminator is the functor — which is a
   token. The pattern engine would be selected for the convenience of naming
   positional arguments in the case head; the same convenience is a `format/2`
   argument list. §4.1 already names this as the "no benefit" case, and the
   restructuring it requires (`emit_one/2` printing via `format('~s')` of a
   rendered fragment for 26 clauses) is the cost the first draft's condition
   was hedging against. The audit finds nothing on the other side of that
   cost.
4. **The head-match family as a one-case `.stache`.** Tempting only because
   the pattern engine substitutes each scope once and never rescans a
   rendered value (SPEC, "Interpolation"), which the head-match fragment
   needs (`'{{Ai}}'`-shaped atoms). But the adapter's ordered-repeated-marker
   splice gives the same guarantee without a header, a term reader and
   overlap checks; a one-case match is a degenerate use of a dispatch
   dialect. It stays `.mustache` + splice (§4.3 row).

**Decision.** No Go instruction has a second body shape selected by a bound
sub-term; therefore no `.stache` file is added to `templates/targets/go_wam/`
in this refactor, and the conditional `lowered/instruction.go.stache` item
is **removed** from §5 and §8 rather than left open. Every Go template is
literal: shells, sections and literal-match libraries.

**Reopen condition** (recorded the way the SPEC records exclusions — a
condition to check, not a date). The item reopens when the Go lowered
emitter gains, for one instruction, **two Go statement bodies selected by
the shape of a bound sub-term**. The realistic trigger is a
`get_constant` atom fast path — e.g. comparing the dereferenced register
against the interned `*Atom` pointer before falling back to `valueEquals`,
the Go analogue of C++'s `match_reg_atom`. That is a **semantic change to
generated code** and belongs to a performance PR, not to this byte-identical
refactor. When it lands, the deliverable is the direct mirror of the
precedent and needs no further design:
`lowered/head_match.go.stache` with `{{match op}}{{case head_constant(atom(AtomVar))}}
…{{case head_constant(value(GoVal))}}…{{/match}}` and outer keys
`I`/`Comment`/`Ai`; `emit_one(get_constant(..))` classifies the token and
builds the `op` term exactly as `wam_cpp_lowered_emitter.pl:506–516` does,
`get_integer` passes `value(...)` with the raw token, `get_nil` passes
`atom(<interned [] var>)`; the adapter gets `go_render_stache/3` with a
path+SHA-256 keyed cache and a four-key contract check
(`wam_cpp_templates.pl:178–277`); the gate is the frozen per-fragment
digest table of Phase 4 slice 0 (§8) re-baselined *in the performance PR
that introduces the second body*, with the template-hygiene change landing
as a separate byte-identical PR after it. Until that trigger fires, adding
the file would be the over-engineering §4.1 warns against.

---

## 5. Target layout for `templates/targets/go_wam/`

```
templates/targets/go_wam/
  go.mod.mustache                     (unchanged)
  value.go.mustache                   (unchanged)
  instructions.go.mustache            (unchanged)
  state.go.mustache                   (unchanged, 147 KB)
  main_bench.go.mustache              (unchanged)
  runtime.go.mustache                 SHELL — package/imports + ordered slots (§5.1)
  runtime/
    run_loop.go.mustache              static section  (Run … resolveInstructions)
    aggregate.go.mustache             static section
    foreign_registry.go.mustache      static section
    atom_fact2_sources.go.mustache    static section
    foreign_results.go.mustache       static section
    native_kernels.go.mustache        static section
    execute_foreign.go.mustache       static section  (executeForeignPredicate)
    seek_fact_source.go.mustache      static section  (factIO … executeCallFactStream)
  step/
    step_shell.go.mustache            SHELL — Step() header/footer with one slot {{cases}}
    head_unification.go.mustache      LIBRARY — {{match instr}} 8 literal cases
    body_construction.go.mustache     LIBRARY — 8 cases
    control.go.mustache               LIBRARY — 20 cases
    choice_point.go.mustache          LIBRARY — 8 cases
    indexing.go.mustache              LIBRARY — 6 cases
  lowered/                            Phase 4
    function.go.mustache              SHELL — {{comment}} {{name}} {{body}}
    ite.go.mustache                   SHELL — ordered {{indent}}/{{condition}}/{{then}}/{{else}} slots
    head_match.go.mustache            fragment — ordered repeated slots {{I}}×12 {{Comment}} {{Ai}} {{GoVal}}×2
    (no .stache file: §4.4 — reopens only when a Go instruction gains a second
     body shape selected by a bound sub-term; then lowered/head_match.go.stache)
  project/                            Phase 4 (optional)
    atoms_runtime_only.go.mustache    {{package_name}} (the 35-line static fallback at 175–198)
    lib_header.go.mustache, lowered_header.go.mustache, main_parallel.go.mustache
```

### 5.1 `runtime.go.mustache` after Phase 2

```
package {{package_name}}

import ( … unchanged … )

var ( … unchanged … )

{{step_method}}

{{run_loop}}

{{aggregate}}

{{foreign_registry}}

{{atom_fact2_sources}}

{{foreign_results}}

{{native_kernels}}

{{execute_foreign}}

{{seek_fact_source}}

```
Every marker stands alone on its line; the shell owns all inter-section blank
lines; fragments are stored as ordinary text files ending in one LF, and the
adapter strips exactly that LF (§7.3). Phase 1 uses the same shell with a
single `{{helper_methods}}` slot holding the intact blob.

### 5.2 A step library file (`step/head_unification.go.mustache`)

```
{{match instr}}
Go bodies for the head-unification instructions of Step(). One case per
Go instruction type; the Prolog assembler (go_step_case_order/1 in
wam_go_templates.pl) owns the switch ORDER and the `case *Type:` line.
Text in this preamble is discarded by the engine.
{{case GetConstant}}
        val := vm.Regs[i.Ai]
        if val == nil { return false }
        // Dereference before inspecting: …
        …
        return false
{{case GetVariable}}
        val := vm.Regs[i.Ai]
        …
{{/match}}
```
Bodies keep their eight-space indentation exactly as today (so the generated
switch is unchanged). Each case body is the literal text between its tag line
and the next tag line; the adapter removes the one leading LF (after the tag)
and the one trailing LF (before the next tag) — §7.3.

### 5.3 `step/step_shell.go.mustache`

```
// Step executes a single WAM instruction.
func (vm *WamState) Step(instr Instruction) bool {
    switch i := instr.(type) {
{{cases}}
    default:
        return false
    }
}
```
(file ends with one LF, stripped by the adapter so `compile_step_wam_to_go/2`
still returns text without a trailing LF).

### 5.4 Why five family files and not one or fifty

One 800-line library is a single blob again; fifty one-case files are the
"file per instruction" the C++ plan rejected. Five files follow the headings
already in the source, keep each file at 60–240 lines, and let a reviewer of,
say, choice-point semantics open one file. The Prolog order list is the single
source of truth for dispatch order; the adapter verifies that the union of case
names across the family files equals that list exactly (no missing, duplicate,
or unknown case), so a body can be moved between family files without changing
output.

---

## 6. Prolog changes

### 6.1 New module `src/unifyweaver/targets/wam_go_templates.pl` (the adapter)

Modelled on `wam_cpp_templates.pl`; proposed API (names are proposals):

```prolog
:- module(wam_go_templates, [
    go_render_template/3,          % +Id, +Vars, -Text        (shells and static sections)
    go_render_template_at_root/4,  % +Root, +Id, +Vars, -Text (fixture root for tests)
    go_step_case_order/1,          % -[GoTypeName,...]         the 50 names, in today's findall order
    go_step_case_text/2,           % +GoTypeName, -Body        one library case, standalone-trimmed
    go_step_case_text_at_root/3,
    go_render_step_method/1,       % -StepText                 assembled Step() (used by compile_step_wam_to_go/2)
    go_render_helper_methods/1,    % -HelpersText              Phase 1: intact blob; Phase 2+: ordered sections joined by the shell
    go_preflight/0,                % loads + validates every required asset, renders every case once
    go_with_template_root_for_test/2
]).

go_template_path(runtime_shell,            'runtime.go.mustache').
go_template_path(step_shell,               'step/step_shell.go.mustache').
go_template_path(runtime_run_loop,         'runtime/run_loop.go.mustache').
… (one fact per static section) …
go_step_library(head_unification,          'step/head_unification.go.mustache').
… (five facts) …
```

Registry contract per ID: kind (`shell(Markers)` with the exact ordered marker
list, `section` = no variables, `library` = one `{{match instr}}` block), and the
declared variables. `go_render_template/3` rejects undeclared variables
(`domain_error(go_wam_template_variables, …)`), missing files
(`go_wam_template_load/3`), empty files (`go_wam_template_empty/2`), and
marker faults (`go_wam_template_tags/2`), mirroring the C++ error family so
`handle_compile_error`-style policies can be reused.

Root resolution: `source_file(go_render_template(_,_,_), Src)` →
`../../../templates/targets/go_wam`, plus a thread-local test override, exactly
like `cpp_template_root/1`. This replaces `resolve_template_path/2`'s
cwd-then-module lookup for the runtime assets (cwd lookup is a hazard: a
stray `templates/` in the cwd would win).

### 6.2 `wam_go_target.pl` after the migration

| Today | After |
|---|---|
| `compile_wam_helpers_to_go(_Options, GoCode) :- format(atom(GoCode), '<1,942 lines>', []).` | `compile_wam_helpers_to_go(_Options, GoCode) :- go_render_helper_methods(GoCode).` (exported name kept; 3 tests and `compile_wam_runtime_to_go/2` keep working) |
| `compile_step_wam_to_go/2` + `compile_go_step_case/1` + 50 `wam_go_case/2` facts | `compile_step_wam_to_go(_Options, GoCode) :- go_render_step_method(GoCode).` where the adapter does: `go_step_case_order(Names), maplist(case_line, Names, Lines), atomic_list_concat(Lines, '\n', Cases), splice step_shell [cases=Cases]`, and `case_line(Name, L) :- go_step_case_text(Name, Body), format(atom(L), '    case *~w:\n~w', [Name, Body])`. The `.pl` still owns the order, the `case *T:` line, and the separator — the **assembly** — and owns no Go text. |
| `wam_go_case/2` facts | removed. Docs that cite them (`docs/WAM_GO_STATUS.md:208`, `docs/reports/wam_bindthrough_cross_target_sweep.md:31`, `benchmarks/go_wam_runtime_findings.md:73`) get a one-line pointer to `templates/targets/go_wam/step/`. Optionally keep `wam_go_case(Type, Body) :- go_step_case_text(Type, Body).` as a thin compatibility accessor for one release. |
| lines 117–127: `render_template(RuntimeTemplate, [package_name, step_method, helper_methods])` | `go_render_template(runtime_shell, [package_name=PackageName], RuntimeCode)` — the shell renderer splices `{{package_name}}` (a variable) and the section/step slots (children it renders itself). No rescan of inserted Go. |
| line 91 `make_directory_path(ProjectDir)` before any rendering | `go_preflight` **before** `make_directory_path/1` (C++ precedent). Success path unchanged; failure path now leaves no directory. |
| `read_template_file/2` silent fallback (417–423) | still used for go.mod/value/instructions/state/main_bench in Phases 1–3 (unchanged bytes). Phase 4 may route those through the adapter too, at which point the fallback comment is deleted. |

`wam_go_lowered_emitter.pl` is untouched until Phase 4.

### 6.3 Rendering rules inside the adapter

1. **Shells**: split on the declared markers in order, each exactly once
   (`findall(At, sub_string(Text, At, L, _, Marker), [At])`), lint every static
   span (§6.4), then `atomics_to_string/3` once over spans and children. This
   is `runtime_shell_parts/2` / `program_shell_parts/4` from the C++ adapter,
   with a Go marker list.
2. **Static sections**: read with `read_file_to_string(Path, S, [encoding(utf8)])`,
   require non-empty, require final LF, strip exactly one final LF, lint, return
   literally. Never passed to `render_template/3`.
3. **Libraries**: `render_template(FileText, [instr=Name], Body0)` through the
   generic engine (this *is* the engine's designed use; the case set is
   validated by a light `{{case ` scan in preflight), then require
   `Body0 = "\n" ++ Body ++ "\n"` and return `Body`; an empty or
   un-bracketed body is `go_wam_step_case_shape(Name)`. A library file's
   only variable is `instr`; the adapter never passes other keys, so
   `render_template_string` cannot touch Go text.
4. **Determinism**: same inputs → same bytes; add a repeated-render test.

### 6.4 The Go-aware tag lint (replaces C++'s "no `{{` at all")

Reject, in any static span or section:

- any **declared marker** of any Go template (`{{package_name}}`, `{{cases}}`,
  `{{step_method}}`, `{{run_loop}}`, …) appearing in a fragment, or appearing
  other than exactly once in its own shell;
- any **structural tag**: `{{#`, `{{^`, `{{/`, `{{>`, `{{!`, `{{match `,
  `{{case `, `{{default}}`.

Ignore everything else, in particular Go composite literals (`{{source, 0}}`,
`{{next, source}}`) and bare `}}`. Rationale: those are program text and the
splicing renderer never interprets them; the lint's job is to catch a
*template-origin* tag that would otherwise be output unresolved, which is the
C++ plan's own distinction ("do not reject arbitrary `{{...}}` sequences in
final C++ indiscriminately"). A false positive (e.g. a future
`[]struct{b bool}{{!ok}}`) fails loudly with the line number and is rewritten
by the author.

---

## 7. Byte-identity strategy

### 7.1 Extraction by execution, never by copy

The template files are **written from the running generator**, not
copy-pasted from the `.pl`:

```prolog
% one-off extraction script (scripts/dev/extract_go_wam_templates.pl, deleted after Phase 3)
compile_wam_helpers_to_go([], H),            write_utf8('runtime/helpers.go.mustache', H),
forall(wam_go_case(T, B), write_case(T, B)), % grouped by family, in findall order
```
This resolves `''`→`'`, `\\n`→`\n`, `\` line continuations, and Unicode in one
step, and it is the only way to guarantee that the file holds the *rendered*
bytes. The Phase 2 split is done on the extracted file at declaration
boundaries (blob lines listed in §2.3), and the split is validated by
re-concatenation before any Prolog changes.

### 7.2 Encoding and line endings

- `wam_go_target.pl` is `:- encoding(utf8)`; the Go bodies contain em dashes
  and `×`. The adapter reads templates with `encoding(utf8)` **explicitly**
  (the C++ adapter does), so template bytes do not depend on the locale.
  `write_file/2` still writes with the locale encoding; the harness pins
  `LANG=C.utf8 LC_ALL=C.utf8`, and that requirement is unchanged.
- `.gitattributes` already has `* text=auto eol=lf`; add `*.mustache text eol=lf`
  and `*.stache text eol=lf` for explicitness. A CRLF or a missing final LF in a
  fragment fails preflight (`go_wam_template_eol/2`), so it cannot silently
  change output.

### 7.3 Whitespace contract (the one place bytes are easy to lose)

Facts: `Step` text has no trailing LF; the helper blob ends with one LF; the
current shell is `{{step_method}}\n\n{{helper_methods}}\n`; file ends `}\n\n`.
Cases are joined by `\n` and each body has no trailing LF.

Rule: **a fragment file is a POSIX text file (ends with exactly one LF); the
adapter strips exactly that LF and requires it to be there.** Then:

- Phase 1 shell: `{{step_method}}\n\n{{helper_methods}}\n\n` → Step ++ `\n\n`
  ++ Helpers′ ++ `\n\n` = Step ++ `\n\n` ++ (Helpers minus LF) ++ `\n\n`
  = the old bytes (old: Helpers-with-LF ++ `\n`). ✔
- Phase 2 shell (§5.1): each section Pᵢ′ = old text between declarations
  minus its final LF; markers on their own lines separated by one blank line
  reproduce `}\n\nfunc …` between sections and `}\n\n` at EOF. The split
  script asserts `Helpers == join("\n\n", Sections′) ++ "\n"` before anything
  is committed. ✔
- Step shell file ends `}\n`; stripped → identical to today's `Step` text. ✔
- Library cases: body between tag lines = `"\n" ++ Body ++ "\n"`; the adapter
  strips both; `Body` is byte-identical to today's `wam_go_case/2` second
  argument. ✔ (The last case before `{{/match}}` is followed by the LF that
  ends its last line, then the tag on its own line — same shape.)

This is standalone-line trimming implemented **in the adapter**, consistent
with the SPEC's "caller owns whitespace policy"; neither engine changes.

### 7.4 Pinned digests (new `tests/test_wam_go_templates.pl`)

Modelled on `tests/test_wam_cpp_templates.pl`:

| Test | Asserts |
|---|---|
| `helpers_exact_bytes` | `compile_wam_helpers_to_go([], H)`: 61,313 chars, SHA-256 `dab3b8c2…d06552` (Phase 1: with the trailing LF re-added by the exported predicate if the design keeps its old contract; otherwise re-baseline the *predicate* digest in the same PR and keep the *file* digest unchanged — see §10 Q1) |
| `step_exact_bytes` | `compile_step_wam_to_go([], S)`: 27,593 chars, SHA-256 `433eea3c…1416e` |
| `runtime_go_exact_bytes` | `write_wam_go_project` of the pkg_resolver predicate set (or a smaller frozen fixture) → runtime.go 89,244 bytes, SHA `6b0e707c…dabbd`; state.go `f5a9b98f…97f0` |
| `case_order_and_set` | `go_step_case_order/1` has 50 distinct names in the §2.4 order; every library case name is in the list and vice versa |
| `case_bodies_literal` | a fixture library whose body contains `{{instr}}`, `{{package_name}}`, `[]qd{{source, 0}}` renders those verbatim |
| `shell_does_not_rescan` | a fixture section containing `{{step_method}}` text is spliced literally |
| `foreign_cwd`, `repeated_render` | as in C++ |
| preflight fault fixtures | missing/empty/malformed section; duplicate/reordered/missing shell marker; missing/duplicate/unknown case; missing final LF; CRLF; no project directory created |

### 7.5 Gold comparison procedure (per phase)

```bash
# at the base commit (or with the base generator checked out in a worktree)
LANG=C.utf8 LC_ALL=C.utf8 swipl -q -g main -t halt examples/pkg_resolver/go/build.pl -- \
     examples/pkg_resolver/resolver.pl /tmp/gold/go
# at the refactor commit
LANG=C.utf8 LC_ALL=C.utf8 swipl -q -g main -t halt examples/pkg_resolver/go/build.pl -- \
     examples/pkg_resolver/resolver.pl /tmp/new/go
diff -r /tmp/gold/go /tmp/new/go && echo BYTE-IDENTICAL
```
Note (§2.5): `build.pl` honours `argv` `[Src, OutDir]`, so both builds can go
to scratch directories; do **not** compare against the checked-in `go` lane
at HEAD (stale). Repeat for `go_store` with
`WAM_OUT=/tmp/gold/go_store bash examples/pkg_resolver/go_store/build.sh`
(its `build.sh` honours `WAM_OUT`; the store fixtures under
`examples/pkg_resolver/store/.out/corpus` are built on first run).

---

## 8. Phased migration (each phase independently byte-verifiable and revertible)

### Phase 0 — freeze (no generator change)
- Add `tests/test_wam_go_templates.pl` with the §7.4 digests against the
  *current* predicates (they pass before any refactor — that is the point).
- Add the extraction script (§7.1). Record `swipl` version and commit.
- Gate: new test file green; `bash examples/pkg_resolver/go/build.sh` +
  corpus/differential green (§9) — establishes the harness works on this host.

### Phase 1 — helper blob → one static section (biggest hazard, lowest risk)
- Extract the rendered blob to `runtime/helpers.go.mustache` (temporary name).
- Add `wam_go_templates.pl` with registry, root resolution, shell splicing,
  section loading, lint, preflight.
- `compile_wam_helpers_to_go/2` → `go_render_helper_methods/1`;
  `write_wam_go_project/3` renders runtime.go through the adapter; preflight
  moved before `make_directory_path/1`. Delete the 1,942-line atom.
- Gate: §7.4 digests unchanged; §7.5 diff empty for `go` and `go_store`;
  the three tests calling `compile_wam_runtime_to_go/2` green; corpus 51/51,
  differential 2600/0, store 51/51 + 503/0.
- Result: `wam_go_target.pl` shrinks by ~1,940 lines; no Go apostrophe/tilde
  hazard remains in the blob.

### Phase 2 — split the section into eight ordered partials (relocation)
- Split `runtime/helpers.go.mustache` at the §2.3 boundaries into the eight
  `runtime/*.go.mustache` files; shell gains the eight markers (§5.1); the
  split script re-concatenates and asserts equality before commit.
- Gate: identical to Phase 1's, plus fault fixtures for each new section.
- Boundaries may be adjusted at execution (the gate does not care where the
  cuts are, only that concatenation is exact and each file is coherent).

### Phase 3 — the 50 step cases → five literal-match library files
- Extraction script writes the five family files in findall order; add
  `go_step_case_order/1` (the 50 names, §2.4), `step/step_shell.go.mustache`,
  library loading with the §7.3 trimming and the case-set check.
- `compile_step_wam_to_go/2` → `go_render_step_method/1`; delete the 50 facts
  and `compile_go_step_case/1`; update the three doc references.
- Gate: `step_exact_bytes` unchanged; §7.5 diff empty; full Go plunit list
  (§9.2) green; harness green.
- Measure generation time before/after (20 runs of `build.pl`); expect
  single-digit ms change. Add a parsed-case cache only if a real project shows
  material cost (C++ precedent: deferred).

### Phase 4 — lowered emitter and project leftovers, only where composition pays

All Phase 4 templates are `.mustache`; there is **no `.stache` slice** (§4.4
decided it, with a recorded reopen condition). Slices, in order:

- **Slice 0 — freeze lowered bytes (no generator change).** The Go lowered
  suites (`test_wam_go_lowered_phase{1,2,3}.pl`, `_t{4,5,6}.pl`,
  `_ite_exec.pl`) assert with `sub_string`/`sub_atom` only; none pins a
  digest, so today nothing would catch a one-byte drift in a lowered
  fragment. Add to `tests/test_wam_go_templates.pl`, modelled on the C++
  `old_head_constant_digest/4`, `old_head_integer_nil_digest/3` and
  `old_lowered_function_digest/3` tables:
  - per-fragment digests of `with_output_to(string(T), emit_one(Instr, "    "))`
    for `get_constant` with an atom, an integer token, a float token, an
    escaped atom (`'a\"b'`, `'a\\b'`) and the placeholder-shaped atoms
    `'{{name}}'` and `'{{Ai}}'` (the rescan trap), `get_integer` with `42`,
    `-7` and the raw `0007`, and `get_nil` on `A1`/`Y3` — each captured
    right after `wam_go_target:init_atom_intern_table_go/0` so the interned
    variable names (`wamAtom_<sanitized>_<seq>`) are stable across runs;
  - whole-function digests of `lower_predicate_to_go/4` on fixed WAM lists
    covering the plain, T4, T5, T6 (`t6_min_clauses(3)` as `_t6.pl` does),
    single-ITE, sequential-ITE and nested-ITE shapes — length + SHA-256 of
    `atomic_list_concat(Lines, '\n', Code)`.
  - Gate: the new tests pass against the *unchanged* emitter.
- **Slice 1 — `lowered/function.go.mustache` + `lowered/ite.go.mustache`**
  shells (the three header `format/2`s at 348–352, 363–365, 407–409 share
  one shape; the ITE block 506–527 is a fixed shell around three rendered
  children plus the fresh-variable reset) — C++ `#4263`/`#4264` are the
  template. Rendered by ordered exactly-once (function) and
  ordered-repeated (ITE: every `{{indent}}` occurrence listed, as
  `ite_shell_markers/1` does) marker splices.
- **Slice 2 — `lowered/head_match.go.mustache`** for
  `get_constant`/`get_integer`/`get_nil`: three duplicated twelve-line bodies
  → one fragment with the repeated ordered slots `{{I}}`×12, `{{Comment}}`,
  `{{Ai}}`, `{{GoVal}}`×2. Prolog keeps everything it keeps in C++
  (`CPP_WAM_TEMPLATE_REFACTOR_PLAN.md`, "shared head bind/match body"):
  register normalisation, token classification, `intern_atom_go/2`, the raw
  numeric spelling for `get_integer`, and the exact comment. The adapter
  gets `go_render_head_match/5` (`I`, `Comment`, `Ai`, `GoVal` → text) with a
  four-key contract check and preflight render; the emitter's three clauses
  become `emit_head_match(Comment, Ai, GoVal, I)`. The fragment is **never**
  passed to `template_system:render_template/3` (sequential whole-string
  replacement would rescan a `Comment` containing `{{Ai}}`; slice 0's
  digests for `'{{Ai}}'` pin that).
- **Slice 3 — `project/*.go.mustache`** for the remaining small
  atom-embedded Go in `write_wam_go_project/3` (atoms.go fallback, lib/lowered
  headers, parallel main) and routing go.mod/value/instructions/state/
  main_bench through the adapter, deleting the silent `// Template not found`
  fallback (§2.1).
- **Not a slice — `.stache`.** Dropped per §4.4. Reopens only when a Go
  instruction acquires a second statement body selected by a bound
  sub-term's shape (the realistic case: a `get_constant` atom fast path, a
  performance change outside this refactor); the design for that moment is
  already written there and needs a re-baseline of slice 0's digests in the
  PR that changes the bytes, followed by a byte-identical hygiene PR.
- Gate per slice: slice-0 digests unchanged, lowered plunit suites (§9.2)
  green, harness green (§9.1), no Go text left in the `.pl` for that slice's
  scope.

Rollback: every phase is a single PR touching adapter + templates + call sites
together; reverting it restores the previous byte-identical state.

---

## 9. Verification harness

### 9.1 Generated-output gates (from repo root, `LANG=C.utf8 LC_ALL=C.utf8`)

```bash
bash examples/pkg_resolver/go/build.sh                 # regenerates the go lane in place + go build
bash examples/pkg_resolver/go/run_corpus_go.sh         # expect 51/51 vs SWI
bash examples/pkg_resolver/go/run_differential_go.sh   # expect 2600 cases, 0 divergences, 0 crashes
bash examples/pkg_resolver/go_store/build.sh
bash examples/pkg_resolver/go_store/run_corpus_go_store.sh        # 51/51, byte-identical to ../go corpus
bash examples/pkg_resolver/go_store/run_differential_go_store.sh  # 503 cases, 0 divergences
```
Plus the §7.5 generator-vs-generator `diff -r` (the gold test for a pure
refactor). Because `build.sh` regenerates in place, `git diff --stat
examples/pkg_resolver/go` after a base-commit build and again after the
refactor build is an equivalent check *if the base build is committed first*;
the scratch-directory form avoids touching the checked-in lane.

### 9.2 Prolog suites (plunit unless noted; inspect each runner)

- New: `tests/test_wam_go_templates.pl` (§7.4).
- Existing Go suites (all must stay green; the three marked `*` call
  `compile_wam_runtime_to_go/2` directly): `test_wam_go_generator.pl`,
  `test_wam_go_foreign_lowering.pl`\*, `test_wam_go_iso_smoke.pl`\*,
  `test_go_lmdb_materialisation.pl`\*, `test_wam_go_bindthrough.pl`,
  `test_wam_go_cut_semantics.pl`, `test_wam_go_switch_default_chain.pl`,
  `test_wam_go_switch_on_structure.pl`, `test_wam_go_last_call_builtin.pl`,
  `test_wam_go_maplist_predsort.pl`, `test_wam_go_negation_cut.pl`,
  `test_wam_go_findall_freeze.pl`, `test_wam_go_empty_list_constant.pl`,
  `test_wam_go_varid_collision.pl`, `test_wam_go_frameless_ite_level.pl`,
  `test_wam_go_lowered_phase{1,2,3}.pl`, `test_wam_go_lowered_t{4,5,6}.pl`,
  `test_wam_go_lowered_ite_exec.pl`, `test_wam_go_parser_smoke.pl`,
  `test_wam_go_sort_compounds.pl`, `test_go_wam_builtins.pl`,
  `test_wam_go_foreign_tuple_aliases.pl`, `test_wam_go_native_t5_parallel.pl`.
- Engine suites (unchanged engines, but run them): `tests/test_template_match_case.pl`
  (note the C++ plan's caveat: 29/31 at its baseline for F#-related reasons),
  `tests/core/test_template_sections.pl`, `tests/core/test_pattern_stache.pl`,
  `tests/core/test_pattern_stache_whitespace.pl`.
- Core suite from `AGENTS.md` / `docs/TESTING.md` (`mkdir -p output/advanced` first).

### 9.3 What counts as done for a phase

Digests unchanged (or, for an intentionally re-baselined *predicate* digest,
the *file* digests unchanged and the PR says why), `diff -r` empty for both
lanes, corpus/differential numbers as above, all suites green, and no
Go text left in the `.pl` for that phase's scope.

---

## 10. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Hand-copying the atom text (keeps `''`, `\\n`, drops line continuations) | Extraction by execution only (§7.1); digest tests catch any slip. |
| 2 | Off-by-one LF at section joins / EOF | Single explicit rule (§7.3) enforced by the adapter (final-LF required, stripped once); split script asserts re-concatenation; file digests. |
| 3 | Go `{{…}}` composite literals mistaken for tags | Splicing renderer never interprets static text; Go-aware lint (§6.4) targets declared markers and structural tags only; fixture test with `[]qd{{source, 0}}`. |
| 4 | Editor adds/removes a final newline or converts EOL | `.gitattributes` entries; preflight EOL check; digests. |
| 5 | Locale-dependent template read (em dashes) | `encoding(utf8)` on every read; harness keeps `C.utf8` for the write side; `foreign_cwd`-style test. |
| 6 | cwd-relative template lookup picks a stray file | Adapter uses module-relative root only, test override via thread-local (C++ pattern). |
| 7 | Silent fallback comment masks a missing template | Adapter throws `go_wam_template_load/3`; preflight before directory creation; fault fixtures. |
| 8 | Switch order drifts when bodies move between family files | Order lives in `go_step_case_order/1`, not in file order; set-equality check; `step_exact_bytes`. |
| 9 | Generic engine behaviour changes later (it is shared by 67 targets) | Only the library path uses it, with a single declared key; the adapter's post-conditions (`"\n"…"\n"` shape, non-empty) and digests would fail loudly. If that ever bites, replace with a bounded `{{case ` splitter in the adapter without touching templates. |
| 10 | Performance regression from 50 library renders + 9 file reads | Measure (C++ measured tens of ms per project); cache only with evidence, keyed by path + SHA-256 as in `load_head_constant_stache/2`. |
| 11 | Checked-in `examples/pkg_resolver/go` lane is stale at HEAD | Gold is generator-vs-generator (§7.5); regenerate the lane in Phase 0 if a committed golden copy is wanted. |
| 12 | Concurrent Go-runtime edits during the migration (D-series perf work) | Land Phase 1 first and fast (it removes the hazard where those edits happen); rebase later phases; each phase's digest tests make merge conflicts visible as byte changes. |

---

## 11. Open decisions for the owner

- **Q1 — trailing LF of `compile_wam_helpers_to_go/2`.** With the uniform
  strip rule, the exported predicate returns the blob *without* its old final
  LF while `runtime.go` stays identical. Either (a) re-baseline that predicate's
  digest (its callers only grep content), or (b) have the exported wrapper
  append `"\n"` to preserve the old contract exactly. Recommendation: (b)
  during Phases 1–3 (zero behaviour change anywhere), drop it in Phase 4.
- **Q2 — five family files vs one library file.** Recommendation: five (§5.4).
- **Q3 — keep a `wam_go_case/2` compatibility accessor** for one release
  (docs cite it; no test does). Recommendation: yes, one clause delegating to
  the adapter, removed in Phase 4.
- **Q4 — section boundaries** (§2.3) are a proposal; the split script may cut
  differently as long as re-concatenation is exact.
- **Q5 — Phase 4 scope**: whether to open the lowered-emitter work at all.
  The `.stache` half of this question is **closed** by §4.4: no Go
  instruction has a second body shape selected by a bound sub-term, so no
  `.stache` library is justified in this refactor; the reopen condition is
  recorded there. What remains for the owner is only whether slices 0–3
  (§8) are worth doing now; recommendation: slice 0 (freeze lowered bytes)
  unconditionally, since it costs one test table and protects the lowered
  lane regardless, then slices 1–2 together as one PR, slice 3 last.

---

## Appendix A — adapter sketch (shell splice + library case)

```prolog
go_render_template(runtime_shell, [package_name=Pkg], Text) :-
    go_template_root(Root),
    go_shell_markers(runtime_shell, Markers),               % ["{{package_name}}","{{step_method}}","{{run_loop}}",...]
    read_asset(Root, runtime_shell, Shell),
    split_ordered_source_markers(Shell, Markers, Spans),     % exactly once, in order (C++ helper)
    maplist(go_lint_span, Spans),                            % §6.4
    go_render_step_method(Step),
    maplist(go_section_text(Root), [runtime_run_loop, ...], Sections),
    interleave_source_slots(Spans, [Pkg, Step | Sections], Parts),
    atomics_to_string(Parts, "", Text).

go_section_text(Root, Id, Text) :-
    read_asset(Root, Id, Raw), Raw \== "",
    ( string_concat(Text, "\n", Raw) -> true ; throw(error(go_wam_template_eol(Id), _)) ),
    go_lint_span(Text).

go_step_case_text_at_root(Root, Name, Body) :-
    go_step_library_for(Root, Name, LibText),                % preflight built the name→file map
    template_system:render_template(LibText, [instr=Name], Rendered),
    ( string_concat("\n", Mid, Rendered), string_concat(Body, "\n", Mid), Body \== ""
    -> true ;  throw(error(go_wam_step_case_shape(Name), _)) ).

go_render_step_method(Step) :-
    go_step_case_order(Names),
    maplist([N, L]>>( go_step_case_text(N, B), format(atom(L), '    case *~w:\n~w', [N, B]) ), Names, Lines),
    atomic_list_concat(Lines, '\n', Cases),
    go_render_template(step_shell, [cases=Cases], Step).    % splice; step_shell's final LF stripped
```

## Appendix B — files touched per phase

| Phase | Adds | Changes | Deletes |
|---|---|---|---|
| 0 | `tests/test_wam_go_templates.pl`, extraction script | `.gitattributes` | — |
| 1 | `wam_go_templates.pl`, `runtime/helpers.go.mustache` | `wam_go_target.pl` (119–127, 3153, preflight order), `runtime.go.mustache` | 1,942-line atom |
| 2 | `runtime/*.go.mustache` ×8 | `runtime.go.mustache`, registry | `runtime/helpers.go.mustache` |
| 3 | `step/step_shell.go.mustache`, `step/*.go.mustache` ×5 | `wam_go_target.pl` (2348–2364), three docs | 50 facts, `compile_go_step_case/1` |
| 4 | `lowered/*`, `project/*` as chosen | `wam_go_lowered_emitter.pl`, remaining `format/2` shells | `read_template_file/2` fallback |
