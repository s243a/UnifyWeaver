<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 John William Creighton (s243a)
-->

# plawk — Implementation Plan

*Project: UnifyWeaver — Hybrid WAM/LLVM Target*
*Author: John Creighton*
*Status: Early Design / Prototype Phase*

> **Companion docs:** [Philosophy](PLAWK_PHILOSOPHY.md) ·
> [Specification](PLAWK_SPECIFICATION.md) ·
> [Submodule README](../../examples/plawk/README.md)

---

## Codebase reconciliation (audit findings)

Before the phased plan, three findings from auditing the existing target. They
change *what* the early phases actually have to build.

### Finding 1 — Mode declarations are already consumed by codegen ✅

`:- mode foo(+,-,?)` is read by `demand_analysis:read_mode_declaration/3` (via
`user:mode/1`), mapped `+/-/?` → input/output/any, and fed to
`binding_state_analysis`, which computes per-variable bound/unbound states at
each program point (`binding_state_at_var/3`). `wam_target.pl` — the WAM pipeline
the LLVM target rides — consults those states at dozens of sites to specialise
builtin lowering (forward vs reverse `atom_concat`, deterministic `sub_atom`,
`memberchk` vs `member`) and to index on input positions.

**Consequence:** the "single mode ⇒ less backtracking" thesis is real and
**already wired**. The DSL gets it for free by declaring `:- mode` on its core
predicates.

### Finding 2 — `:- det` declarations are *not* consumed ❌

There is no determinism-directive parser in `core/`. Determinism is achieved
**structurally**: cut (`!` → the clause's `cut_barrier`), if-then-else, first-
argument / switch indexing, and `musttail` TCO in `llvm_target.pl`; plus each
builtin is hand-lowered to push-or-not a choicepoint based on its own known
determinism (sometimes gated on binding state).

**Consequence:** Specification §8's machine-checkable `:- det` directive that
*drives choicepoint elision for user predicates* is **net-new work**, not an
existing lever. Two paths:

- **(a) Rely on structural determinism (recommended for Phase 0–1).** Write
  `handler`/`reader`/`select_writer` with cut / if-then-else / single-clause-
  per-shape so they are already deterministic; declare `:- mode` so the existing
  binding analysis kicks in.
- **(b) Add a `:- det` pass later (measurement-justified).** A directive parser
  + a determinism-check/choicepoint-elision pass, only if profiling shows
  residual choicepoints in the hot loop after (a).

### Finding 3 — No streaming input builtin; one needs adding 🔧

The LLVM target's only input builtins are whole-file (`read_file_to_atom/2`,
`read_link/2`). There is no `read/1`, `get_char`, `get_code`, or `read_line`. The
awk lazy record loop needs a buffered reader that holds an fd + buffer position
**across calls**.

**Builtin addition surface** (from M129 `read_file_to_atom`): a builtin is three
things — (1) a registration `builtin_op_to_id('name/arity', N).`; (2) a dispatch
entry in the op switch, `i32 N, label %builtin_name`; (3) a hand-written IR block
`builtin_name:` that reads registers via `wam_get_reg_deref`, does the work via
already-linked libc (`open`/`read`/`close`/`stat`), allocates with
`wam_arena_alloc`, interns results with `wam_intern_atom`, and binds outputs with
`wam_unify_value`, returning `i1`.

**Sketch — buffered streaming line reader** (recommended; this *is* `reader/4`):

```
runtime struct  WamLineReader { i32 fd; i8* buf; i64 cap; i64 len; i64 pos; }
                arena-allocated; the DSL handle is the pointer as an i64 Value
                (or an index into a small fixed stream table). Handle 0 ⇒ fd 0.

stream_open(+Path, -H)
    open(Path, O_RDONLY); arena_alloc(struct) + arena_alloc(cap=65536);
    len=pos=0; return H.

read_line(+H, -Line)
    scan buf[pos..len) for '\n';
    if not found: memmove(buf, buf+pos, len-pos); pos=0; len=remainder;
                  n = read(fd, buf+len, cap-len); len += n; rescan;
    on EOF with empty buffer: unify Line = atom 'end_of_file'  (do NOT fail —
        preserves the EOF/error distinction from Specification §1);
    else: intern buf[pos..nl) (excluding '\n') via wam_intern_atom; advance pos;
          unify Line.

stream_close(+H)
    close(fd).
```

Roughly ~120–180 lines of IR following the M129 template, plus the small runtime
struct. Tractable, and it unblocks every later phase.

**Phase-0 shortcut (no runtime change):** `read_file_to_atom(Path, Whole)` +
`split_string(Whole, "\n", "", Lines)`, iterate `Lines` in Prolog. Correct, lets
the core loop be validated immediately, but not unbounded-stream-safe — replace
with the buffered reader in Phase 1.

---

## Phase 0: Prolog core prototype (no LLVM)

**Goal:** validate the execution model and API in standard SWI-Prolog.

1. Implement `process_all/4` per Specification §1 (det Reader, `end_of_file`
   sentinel).
2. Implement a text line reader (Phase-0 shortcut: slurp + `split_string/4`).
3. Implement `item_field/3` for list-based `Fields`.
4. Implement `state/N` with counter + output slot.
5. Implement `select_writer/2` for text records (stdout).
6. End-to-end test: count all records; print those where `$1 == "ERROR"`.
7. Add `:- mode` declarations to all predicates.
8. **Verify determinism with SWI tooling** — e.g. `setup_call_cleanup`/
   `deterministic/1` or the determinism checker — to *prove* handlers succeed
   exactly once and leave no choicepoints. This is the evidence the Phase-1
   codegen story depends on (Finding 2, path a).

**Success:** a Prolog program behaving like a minimal `awk`, using only the core
predicates, with determinism demonstrated.

Lands in `examples/plawk/core/`.

### Phase-0 compile probe status (2026-06-08)

The first WAM/LLVM probes now live under `examples/plawk/probes/`.

- **Core helper probe:** `state_counter/2`, `increment_counter/2`,
  `item_field/3`, `item_field_count/2`, `nr/2`, `nf/2`, `fs/2`, `ofs/2`,
  `append_output/3`, `state_outputs/2`, `print_item/3`, and `print_fields/3`
  generate LLVM IR through WAM fallback. WAM/LLVM now computes a conservative
  same-module helper closure before project classification, so private helper
  dependencies such as `normalize_outputs/2` no longer need to be listed as
  probe roots. The generated helper probe now verifies with `llvm-as`; the
  earlier `%Instruction` array mismatch was fixed
  by lowering `switch_on_constant_fallthrough` to a real no-op instruction
  instead of a `; TODO` comment inside `@module_code`.
- **Loop probe:** `process_all/4` now emits IR without unresolved-label warnings,
  and the generated loop probe verifies with `llvm-as`. WAM/LLVM recognizes
  `call/N` bytecode targets and routes meta-calls through a generated numeric
  dispatch table, preserving the existing numeric label-to-PC jump path. Atom
  goals dispatch by `(atom_id, effective_arity)`. Compound closures dispatch by
  `(compiled_functor_pointer, effective_arity)` and lay out captured closure
  arguments before the extra `call/N` arguments. This keeps the hot path numeric
  rather than doing string lookup at runtime.
- **Meta-call probe:** atom and compound closure meta-calls now emit IR and
  verify with `llvm-as` in `examples/plawk/generated/plawk_meta_call_probe.ll`.
- **Reader probe:** the WAM/LLVM target now has general buffered stream builtins
  `stream_open/2`, `read_line/2`, and `stream_close/1`. Handles are numeric
  IDs into a target-owned `%WamLineReader` table carrying fd, buffer, length,
  and position state; `read_line/2` reads through a 4 KB buffer, returns line
  atoms without the trailing newline, and unifies `end_of_file` at EOF rather
  than failing. The reader probe emits
  `examples/plawk/generated/plawk_reader_probe.ll` and verifies with `llvm-as`.
  `read_line/2` builds each output line in a malloc-owned temporary buffer before
  interning, growing the buffer as needed, so long streams do not consume WAM
  arena space per record and individual line atoms are not capped at 64 KiB.
- **Compiled stream-core smoke:** `tests/test_plawk_compiled_stream_core.pl`
  builds a native LLVM binary that streams a text file, splits records with
  `atom_split/3`, runs a PLAWK-style handler over `record(text, Line, Fields)`,
  counts records via `state/4`, and collects the two `ERROR` lines through the
  normal `append_output/3` and `state_outputs/2` path. The smoke also exercises
  the WAM/LLVM choice-point stack restore needed for else-branch control flow
  after called predicates allocate and fail.

---

## Phase 1: UnifyWeaver LLVM integration

**Goal:** transpile the Phase-0 core to LLVM via the hybrid WAM target.

1. Compile the Phase-0 predicates through the existing pipeline; confirm `:- mode`
   drives binding analysis (Finding 1) — no new mode plumbing expected.
2. Confirm the tail call in `process_all/4` is lowered with `musttail` (loop, not
   stack growth); if not, identify why the clause isn't recognised as
   last-call-deterministic.
3. **Add the buffered streaming reader builtin** (Finding 3 sketch): `stream_open`,
   `read_line`, `stream_close` in `wam_llvm_target.pl`; replace the Phase-0
   slurp shortcut.
4. Map `state/N` to an LLVM aggregate; thread as a pointer parameter.
5. Map `item_field/3` to an array index on the aggregate.
6. Validate the native binary against the Phase-0 SWI prototype on identical
   inputs.

**Success:** a native binary that reads stdin, counts records, prints matching
lines — identical behaviour to Phase 0, running as compiled LLVM.

**Current boundary:** the compiled and native stream smokes still read from file
paths rather than stdin. WAM/LLVM exposes general stream helpers
(`@wam_stream_open_value`, `@wam_stream_read_line_value`, and
`@wam_stream_close_value`) that native LLVM code can call directly, alongside
the existing `stream_open/2`, `read_line/2`, and `stream_close/1` builtins.
The `tests/test_plawk_native_stream_loop_driver.pl` smoke proves a native LLVM
loop can open a runtime file path, read lines until `end_of_file`, call a
compiled PLAWK handler once per record, and thread PLAWK state through WAM.
The `tests/test_plawk_native_counter_stream_loop_driver.pl` smoke then lowers
the hot record counter to a native `i64` loop variable while keeping output
accumulation as ordinary WAM terms via `append_output/3` and `state_outputs/2`.
The `tests/test_plawk_native_output_stream_loop_driver.pl` smoke moves the next
piece of hot-loop state into LLVM: native code owns the reader, record counter,
output counter, and fixed output slots, while WAM only returns the deterministic
handler decision (`yes`/`no`) for each record. The next boundary is lowering
more deterministic handler logic itself into native code without making the
target PLAWK-specific. The first reusable emitter path for that boundary is now
`llvm_emit_atom_prefix_guard/5`, which emits a prefix global plus a call to the
general `@wam_atom_prefix_value` runtime helper. That lets native LLVM lower
deterministic `sub_atom(Line, 0, Len, _, Prefix)` guards without allocating a
substring or entering `run_loop` for each record. The
`tests/test_plawk_native_lowered_handler_stream_loop_driver.pl` smoke proves
that emitter-backed shape in a native stream loop.

**Compiler note:** WAM/LLVM now normalizes quoted atom tokens before interning.
Atoms such as `'ERROR disk full'` and `'it\'s bad'` compile to raw runtime
atom names without the outer WAM token quotes.
`tests/test_wam_llvm_quoted_atom_literals.pl` covers that regression.

---

## Phase 2: AWK syntactic sugar

**Goal:** an awk-like front-end that parses programs and emits Phase-0/1 core.

1. AST: `program(BeginClauses, Rules, EndClauses)`, `rule(Pattern, Body)`.
2. Prolog/DCG parser for the awk subset (Specification §9).
3. Code generator AST → Prolog core (one guarded clause per pattern-action pair,
   so `next` is structural — Specification §7).
4. Integrate into the pipeline: awk source → AST → core → WAM IR → LLVM → binary.
5. `BEGIN`/`END` → init/finalize predicates around `process_all/4`.

**AWK subset (minimum viable):** field-comparison and regex patterns, `BEGIN`/
`END`; actions `print`, basic `printf`, `$N = expr`, var assignment, `++`/`+=`/
`is`, `if/else`, `next`, `break`; specials `$0`/`$N`/`NR`/`NF`/`FS`/`OFS`.

Parser/codegen land in `examples/plawk/{parser,codegen}/`.

**Current slice:** `examples/plawk/parser/plawk_parser.pl` parses the first
surface forms, `/^PREFIX/ { print $0 }` and
`$N == "VALUE" { print $0 }`, plus selected-field actions such as
`$N == "VALUE" { print $M, $K }`, and the first scalar state form,
`$N == "VALUE" { count++ } END { print count }`, to explicit pattern/action
ASTs. Rule bodies now carry semicolon-separated action lists, and scalar
variables lower through indexed native slots.
`examples/plawk/codegen/plawk_native_codegen.pl` lowers that AST to a native
streaming WAM/LLVM driver using `llvm_emit_atom_prefix_guard/5` or
`llvm_emit_atom_field_eq_guard/7`, and
`tests/test_plawk_surface_prefix_print.pl` proves that the generated binary
prints matching records from a text file without calling `run_loop` in the hot
record loop. The reusable file open/read/eof/close skeleton now lives in the
WAM/LLVM target as `llvm_emit_stream_driver_ir/3`, so PLAWK supplies only the
surface-specific globals, per-record lowering, continuation phis, and close
block. The field-equality path scans fields in native code without
allocating substrings; selected-field printing projects byte slices directly.
Rule prints can include native `NR`, implemented as a record-number `i64` phi in
the print-only streaming loop, and native `NF`, implemented with the shared
`@wam_atom_field_count_value` helper over the active single-byte `FS`. Rule
prints can also call native `length($N)` through the shared
`@wam_atom_field_length_value` helper, native `substr($N, Start, Len)` through
the allocation-free `@wam_atom_field_subslice_value` helper, and native
`index($N, "literal")` through the shared `@wam_atom_field_index_value` helper.
Print-only `tolower($N)` and `toupper($N)` lower through shared
`@wam_print_ascii_lower_slice` and `@wam_print_ascii_upper_slice` helpers, so
case mapping streams bytes without allocating a transformed atom.
Numeric field guards such as `$3 > 100` lower through the shared
`@wam_atom_field_i64_cmp_value` helper, which parses the projected field slice
as a strict signed decimal `i64` and compares with numeric op codes.
The parser itself is factored as `@wam_atom_field_i64_value`, returning a value
plus success flag, so the same machinery also feeds scalar expressions such as
`bytes += $3` and `last = $3`; PLAWK uses zero for failed numeric coercions in
those scalar arithmetic contexts. That coercion is emitted through the
target-side `llvm_emit_atom_field_i64_or_default/7` helper so future native
numeric consumers can share the parse-plus-default lowering instead of
rebuilding it in PLAWK-specific code.
The scalar counter
path threads a native `i64` loop variable and prints it from the `END` action. Multiple scalar counters
become parallel `i64` phi slots in the native streaming loop.
Scalar counters lower through an explicit codegen state plan that keeps
source-level state recognition separate from LLVM slot numbering. The same
native slots now support `+=` with integer constants and field lengths, so
programs such as `$1 == "ERROR" { bytes += length($0); hits += 2 } END { print bytes, hits }`
stay in the compiled stream loop. Plain scalar assignment to integer literals or
field lengths uses the same native slot path and is folded in source order with
later `++`/`+=` updates, so programs such as `$1 == "ERROR" { last_len =
length($0); hits++ } END { print hits, last_len }` also stay native. The first
native `if/else` action slice lowers field-equality conditions once at rule-body
scope, threads the whole scalar slot vector through then/else action sequences,
emits per-slot phis at the branch join, and can run field-key associative
increments or selected-field/string-literal `print` as branch-local side effects. Scalar,
mixed, and assoc-only branch bodies now share the same rule-body action walker;
branch phis use each branch's actual exit block, including assoc side-effect
`_done` blocks. Branch-local `print` uses prefixed SSA names so multiple branch
prints do not collide; branch-local `NR` printing uses the same native record
counter threaded through the stream loop as top-level `print NR`. Terminal
branch-local `next` branches directly to the stream-loop continuation and adds
the selected branch's scalar values to the loop phi inputs. Terminal
branch-local `break` branches to the same dedicated stream-close block as
rule-level `break` and feeds the selected branch's scalar values through
final-state phis. Native rule chains also support terminal `next` by branching
directly to the stream-loop continuation; scalar and mixed chains add the
early-exit rule's scalar values to the loop phi inputs, while assoc-only chains
update tables before the continuation branch. Terminal `break` uses the sibling
path: matching rules branch to a dedicated stream-close block and scalar/mixed
programs feed `END` through final-state phis. This native slice deliberately
supports terminal `next`/`break` only; non-terminal loop control needs a future
intra-rule control-flow lowering rather than implicit fall-through after the
skipped or final record. The current
associative-count surface allocates one reusable WAM/LLVM
interned-atom-keyed growable `i64` table primitive (`wam_assoc_i64_*`) per source
array, increments those tables in the native streaming loop, and performs `END`
lookups through the matching source-array table. Multiple associative increments
in one rule lower as sequential native action blocks, so no WAM dispatch is
needed per record for those count updates.
Guarded associative-count rules now reuse the same native guard emitters and
rule-chain structure as scalar counters, so the loop can run field/prefix checks
and table increments without per-record WAM dispatch for the supported surface.
Mixed scalar/associative programs use a combined native state plan: scalar
counters remain `i64` phi slots, assoc arrays remain runtime table pointers, and
`END` printing can interleave scalar variables with associative lookups.
`END` printing also supports string literal fields for report labels; codegen
emits indexed string globals and prints them with the same separator rules.
The first `BEGIN` slices emit literal `print` headers before stream setup, using
separate indexed string globals so they do not collide with `END` literals, and
thread `FS` assignments through native field-equality, selected-field print, and
associative-key extraction helpers. The default space `FS` follows AWK-style
whitespace splitting by ignoring leading/trailing whitespace and treating runs of
space, tab, CR, or LF as one separator; explicit non-space `FS` values still use
the single-byte literal path. Single-byte `OFS` assignments now configure the
separator used by comma-separated `print` fields in `BEGIN`, rule, and `END`
actions. Native separator emission uses direct byte output rather than
passing the separator through `printf` as a format string.

**Success:** a user-written awk-style program parses, lowers, compiles, and
produces correct output on standard awk test cases.

---

## Phase 3: Binary data structures and DCG reader

**Goal:** replace text records with binary structures; add DCG-based readers.

1. Binary record ABI: struct layout, alignment, (de)serialisation convention.
2. DCG grammar for parsing binary records into terms.
3. Compile the DCG through UnifyWeaver to LLVM.
4. Extend `item_field/3` and `select_writer/2` for `record(binary, Type, Payload)`.
5. Map `State` stream handles to OS file descriptors.

**Associative-array note:** typed/native associative arrays are a high-value
later feature. AWK associative arrays are string keyed; PLAWK can eventually
lower common table shapes to binary hash tables keyed by typed values or
interned IDs. That should preserve awk-like ergonomics while creating a
plausible performance win over string-centric AWK loops once the basic compiled
reader/handler/output path is stable.

> **Perf caveat:** DCGs over difference-lists of bytes are correct but can be
> slow in WAM without first-argument indexing on the byte / partial evaluation.
> Treat as a correctness path first; specialise before claiming "no text in the
> hot path."

**Success:** a binary that reads a custom binary format, processes it with
pattern-action rules, and writes binary output — no text serialisation in the
hot path.

---

## Phase 4: File descriptors and bash compatibility

**Goal:** multi-stream support and bash-like subprocess/redirection.

1. Extend `State` to hold a map of named stream handles.
2. **Wire to existing builtins** rather than rebuild: `shell/1,2`, `mkfifo/2`,
   `kill/2`, and the filesystem surface already in `wam_llvm_target.pl`
   (Specification §10). Reuse `glue/{shell,pipe,native,streaming}_glue.pl`.
3. `spawn_process/5` for subprocess creation with connected pipes.
4. DSL redirection syntax (`>`, `<`, `2>`, `|`) desugaring to the primitives.
5. stdin/stdout isolation: spawned subprocesses must not inherit the DSL's
   primary I/O unless explicitly redirected.

**Success:** a DSL program orchestrating multiple streams (incl. subprocesses)
while the main handler loop processes its own primary input independently.

---

## Phase 5: JIT grammar and interactive query

**Goal:** runtime grammar extension via DCGs and clause-database querying.

1. Expose a `read_term/3`-equivalent DSL builtin — i.e. **add LLVM to the
   runtime-parser capability table** (`wam_runtime_parser_capability.pl`) in
   `compiled` mode; default stays `off`/`native` so non-JIT binaries pay nothing
   (Specification §11, `docs/WAM_RUNTIME_PARSER_STATUS.md`).
2. Load new DCG grammar rules at runtime (read via the parser, compiled via
   WAM/LLVM JIT).
3. Assert/retract clauses from within a running DSL program.
4. REPL mode: read queries from a stream, evaluate, return results.
5. If a `:- det` pass was added (Finding 2, path b), enforce declared modes/det
   on dynamically loaded predicates before acceptance.

**Success:** a running DSL binary accepts new grammar rules at runtime, compiles
them to native code, and uses them in subsequent iterations of the main loop.

---

## Appendix: compilation pipeline

```
awk-like source
      │  [Phase 2: parser]
      ▼
AST: program(Begin, Rules, End)
      │  [Phase 2: AST → Prolog core codegen]
      ▼
Prolog core: handler/4, reader/4, ...  + :- mode declarations
      │  [UnifyWeaver: Prolog → Hybrid WAM IR]   (modes drive binding analysis)
      ▼
WAM IR  (deterministic builtins selected via binding state)
      │  [UnifyWeaver: WAM IR → LLVM IR]   (state/N → aggregate; tail call → musttail)
      ▼
LLVM IR
      │  [llc + clang]
      ▼
Native executable: reads streams, processes records, writes output
```
