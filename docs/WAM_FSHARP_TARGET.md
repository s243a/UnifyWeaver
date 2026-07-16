# WAM F# Target -- Usage Guide

The WAM F# target compiles Prolog predicates to a standalone .NET
project (`.fsproj` + a small F# module set) that runs on the .NET 8
runtime via `dotnet run`.  It implements a hybrid WAM: predicates are
compiled into an instruction array that a small F# interpreter
(`step` + `run`) executes, with an optional "lowered" emit mode that
also emits each predicate as a direct F# function bypassing the
instruction-array dispatch.

This is **distinct from** the older [FSHARP_TARGET](FSHARP_TARGET.md),
which compiles Prolog directly to idiomatic functional F# (records,
`let rec`, pattern matching) without a WAM.  Use FSHARP_TARGET when
you want readable F# that hand-translates the Prolog; use WAM F# when
you want full WAM semantics (backtracking, unification with shared
variables, list-spine pattern matching) and don't mind the
interpreter overhead.

Living status (shipped / gaps / path forward):
[WAM_FSHARP_STATUS.md](WAM_FSHARP_STATUS.md). Parity vs Haskell/Rust:
[design/WAM_FSHARP_PARITY_AUDIT.md](design/WAM_FSHARP_PARITY_AUDIT.md).

The target lives in:

- [src/unifyweaver/targets/wam_fsharp_target.pl](../src/unifyweaver/targets/wam_fsharp_target.pl)
  -- codegen, runtime emitter, project writer
- [src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl](../src/unifyweaver/targets/wam_fsharp_lowered_emitter.pl)
  -- the optional `emit_mode(functions)` lowering pass
- [src/unifyweaver/bindings/fsharp_wam_bindings.pl](../src/unifyweaver/bindings/fsharp_wam_bindings.pl)
  -- the shared F# helper module emitted into every project
  (`Value`, `WamState`, `derefVar`, `putReg`, `addToBuilder`, ...)

## Quick start

```prolog
:- use_module('src/unifyweaver/targets/wam_fsharp_target',
              [write_wam_fsharp_project/3]).

:- dynamic user:greet/1.
user:greet(world).
user:greet(fsharp).

:- initialization((
    write_wam_fsharp_project(
        [user:greet/1],
        [ module_name('greet_demo'),
          no_kernels(true)
        ],
        '/tmp/greet_demo'),
    halt
)).
```

After running this script:

```bash
cd /tmp/greet_demo
dotnet build --nologo -v quiet
dotnet run --no-build --nologo
```

The default `Program.fs` produced by `write_wam_fsharp_project/3` is
a minimal stub.  Real driver code lives in your own `Program.fs` --
overwrite it after generation (every smoke test in
`tests/core/test_wam_fsharp_*_smoke.pl` does this) using the shape:

```fsharp
module Program

open WamTypes
open WamRuntime
open Predicates
// open Lowered  // only if emit_mode is functions/mixed

let mkContext () =
    let resolvedCode =
        resolveCallInstrs allLabels [] (Array.toList allCode)
        |> List.toArray
    { WcCode              = resolvedCode
      WcLabels            = allLabels
      WcForeignFacts      = Map.empty
      WcFfiFacts          = Map.empty
      WcFfiWeightedFacts  = Map.empty
      WcAtomIntern        = Map.empty
      WcAtomDeintern      = Map.empty
      WcForeignConfig     = Map.empty
      WcLoweredPredicates = Map.empty  // or `loweredPredicates` from Lowered
      WcCancellationToken = None }

let mkState () : WamState =
    { WsPC = 0; WsRegs = Array.create MaxRegs (Unbound -1)
      WsStack = []; WsHeap = []; WsHeapLen = 0
      WsTrail = []; WsTrailLen = 0; WsCP = 0; WsCPs = []; WsCPsLen = 0
      WsBindings = Map.empty; WsCutBar = 0; WsVarCounter = 0
      WsBuilder = None; WsBuilderStack = []; WsAggAccum = []
      WsB0Stack = []; WsCatchers = [] }

[<EntryPoint>]
let main _argv =
    let ctx = mkContext ()
    let s = mkState ()
    match dispatchCall ctx "greet/1" s with
    | Some s1 ->
        match run ctx s1 with
        | Some _ -> printfn "succeeded"; 0
        | None   -> printfn "failed"; 1
    | None -> printfn "predicate not found"; 1
```

## API

### `write_wam_fsharp_project(+Predicates, +Options, +ProjectDir)`

Generates an `.fsproj` plus every F# module needed to compile and run
the predicates as a self-contained .NET project.

`Predicates` is a list of module-qualified indicators (`user:Name/Arity`
or `Module:Name/Arity`).

`Options` is a list; the ones handled directly by the F# target are:

| option | default | meaning |
| --- | --- | --- |
| `module_name(Name)` | `'wam-fsharp-bench'` | `<AssemblyName>` written into the `.fsproj`. |
| `no_kernels(true)` | `false` | Skip recursive-kernel detection.  Required for predicates with no real recursion; the kernel detector can otherwise spend time and emit unused code. |
| `emit_mode(Mode)` | `interpreter` | One of `interpreter`, `functions`, `mixed([Pred/Arity, ...])`.  See [Emit modes](#emit-modes). |
| `runtime_parser(Mode)` | `none` | One of `none`, `compiled(prolog_term_parser)`.  See [Runtime parser](#runtime-parser). |
| `base_pc(N)` | computed | Override the starting PC of the first emitted predicate.  Mostly for testing -- normal use leaves this unset. |
| `lmdb_path(Path)` | (none) | When set, includes `LmdbFactSource.fs` + LightningDB NuGet in the generated project. The module provides `openEnv`, `loadCategoryParent`, `LmdbCursorLookup`, `TwoLevelCachedLookupSource`, etc. |
| `lmdb_materialisation(Mode)` | `cached` | One of `eager`, `lazy`, `cached`. Controls how LMDB facts are accessed at runtime. `cached` (two-level L1/L2) is the recommended default — zero startup cost, sub-ms warm hits, bounded memory. See [LMDB modes](#lmdb-modes). |
| `lmdb_l2_capacity(Spec)` | `auto` | L2 cache sizing. `auto` = runtime memory formula; `small`/`medium`/`large` = T-shirt sizes (8/80/800 MB); `enwiki`/`simplewiki` = corpus presets; `'80mb'` = explicit byte budget; integer = raw entry count; `unlimited` = no cap. |
| `csr_path(Path)` | (none) | When set, includes `CsrReader.fs` in the generated project. The module provides `CsrLookupSource` implementing `ILookupSource` for reverse child-edge lookup from CSR artifacts (`.csr.idx` / `.csr.val` / `.csr.meta`). No extra NuGet dependencies. |

Beyond these, the F# target threads the standard cross-target options
(kernel detection, mode hints, etc.) through to `wam_target` and the
helper modules.

### `compile_wam_predicate_to_fsharp(+Pred/Arity, +WamCode, +Options, -FSharpCode)`

Compile one predicate to its F# source.  Used internally by
`write_wam_fsharp_project/3`; exposed for testing.  `WamCode` is the
WAM-textual output of `wam_target:compile_predicate_to_wam/3`.

### `compile_wam_runtime_to_fsharp(+Options, +DetectedKernels, -Code)`

Emit the `WamRuntime.fs` module.  Wraps the template in
`fsharp_wam_bindings.pl` and stitches the `step` function's instruction
branches together.  Returns the full module source as a string.

### `wam_fsharp_resolve_emit_mode(+Options, -Mode)`

Compute the effective emit mode for a project, applying the selector
hierarchy:

1. `option(emit_mode(M), Options)`
2. `user:wam_fsharp_emit_mode(M)` dynamic fact (multifile)
3. `interpreter` (the safe default; see [Emit modes](#emit-modes) for
   why this is currently the default even after lowering was made
   functionally correct)

### `wam_fsharp_partition_predicates(+Mode, +Predicates, +DetectedKernels, -Interpreted, -Lowered)`

Given an emit mode and a list of predicates, split into two sets: the
ones that go through the interpreter and the ones the lowered emitter
will handle.  Used both internally and as a way for callers to
inspect what would happen without actually generating code.

## Architecture

The generated project has four F# modules in build order:

```
WamTypes.fs       <-- DUs + records (Value, Instruction, WamState, ChoicePoint, ...)
WamRuntime.fs     <-- step, run, dispatchCall, backtrack, resumeBuiltin, helpers
Predicates.fs     <-- per-predicate Instruction lists + allCode + allLabels
Lowered.fs        <-- only emitted when emit_mode != interpreter
Program.fs        <-- a minimal stub; usually overwritten by the caller
```

`Lowered.fs` is omitted entirely when `emit_mode(interpreter)`.

### Value, the universal term type

```fsharp
type Value =
    | Atom of string
    | Integer of int
    | Float of float
    | Str of string * Value list      // compound: functor + args
    | VList of Value list             // proper list (ground tail)
    | Unbound of int                  // variable id
    | Ref of int                      // heap reference (rarely used)
    | VSet of Set<Value>              // for visited-set built-ins
```

**Two list encodings, equivalent at the Value level.**  A list of three
elements can materialise either as `VList [a; b; c]` (the
ground-tail-known form) or as `Str ("[|]", [a; Str ("[|]", [b; Str ("[|]", [c; Atom "[]"])])])`
(cons cells -- the form `addToBuilder` produces when the tail is unbound
during build, because the tail must stay symbolic).  Every builtin
that walks a list must accept both encodings.  The `flattenList`
helper in `wam_fsharp_target.pl` (emitted into the runtime) does this
once for all list-consuming builtins (`append/3`, `length/2`,
`reverse/2`, `nth0/3`, `member/2`, ...).

**Empty list equivalence.**  `VList []` and `Atom "[]"` both denote
Prolog's empty list.  Both shapes appear because two materialisation
paths produce them: `Atom` when the WAM emits a literal `[]` constant,
`VList []` when an `addToBuilder` run collapses a `[H|[]]` to a
singleton.  `unifyTerms` and `termEqual` both normalise across these
two shapes.  `GetConstant Atom "[]"` succeeds against either.

### WamState

The interpreter's per-call mutable-by-step state.  All updates flow
through `step`'s `WamState option` return; the run loop in `run`
threads the new state into the next iteration.

```fsharp
type WamState =
    { WsPC          : int
      WsRegs        : Value array          // X / Y / A registers
      WsStack       : EnvFrame list        // env frames (one per Allocate)
      WsHeap        : Value list
      WsHeapLen     : int
      WsTrail       : TrailEntry list      // for unbinding on backtrack
      WsTrailLen    : int
      WsCP          : int                  // current continuation PC
      WsCPs         : ChoicePoint list     // stack of choice points
      WsCPsLen      : int
      WsBindings    : Map<int, Value>      // immutable variable env
      WsCutBar      : int                  // for B0 cut protocol
      WsVarCounter  : int                  // next fresh Unbound id
      WsBuilder     : BuilderState option  // active build context
      WsBuilderStack: BuilderState list    // pushed builds (nested structs)
      WsAggAccum    : Value list           // findall / bagof accumulator
      WsB0Stack     : int list             // cut-barrier nesting
    }
```

**Single-owner state semantics.**  Each `step` returns the new state,
the run loop uses it, and the previous reference is dead by the time
the next instruction's `step` fires.  This is what lets `putReg`
mutate `WsRegs` in place (and saves the ~30% of CPU the per-write
`Array.copy` used to cost; see #2428).  Choice points snapshot
`WsRegs` via `Array.copy` at creation (`TryMeElse`, `BeginAggregate`,
`FactRetry`, `SelectRetry`, `MemberRetry`), and `backtrack` does the
same on restore so a CP's saved registers can never be mutated by a
later in-place write.

### Dispatch order

`dispatchCall` resolves a predicate-by-name call in three layers:

1. `WcLoweredPredicates` map.  If a lowered function exists for the
   key, call it directly (the lowered F# function manages its own
   continuation, returns a `WamState option`).
2. `callIndexedFact2`.  Fast path for 2-arg fact tables with a
   constant first arg -- avoids running through the interpreter for
   the common Prolog "facts" pattern.
3. `WcLabels` map -> `run ctx { sc with WsPC = pc; WsCP = 0 }`.
   General fallback.

The `WsCP = 0` on entry to `run` is load-bearing.  If a lowered F#
caller passes `sc.WsCP = <caller's post-call PC>`, the callee's
`Proceed` sets `WsPC = WsCP`, and `run` then keeps executing the
**caller's** WAM continuation in the interpreter alongside the
caller's F# continuation -- the bug fixed by #2423.

### Choice points and retry kinds

`ChoicePoint.CpBuiltin : BuiltinState option` is the polymorphic slot
for retry semantics other than the standard try/retry/trust chain:

| variant | used by |
| --- | --- |
| `FactRetry` | `callIndexedFact2` (alternate matching facts) |
| `HopsRetry` | distance-aware path searches |
| `SelectRetry` | `select/3` enumeration |
| `MemberRetry` | `member/2` non-deterministic backtracking (#2415) |
| `FFIStreamRetry` | FFI-backed multi-result streams |

`MemberRetry` was added in #2415 to fix the parser's `resolve_prefix`
chain: `member(op(Name, Prec, Type), OpTable), is_prefix_type(Type), !`
needs `member` to be backtrackable through the cut, because operators
like `:-` have both `xfx 1200` and `fx 1200` entries in the op table
and the prefix path has to skip past the xfx one.

## Emit modes

```
interpreter   default; every predicate runs via the instruction-array dispatch.
functions     every predicate attempts lowering to a direct F# function;
              falls back to the interpreter for the ones the lowerable check
              rejects (anything with BeginAggregate, anything multi-clause
              with non-trivial indexing, ...).
mixed([P/N])  only the named predicates attempt lowering; everything else
              uses the interpreter.
```

`emit_mode(functions)` does work -- it generates correct code for
every smoke test in the repository.  But the relative wall-clock gain
over `emit_mode(interpreter)` is small (see
`tests/core/test_wam_fsharp_lowered_bench.pl`):

| workload | iter | interpreter | functions | speedup |
| --- | ---: | ---: | ---: | ---: |
| parser-heavy (`read_term_from_atom`) | 10k | ~22 s | ~22 s | ~1.0x |
| fully-lowered (`X = foo(...), X == foo(...)`) | 200k | ~2.1 s | ~1.9 s | ~1.07x |

The CLR JIT optimises the `step` function's `match instr with`
exceptionally well, so the dispatch was essentially free before
inlining.  The big-ticket performance wins are in the runtime
helpers themselves -- `putReg` (#2428), `unifyTerms`, the `WsBindings`
Map -- and benefit both emit modes equally.

**Recommendation: leave `emit_mode(interpreter)` as the default.**
Switch to `functions` or `mixed` only if (a) you've profiled and
found a specific predicate that benefits, or (b) you need the
lowered code for readability / debugging (the lowered F# is more
direct than the WAM-textual `Instruction list`).

## Runtime parser

```
runtime_parser(none)                         (default)
runtime_parser(compiled(prolog_term_parser)) // bundles the parser library
```

The runtime parser is the WAM-compiled Prolog term parser (in
`src/unifyweaver/core/prolog_term_parser.pl`) -- it lets the
generated F# program call `read_term_from_atom/2`, `parse_term_from_codes/4`,
etc. at run time.

When `runtime_parser(compiled(prolog_term_parser))` is set, the
project automatically pulls in:

- `prolog_term_parser:tokenize/2`, `parse_expr/8`, `canonical_op_table/1`,
  ... (and every other helper in the library)
- the target-agnostic wrappers (`read_term_from_atom/2,3`,
  `parse_term_from_atom/3,4`, ...) from
  `src/unifyweaver/core/cpp_runtime_parser_wrappers.pl`

The combined predicate list is sorted and deduplicated before
codegen, so user predicates that happen to name themselves the same
as a parser helper are merged (the user's clauses win in source
order).

When `runtime_parser(none)` is set (the default), any user predicate
whose body has a statically visible parser-dependent call (like
`read_term_from_atom`) is **rejected at codegen time** rather than
producing a runtime that silently lacks the parser.  See
`src/unifyweaver/targets/wam_runtime_parser_capability.pl` for the
rejection list.

## Key runtime invariants

A non-obvious set of behaviours that took several PRs to settle.
Worth knowing before reading the runtime code or extending it.

### Cut-barrier (B0) protocol

The cut barrier (`WsCutBar`) is set by `Allocate` to the current
`WsCPsLen` *before* the callee's leading `TryMeElse` pushes CP_self.
This is essential -- the previous bug (#2407) had `Allocate` set the
barrier *after* TryMeElse, so a `!` inside the callee dropped CP_self
along with everything beneath it, leaving subsequent retries with no
CP to backtrack to.

### `member/2` is non-deterministic via `MemberRetry`

Looks like a one-shot find-first.  It isn't.  The parser's
`resolve_prefix(Name, OpTable, Prec, Type) :- member(op(Name, Prec, Type), OpTable), is_prefix_type(Type), !.`
depends on backtracking into member when the type-guard fails.  The
runtime's `member/2` builtin pushes a `MemberRetry` CP on each
successful unify; backtrack resumes with the remaining tail.  See
#2415 for the full reasoning.

### `==/2` and the two list encodings

`==/2` uses a `termEqual` helper that derefs and normalises
`VList <-> Str("[|]", _)` cons-cell shapes, *not* raw F# structural
equality on the `Value` DU.  Without this, an `append`-built list
compared not-equal to the literal `[1,2,3]` that appears in the
caller, because the two materialise through different paths.
See #2419.

### `findall/3` requires `BeginAggregate` to seed its result reg

`BeginAggregate` writes a fresh `Unbound` var to its result register
*before* snapshotting the CP.  Without this, `finalizeAggregate`'s
`getReg resReg` returns `None` (the Y-reg sentinel return) and the
aggregated result is silently dropped.  See #2419.

Importantly, this seeding does **not** add a `vid -> Unbound vid`
entry to the bindings map -- that would loop `derefVar`.  Absence
from the map is the truly-unbound representation.

### `dispatchCall` resets `WsCP` when entering the interpreter

When a lowered F# function calls `dispatchCall` for a non-lowered
predicate, `dispatchCall` saves `sc.WsCP`, sets it to 0 before
calling `run`, and restores on return.  Without this, the called
predicate's outermost `Proceed` would propagate the caller's intended
F# return PC into `WsPC`, and `run` would continue executing the
caller's WAM body alongside the caller's F# continuation -- the
double-execution bug fixed by #2423.

### `putReg` mutates `WsRegs` in place

The 512-entry `WsRegs` array is mutated in place; the `WamState`
record's array reference is unchanged.  Safe because of single-owner
state semantics.  Choice points snapshot via explicit `Array.copy`
at creation and at `backtrack` restore, so any in-place write after
restore can't bleed back into a CP's saved registers.  See #2428 --
this was a ~2-3x speedup on every workload.

### The `undoBinding` trail-undo fold in `backtrack` is load-bearing

The `List.fold undoBinding cp.CpBindings (List.take diff |> List.rev)`
in `backtrack` *looks* like dead code -- `cp.CpBindings` is already
the immutable snapshot of `WsBindings` at CP creation, F# `Map` is
immutable, so a fold that just rebuilds an equivalent map should be
a no-op.  An experiment to remove it (see the conversation around
PR #2428 / #2429) caused a ~34% regression on parser_heavy.  Don't
remove it without re-running `tests/core/test_wam_fsharp_lowered_bench.pl`
and confirming.  The most likely explanation is JIT inlining or `Map`
structural-sharing effects that the F# / CLR optimiser cares about.

## Testing

| test | purpose |
| --- | --- |
| [`tests/core/test_wam_fsharp_parser_smoke.pl`](../tests/core/test_wam_fsharp_parser_smoke.pl) | `read_term_from_atom` end-to-end, 42 inputs |
| [`tests/core/test_wam_fsharp_runtime_smoke.pl`](../tests/core/test_wam_fsharp_runtime_smoke.pl) | runtime builtins (member, append, findall, =.., functor, ...) |
| [`tests/core/test_wam_fsharp_lowered_smoke.pl`](../tests/core/test_wam_fsharp_lowered_smoke.pl) | runtime builtins under `emit_mode(functions)` |
| [`tests/core/test_wam_fsharp_lowered_parser_smoke.pl`](../tests/core/test_wam_fsharp_lowered_parser_smoke.pl) | parser under `emit_mode(functions)` -- regression net for #2423's `dispatchCall WsCP` fix |
| [`tests/core/test_wam_fsharp_dotnet_smoke.pl`](../tests/core/test_wam_fsharp_dotnet_smoke.pl) | minimal "does dotnet build + run a generated project" test |
| [`tests/test_wam_fsharp_target.pl`](../tests/test_wam_fsharp_target.pl) | unit tests for the lowerable check + the emit-mode partition logic |

All smoke tests require the .NET 8 SDK on `PATH`.  They generate a
project under `/tmp/uw_fsharp_*_repro`, build with `dotnet build`, and
run with `dotnet run`.  Run them under `LANG=C.UTF-8` -- the test
files contain em-dashes in their comments, and the default POSIX
locale rejects multibyte sequences when `write/2` flushes them out
to the generated Program.fs.

### Benchmark

[`tests/core/test_wam_fsharp_lowered_bench.pl`](../tests/core/test_wam_fsharp_lowered_bench.pl)
is a side-by-side timing harness for `emit_mode(interpreter)` vs.
`emit_mode(functions)`.  It generates two F# projects per workload,
Release-builds each, drives the inner runPredicate loop via
`System.Diagnostics.Stopwatch`, and reports the median of three timed
rounds after a tier-up warm-up + forced GC.  Use it to validate any
runtime perf claim -- this harness is what caught the #2428 putReg
win and the (would-be) #backtrack-undoBinding regression.

## Supported builtins

The runtime in `fsharp_wam_bindings.pl` + `wam_fsharp_target.pl`
defines step branches for:

**Control:** `true/0`, `fail/0`, `!/0` (cut), `\+/1` (negation),
`call/1+`, `throw/1`, `catch/3`.

**ISO error variants:** Each of `is/2`, the six arithmetic comparisons
(`>`, `<`, `>=`, `=<`, `=:=`, `=\\=`), and `succ/2` ships in three
forms: the default (rewritten per-predicate by the iso_errors config),
an explicit `_iso/2` (always throws structured `error(_, _)` on bad
input), and an explicit `_lax/2` (always silent-fails). Configure via
the `iso_errors_config(File)` / `iso_errors(Bool)` /
`iso_errors(PI, Bool)` options; `wam_fsharp_iso_audit/3` reports
what each call site resolves to. Cross-target ISO contract:
`design/WAM_ISO_ERRORS_CROSS_TARGET_STATUS.md`.

**Arithmetic:** `is/2` (full expression evaluator: `+`, `-`, `*`, `/`,
`//`, `mod`, `rem`, `**`, `abs`, `min`, `max`, `gcd`, `truncate`,
`round`, `ceiling`, `floor`, `sqrt`, `sin`, `cos`, `tan`, `pi`, `e`,
`>>`, `<<`, `/\`, `\/`, `xor`), `</2`, `>/2`, `=</2`, `>=/2`, `=:=/2`,
`=\=/2`.

**Term equality:** `=/2` (unify), `==/2` (standard-order equal, with
list-encoding normalisation), `\==/2`.

**Type checks:** `atom/1`, `atomic/1`, `compound/1`, `integer/1`,
`number/1`, `float/1`, `var/1`, `nonvar/1`.

**Lists:** `member/2` (non-deterministic via `MemberRetry`),
`memberchk/2`, `append/3`, `length/2`, `reverse/2`, `last/2`,
`nth0/3`, `nth1/3`, `delete/3`, `select/3` (non-deterministic via
`SelectRetry`), `sort/2`, `msort/2`, `compare/3`.

**Structural:** `functor/3` (both modes), `arg/3`, `=../2` (univ;
both modes), `copy_term/2`.

**Aggregates:** `findall/3`, `bagof/3`, `setof/3` (via
`BeginAggregate` / `EndAggregate` + `MergeStrategy` discriminated
union; results materialise on backtrack until the aggregate frame
is reached).

**Atom / string:** `atom_codes/2`, `atom_chars/2`, `atom_length/2`,
`atom_concat/3`, `atom_string/2`, `atom_number/2`, `number_codes/2`,
`number_chars/2`, `char_code/2`, `upcase_atom/2`, `downcase_atom/2`,
`sub_atom/5`, `string_codes/2`, `string_chars/2`.

**Database (limited):** `assert/1`, `assertz/1`, `asserta/1`,
`retract/1` -- these go through `WcLoweredPredicates` adjustments
rather than the WAM heap, so they work for facts but **not** for
clauses with bodies.  Persistent dynamic state across `dispatchCall`
invocations is not the WAM F# target's strong suit; if you need
dynamic-database semantics, use the Python WAM target.

**Sets:** `BuildEmptySet`, `SetInsert`, `NotMemberSet` (specialised
visited-set instructions used by graph kernels).

**Parser bridge:** with `runtime_parser(compiled(prolog_term_parser))`,
`read_term_from_atom/2,3`, `parse_term_from_atom/3,4`,
`parse_term_from_codes/3,4` become callable from generated code.

## LMDB modes

When `lmdb_path(Path)` is set, the generated project includes
`LmdbFactSource.fs` with three materialisation modes:

| Mode | Startup | Per-lookup | Best for |
| --- | --- | --- | --- |
| `eager` | Loads entire relation into Map/Dict | O(1) Map.tryFind or Dict.TryGetValue | Full-graph scans, small corpora |
| `lazy` | Zero (opens env only) | LMDB cursor per call (~0.1 ms) | Single queries, memory-constrained |
| `cached` (default) | Zero | L1 array hit (~0.001 ms warm) | Everything else — adapts to demand |

**Why `cached` is the default:** at large scale (enwiki, 10M edges),
eager materialisation costs ~140 seconds — time spent building an
in-memory structure for edges the query may never touch. Cached mode
pays only for edges actually visited, and subsequent visits hit the
L1 per-thread cache at near-zero cost. The advantage is *time savings
from skipping unnecessary work*, not memory pressure (even "large"
cache at 800 MB is modest for a server).

The `TwoLevelCachedLookupSource` implements cached mode:
- **L1**: per-thread fixed array (4096 slots), collision-overwrite,
  zero contention
- **L2**: shared `ConcurrentDictionary`, bounded by
  `lmdb_l2_capacity` (default: auto-sized from available RAM)

Kernel templates accept `int -> int list` lookup functions so they
work transparently with any mode -- no materialisation at kernel entry.

## CSR reverse-index support

When `csr_path(Path)` is set, the generated project includes
`CsrReader.fs` with `CsrLookupSource` for reverse child-edge lookup
(parent -> children) from CSR (Compressed Sparse Row) artifacts.

CSR artifacts are built by `examples/benchmark/build_reverse_csr_artifact.py`
from a Phase 1 LMDB `category_parent` database. The format is:

| File | Content |
| --- | --- |
| `.csr.idx` | `int32_le parent, uint64_le offset, uint32_le count` (16 bytes/record, sorted) |
| `.csr.val` | `int32_le child` (4 bytes each, packed) |
| `.csr.meta` | JSON manifest (`format: unifyweaver.reverse_csr.v1`) |

`CsrLookupSource` loads the index into memory and reads child slices
from the values file on demand. It implements `ILookupSource` and
composes with `TwoLevelCachedLookupSource` for repeated lookups:

```fsharp
use csr = CsrReader.openCsr "/path/to/csr/artifact"
let cachedCsr = TwoLevelCachedLookupSource(csr, l2CapacitySpec = "auto") :> ILookupSource
let children = cachedCsr.Lookup(parentId)
```

No extra NuGet dependencies required (uses `System.IO` and
`System.Text.Json` from the .NET 8 SDK).

## See also

- [FSHARP_TARGET](FSHARP_TARGET.md) -- the older direct-to-functional-F# target (not WAM-based)
- [WAM_R_TARGET](WAM_R_TARGET.md) -- analogous WAM target with mode analysis + lowered emitter
- [WAM_HASKELL_LOWERED_PHILOSOPHY](design/WAM_HASKELL_LOWERED_PHILOSOPHY.md) -- the design rationale for the lowered-emitter pattern (Haskell first, F# / Clojure follow)
- [WAM_PERF_CROSS_TARGET](WAM_PERF_CROSS_TARGET.md) -- why the F# perf wins are target-specific and don't backport
- [DOTNET_COMPILATION](DOTNET_COMPILATION.md) -- general .NET project layout shared with the C# target
