<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 John William Creighton (@s243a)
-->

# The plawk eval arc — a top-level architecture summary

Everything in one place: how a plawk binary compiles Prolog **source
text** at runtime, loads the result, and calls it — and how the pieces
that make that possible stack on each other. This is the map; the
detailed design docs are linked from each layer.

The one-line demo the whole arc exists for:

```awk
{ total += dyncall_at(compile("[(sq(X, R) :- atom_number(X, N), R is N * N)]"), $1) }
END { print total }
```

A compiled, stand-alone native binary reads that grammar as **text**,
compiles it to bytecode **inside the running process** (through a
self-hosted compiler that ships alongside the binary), loads it, and
calls it once per record — compiling exactly once thanks to
content-based deduplication. Editing a `compile_file(...)` source
changes the next run's behaviour with **no rebuild**.

## The layer stack

Each layer is useful on its own and each was shipped (and is tested)
independently. Later layers only *compose* earlier ones.

```
 6  eval surface        compile(src) / compile_file(path) → handle → dyncall_at
 5  self-hosted compiler  cgfull in the loadable subset, shipped as <bin>.evalc.wamo
 4  return marshalling  i64 / double / blob / record / assoc — one call primitive each
 3  plawk call surfaces dyncall, dyncall@name, dyncall_at + typed wrappers
 2  loader + object VM  .wamo text format, @wam_object_load, shared step/run_loop
 1  AOT target          WAM→LLVM compiler, the %Instruction stream, the runtime IR
```

### Layer 1 — the AOT target (the ground everything stands on)

`src/unifyweaver/targets/wam_llvm_target.pl` compiles Prolog predicates
through a WAM instruction stream into LLVM IR, which clang links into a
native binary. Two properties of this layer carry the whole arc:

- **One instruction representation.** Every WAM instruction is a
  `%Instruction {tag, op1, op2}` row; execution is a `step` dispatch
  inside `run_loop`. Anything that can *produce* that array — the AOT
  compiler or a runtime loader — executes on the same engine with the
  same semantics.
- **The runtime IR is a library.** Atom registry, assoc tables, line
  readers, arithmetic, the dynamic clause store: all are plain LLVM
  functions the compiled code calls. New capabilities (a loader, new
  call primitives) are additions to this library, not changes to
  compiled programs.

See [PLAWK_EXECUTION_ARCHITECTURE.md](./PLAWK_EXECUTION_ARCHITECTURE.md)
for the tier picture (plawk rules are fully native; transpiled Prolog is
bytecode on the AOT-compiled interpreter).

### Layer 2 — the `.wamo` object format and loader

`write_wam_object/3` serializes compiled predicates into `.wamo` — a
**text** format (version 2): instruction rows plus trailing sections for
relocations (atoms, floats, functors), the meta-call table, and — when
non-empty — the `switch_on_constant` dispatch tables.
`@wam_object_load` (and `@wam_object_load_bytes` for in-memory buffers)
parses that into a fresh `%WamState` whose instruction array feeds the
**same** `step`/`run_loop` as AOT code. Loaded objects get real indexed
dispatch: the loader rebuilds the `%SwitchEntry` arrays the AOT
dispatcher uses, so a 200-clause fact table probes ~7.6× faster than the
try-chain fallback.

Text format matters twice: it made the loader small, and it made
**emitting** an object mere string assembly — which is what lets a
compiler written in the loadable subset exist at all (layer 5).

### Layer 3 — the plawk call surfaces

plawk programs reach loaded objects through three call forms, all
riding pay-for-what-you-use IR (a program with no dynamic calls emits
none of this):

- `dyncall(args)` — the fixed `BEGIN { DYNLOAD = "lib.wamo" }` object's
  default entry, loaded lazily once per run.
- `dyncall@name(args)` — a named entry of a multi-entry object; the
  entry's PC resolves once through a shared per-entry resolver and is
  cached.
- `dyncall_at(source, args)` — a **runtime-chosen** source, with a
  cache registry (`DYNCACHE = "on" | "mtime" | "off"`) mapping sources
  to loaded VMs.

### Layer 4 — return marshalling (choosing the target type)

A grammar's output cell can materialize into any plawk container; each
shape is one `@wam_object_call_*` primitive that walks the result
**before** the arena rewind and one thin shim per call site kind:

| Return shape | Surface | Primitive |
|---|---|---|
| integer | `dyncall...` | `@wam_object_call_i64` |
| double | `float(dyncall...)` | `@wam_object_call_f64` |
| opaque bytes | `blob(dyncall...)` | `@wam_object_call_bytes` |
| typed record | `(a,b) = ... as (i64 f64)` / record view `as (...) { ... }` | `@wam_object_call_record` |
| assoc, i64 values | `arr = ... as assoc` | `@wam_object_call_assoc` |
| assoc, string values | `arr = ... as assoc(str)` | `@wam_object_call_assoc_str` |

Strings thread through everything by **registry id**: record string
fields and blob returns are `(ptr,len)` slices into the persistent atom
table; assoc atom keys and `(str)`-kind atom values are stored as their
global-registry ids — the same keyspace text-mode field slices intern
into — so for-in reports, literal lookups, and value prints all resolve
ids to text with the one `@wam_atom_to_string` helper.

### Layer 5 — the self-hosted compiler

The payoff needs a Prolog→`.wamo` compiler that *itself* runs as a
loaded object. `src/unifyweaver/targets/wam_bootstrap_compiler.pl` is
that compiler: a minimal (not the 22k-line host) tier-2 compiler written
inside the loadable subset — reader, register allocation, clause
grouping, try-chains, if-then-else, structure patterns, a builtin
whitelist, and the `.wamo` serializer.

The campaign that made the subset rich enough (clause indexing,
meta-call, the dynamic clause store, `catch/throw`, the term
reader/writer) is chronicled in
[PLAWK_EVAL_BOOTSTRAP.md](./PLAWK_EVAL_BOOTSTRAP.md); the staged
self-compile — culminating in the fixpoint `gen2 == gen3`
byte-identical, i.e. the compiler compiled by its own output is its own
output — in [PLAWK_SELFHOST.md](./PLAWK_SELFHOST.md). The campaign
surfaced and fixed **thirteen latent runtime bugs** (register-file
ceiling, choice-point width, `copy_term` aliasing, variable-identity
collapse, the monotonic-heap re-entry pair, the arith error channel,
...) — the self-compile was the most demanding client the runtime ever
had, which is much of its value.

### Layer 6 — the eval surface

`compile(src)` / `compile_file(path)` in the `dyncall_at` source
position tie it together:

- At **build** time, the CLI (`examples/plawk/bin/plawk`) detects
  compile sites and ships the bootstrap compiler as `<bin>.evalc.wamo`
  next to the binary (`BEGIN { EVALC = "..." }` points at an existing
  object instead). No compile sites → nothing ships.
- At **run** time, `@plawk_compile` interns the source text, dedups by
  content (same text → same handle, compiled once), and otherwise runs
  `@wam_object_eval`: load the compiler object (cached), call it on the
  source, load the `.wamo` bytes it emits, register the fresh VM in the
  `dyncall_at` cache, and return the registry **handle**.
  `@plawk_compile_file` reads the file and delegates — content dedup
  makes edit-and-rerun work with no mtime tracking.
- The handle flows into the existing `dyncall_at` shims as
  `(null path, handle)`; every downstream marshalling shape (layer 4)
  works on runtime-compiled grammars unchanged.

Because the cache registry carries the handle, `DYNCACHE = "off"`
plus a compile site is a **build error**, not a silent miscompile.

## Invariants the arc keeps

- **One engine.** Loaded code runs through the same `step`/`run_loop`
  as AOT code — semantics can't drift between the two.
- **Pay for what you use.** Loader, shims, resolvers, the compile
  support, the shipped compiler object: each is emitted only when the
  program has a site that needs it.
- **Constant memory per call.** Every call primitive saves the heap
  mark, marshals results out before the arena rewind, and rewinds —
  a per-record dyncall does not grow the process.
- **Strings are registry ids.** One intern keyspace across text-mode
  fields, blob keys, assoc atom keys, and `(str)` values; one resolver
  (`@wam_atom_to_string`) back to text.
- **Content over mtime.** The eval cache keys on source text, so
  "did it change" has one answer everywhere.
- **Fail loud.** Unsupported constructs throw at compile
  (`throw/1` diagnostics in the walkers); arith errors fail `is/2`
  rather than yielding 0; cache-off + compile is a build error.

## Where each design doc fits

| Doc | Layer(s) | What it holds |
|---|---|---|
| [PLAWK_EXECUTION_ARCHITECTURE.md](./PLAWK_EXECUTION_ARCHITECTURE.md) | 1 | the compiled/interpreted tier picture |
| [PLAWK_JIT_ROADMAP.md](./PLAWK_JIT_ROADMAP.md) | 2–4 | the capability-by-capability history + what's next |
| [PLAWK_DCG_BINARY_READERS.md](./PLAWK_DCG_BINARY_READERS.md) | 4 | grammar-driven readers meeting BINFMT |
| [PLAWK_EVAL_BOOTSTRAP.md](./PLAWK_EVAL_BOOTSTRAP.md) | 5–6 | the subset-expansion milestones + the landed surface |
| [PLAWK_DYNAMIC_DB.md](./PLAWK_DYNAMIC_DB.md) | 5 | the dynamic clause store (`assertz`/`retract`) |
| [PLAWK_SELFHOST.md](./PLAWK_SELFHOST.md) | 5 | the staged self-compile to the fixpoint |

## What remains open

Deliberately deferred, in rough value order:

- ~~**for-in over grammar-populated tables inside rule bodies**~~ —
  LANDED: `for (k in arr) print ...` runs per record inside the rule's
  action chain, over the table as accumulated so far (str-valued
  tables print text there too).
- ~~**`dyncall_at@name(...)`**~~ — LANDED (all three return kinds:
  i64, `float(...)`, `blob(...)`): named entries on runtime sources
  and `compile(...)` handles, resolved per call against the VM's
  materialized entry table. Compiled handles expose their whole
  predicate family: the CLI ships `cgfullm` (cgfull's core + a
  multi-entry name table; byte-identical on single-predicate sources,
  so `cgfull` stays the self-host oracle).
- ~~**Surface B (`DYNENTRY`)**~~ — LANDED as parse-time sugar:
  `DYNENTRY parse, score` reserves the names and rewrites bare calls
  to the named-entry nodes (startup-cached PC); shadowing a userspace
  or builtin name is a parse error.
- **Handle-in-scalar** — storing a compile handle in a plawk variable
  across records (today the dedup makes the nested form equivalent).
- **Further compiler-subset growth** — the bootstrap compiler covers
  the grammar shapes the eval surface targets; widening it is
  demand-driven.
