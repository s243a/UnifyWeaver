# PLAWK WAM/LLVM Compile Probes

This directory contains small compile probes for the PLAWK Phase 0 core.

Run from the repository root:

```bash
swipl -q -s examples/plawk/probes/generate_core_probe.pl -t halt
```

The first probe intentionally targets the non-reader helper predicates before
the stream loop. `text_file_reader/5` is excluded because the Phase 0 reader is a
slurp shortcut built on SWI file I/O; the design already expects streaming input
to become a WAM/LLVM runtime gap.

Additional probe:

```bash
swipl -q -s examples/plawk/probes/generate_loop_probe.pl -t halt
```

## Current results

### Core helper probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_core_probe.pl -t halt
```

Result: succeeds and writes `examples/plawk/generated/plawk_core_probe.ll`. All
listed PLAWK helper predicates currently compile through the WAM fallback path.
Private helper callees must be listed explicitly; omitting `normalize_outputs/2`
caused unresolved-label warnings in the first run.
Verifier status: `llvm-as examples/plawk/generated/plawk_core_probe.ll -o /dev/null` now passes. The previous `%Instruction` array mismatch was caused by `switch_on_constant_fallthrough` being emitted as a `; TODO` comment inside `@module_code`; WAM/LLVM now lowers that instruction to a real no-op fallthrough literal.

### Loop probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_loop_probe.pl -t halt
```

Result: emits `examples/plawk/generated/plawk_loop_probe.ll` without unresolved-label warnings. WAM/LLVM now recognizes `call/N` bytecode targets and routes them through a generated numeric meta-call dispatch table keyed by `(atom_id, effective_arity) -> label_index`. The generated loop probe verifies with `llvm-as`.

Remaining scope: the dispatch table currently handles atom goals such as `call(foo, A, B)`. Compound closures such as `call(foo(X), A)` still need the same table extended to compiled functor identity plus base arguments.

### Reader probe

Not attempted yet. The Phase 0 reader uses `read_file_to_string/3`, which is an
SWI slurp shortcut. The implementation plan already identifies streaming input
as a runtime gap for the WAM/LLVM target.
