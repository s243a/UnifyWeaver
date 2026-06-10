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
swipl -q -s examples/plawk/probes/generate_meta_call_probe.pl -t halt
swipl -q -s examples/plawk/probes/generate_reader_probe.pl -t halt
```

## Current results

### Core helper probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_core_probe.pl -t halt
```

Result: succeeds and writes `examples/plawk/generated/plawk_core_probe.ll`. All
listed PLAWK helper predicates currently compile through the WAM fallback path.
Private same-module helper callees are now pulled into the WAM/LLVM project
closure automatically; `normalize_outputs/2` is intentionally omitted from the
probe roots and still resolves.
Verifier status: `llvm-as examples/plawk/generated/plawk_core_probe.ll -o /dev/null` now passes. The previous `%Instruction` array mismatch was caused by `switch_on_constant_fallthrough` being emitted as a `; TODO` comment inside `@module_code`; WAM/LLVM now lowers that instruction to a real no-op fallthrough literal.

### Loop probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_loop_probe.pl -t halt
```

Result: emits `examples/plawk/generated/plawk_loop_probe.ll` without unresolved-label warnings. WAM/LLVM now recognizes `call/N` bytecode targets and routes them through a generated numeric meta-call dispatch table. The generated loop probe verifies with `llvm-as`.

### Meta-call probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_meta_call_probe.pl -t halt
```

Result: emits `examples/plawk/generated/plawk_meta_call_probe.ll` and verifies
with `llvm-as`. The probe includes both atom-goal meta-calls such as
`call(meta_target, A, B, C)` and compound closure calls such as
`call(meta_target(A), B, C)`. Atom goals dispatch by `(atom_id, arity)`;
compound closures dispatch by `(compiled_functor_pointer, arity)` and copy the
closure's base arguments before the extra `call/N` arguments.

### Reader probe

Command:

```bash
swipl -q -s examples/plawk/probes/generate_reader_probe.pl -t halt
```

Result: emits `examples/plawk/generated/plawk_reader_probe.ll` and verifies
with `llvm-as`. This probe uses the hybrid LLVM WAM builtins
`stream_open/2`, `read_line/2`, and `stream_close/1`. The builtins are general
runtime primitives, not PLAWK-specific code: handles are malloc-backed buffered
readers, `read_line/2` unifies `end_of_file` at EOF, and line atoms exclude the
record newline.
