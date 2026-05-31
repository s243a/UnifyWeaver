# WAM Representation Modes

This document names the representation choices between Prolog clause
compilation and target code generation. These modes describe what representation
is produced and consumed; they do not by themselves choose whether a target uses
an interpreter loop, lowered functions, or mixed emission.

## Modes

| Mode | Pipeline | Notes |
|---|---|---|
| `wam_text` | Prolog clauses -> WAM text -> target path | Historical default for most WAM targets. |
| `wam_items_bridge` | Prolog clauses -> WAM text -> shared parser -> WAM items -> target path | Low-risk migration bridge; still pays the text round trip. |
| `wam_items_native` | Prolog clauses -> WAM items -> target path | Skips WAM text, but still executes WAM represented in the target language. |
| `direct_target` | Prolog clauses -> target-native code | Alias for the existing non-WAM target path where a target supports it. |

`wam_items_native` does not skip WAM. It skips only the serialized WAM-text
form. The target still receives a WAM instruction stream, usually materialized
as target-language data and executed by that target's WAM runtime. This can be
more optimizable than idiomatic target code because the instruction vocabulary is
compact, regular, and easy to rewrite or partially lower through kernels/FFI.

`direct_target` is the name for the existing non-WAM generation path. It is a
separate backend strategy, not a WAM representation mode in the strict sense.

## Resolution

Targets should resolve representation policy through
`wam_ir_mode:wam_ir_mode/4`:

```prolog
wam_ir_mode(Target, EmitMode, Options, Mode).
```

An explicit `wam_ir(Mode)` option overrides the target default. Without an
override, defaults are per target and per emit mode. Current defaults are
conservative:

- Python interpreter mode: `wam_items_bridge`
- Python lowered mode: `wam_text`
- Lua/R interpreter mode: `wam_items_bridge` as the intended migration default
- Unknown WAM targets: `wam_text`

WAM-specific targets may reject `direct_target`; callers should use the ordinary
non-WAM target when they want direct target-native code.
