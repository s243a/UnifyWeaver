# WAM Lua Parity Audit

This note records the Lua hybrid WAM runtime parity surface after the 2026 Lua parity pass. Rust and Haskell were used as the reference targets for builtin/runtime behavior.

## Verified Runtime Surface

The Lua WAM target now has end-to-end coverage in `tests/test_wam_lua_generator.pl` for:

| Area | Lua support | Reference parity |
| --- | --- | --- |
| Direct fact dispatch | `CallIndexedAtomFact2` | Rust/Haskell-style indexed fact paths |
| Fact streams | inline and external fact source streams | Rust/Haskell fact enumeration behavior |
| Aggregates | `findall/3`, `aggregate_all/3` count/sum/min/max/set | Haskell aggregate frame behavior |
| Structural builtins | `member/2`, `length/2` | Rust/Haskell builtin semantics |
| Type and comparison builtins | `atom/1`, `integer/1`, `float/1`, `number/1`, `compound/1`, `var/1`, `nonvar/1`, `is_list/1`, `==/2`, `=:=/2`, `=\=/2`, `>/2`, `</2`, `>=/2`, `=</2` | Rust/Haskell builtin sets |
| Term inspection | `functor/3`, `arg/3` | Rust/Haskell term builtin behavior |
| Univ | `=../2` compose and decompose modes | Rust/Haskell univ behavior |
| Copying | `copy_term/2` with fresh variables and preserved sharing | Rust/Haskell source-var to fresh-var map |
| Control | `true/0`, `fail/0`, `!/0`, `\+/1`, `CutIte` | Rust/Haskell control builtin behavior |
| IO | `write/1`, `display/1`, `nl/0` | Rust IO builtin behavior |

## Notable Semantics

- `copy_term/2` uses a dedicated fresh-copy walker. The older `Runtime.copy_term/3` remains a structural clone helper for aggregation and intentionally preserves unbound variable names.
- `\+/1` evaluates the goal in an isolated substate so bindings from the negated goal do not leak into the caller.
- `display/1` is supported by the Lua runtime. Current shared WAM compilation emits `write/1` and `nl/0` as builtins; `display/1` runtime behavior is covered directly in Lua tests without broadening shared builtin classification.
- `CutIte` removes only the top choice point for if-then-else soft cut, while `!/0` clears runtime choice points.

## Known Non-Gaps

- Rust mentions `append/3`, but its WAM runtime explicitly reports it as not implemented. It is not a Lua parity gap until a reference target implements the behavior.
- Runtime mutation predicates such as `assertz/1` and `retract/1` are not part of the current Lua parity baseline; they require a separate live-store design rather than a small builtin dispatch patch.
- `read/2` and richer IO are out of scope for this parity pass. The implemented IO parity is limited to the Rust runtime's `write/1`, `display/1`, and `nl/0` behavior.

## Verification Commands

Use these checks after touching Lua WAM runtime parity:

```sh
swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl
swipl -q -g run_tests -t halt tests/test_wam_target.pl
swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"
```
