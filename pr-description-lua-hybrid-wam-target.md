# Build Lua Hybrid WAM Target

## Summary

Adds a new `wam_lua` hybrid WAM target with Lua project generation, runtime templates, optional lowered predicate emission, registry integration, and smoke coverage.

## Changes

- Added `src/unifyweaver/targets/wam_lua_target.pl` for WAM-to-Lua project generation, instruction literal emission, atom interning, predicate wrappers, foreign handler wiring, and emit-mode selection.
- Added `src/unifyweaver/targets/wam_lua_lowered_emitter.pl` for deterministic and first-clause lowered Lua functions with fallback to the instruction-array runtime.
- Added Lua runtime/program templates under `templates/targets/lua_wam/`.
- Registered `wam_lua` in `src/unifyweaver/core/target_registry.pl`.
- Added `tests/test_wam_lua_generator.pl` covering exports, registry wiring, generated files, lowered mode, choice points, and Lua CLI execution when `lua` is available.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/core/test_target_registry.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`
