# PR Title

Add Lua WAM resolved dispatch parity

# PR Description

## Summary

- Add Lua WAM direct-PC instruction variants for calls, executes, jumps, choicepoint retries, and indexed switch dispatch.
- Add `Runtime.resolve_program/1` so generated Lua programs rewrite known label-based instructions into PC-based instructions at load time.
- Keep unresolved labels on the existing fallback path while matching the Rust/Haskell resolved-dispatch shape for known labels.
- Add Lua generator tests that verify generated programs invoke the resolver and loaded programs contain resolved ops such as `CallPc`, `TryMeElsePc`, and `SwitchOnConstantA2Pc`.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This closes the Lua resolved-dispatch parity gap against Rust/Haskell for known in-program labels. Remaining Lua parity gaps include fact-stream/indexed fact dispatch, Haskell-style parallel aggregate execution, and broader lowered-emitter coverage.
