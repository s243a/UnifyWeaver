# PR Title

Add Lua WAM aggregate and switch parity

# PR Description

## Summary

- Add Lua WAM emission for `switch_on_constant_a2`, `begin_aggregate`, and `end_aggregate`.
- Implement Lua runtime support for real WAM call/proceed continuations, aggregate collection/finalization, and constant/structure switch dispatch.
- Add Lua generator/e2e tests covering `findall/3`, `aggregate_all(count/sum/min/max/set)`, and second-argument switch dispatch.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This closes the main Lua parity gap against the Rust and Haskell WAM targets for aggregate instructions and `switch_on_constant_a2`. Remaining parity gaps include resolved-PC instruction variants, fact-stream/indexed fact paths, Haskell-style parallel aggregate execution, and broader lowered-emitter coverage.
