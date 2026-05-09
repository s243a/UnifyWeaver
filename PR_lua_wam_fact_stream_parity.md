# PR Title

Add Lua WAM fact stream parity

# PR Description

## Summary

- Add Lua WAM `CallFactStream` instruction/runtime support for inline binary fact streams.
- Add generated-program `inline_facts` storage for simple binary ground fact-only predicates.
- Teach the Lua WAM generator to detect eligible `P/2` fact-only WAM and emit `CallFactStream` instead of full clause WAM.
- Stream inline fact rows at runtime, unifying `A1`/`A2` and backtracking over remaining rows.
- Add focused emission and e2e tests, including a normal caller predicate that returns through a fact-stream predicate.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This closes the first Lua parity gap against Haskell's `CallFactStream` path for in-memory binary facts. External `FactSource`/file/LMDB-backed fact streaming remains a larger follow-up.
