# PR Title

Filter Lua WAM fact stream rows

# PR Description

## Summary

- Add bound-argument filtering to Lua WAM `CallFactStream`.
- Filter candidate rows against already-bound `A1` and/or `A2` before unification.
- Apply the same filtering path to inline facts and external fact-source rows.
- Preserve external `A1` indexing as the candidate source optimization, then filter by bound `A2`.
- Add regression assertions for mismatched bound-argument cases.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This aligns Lua fact-stream behavior with Haskell's `streamFactRows` filtering semantics while keeping the existing inline and external fact-source paths intact.
