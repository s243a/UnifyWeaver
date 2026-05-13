# PR Title

Add Lua WAM external fact sources

# PR Description

## Summary

- Add `lua_fact_sources([source(P/A, file(Path))])` support for Lua WAM binary fact streams.
- Generate `fact_sources` metadata for configured external `P/2` fact predicates.
- Extend `CallFactStream` to fall back from inline facts to file-backed CSV/TSV fact sources.
- Add Lua runtime file loading, atom interning, row caching, and streaming through the existing fact-stream backtracking path.
- Add e2e coverage for direct and caller predicates backed by an external CSV file.

## Verification

- `swipl -q -g run_tests -t halt tests/test_wam_lua_generator.pl`
- `swipl -q -g run_tests -t halt tests/test_wam_target.pl`
- `swipl -q -g "use_module(src/unifyweaver/targets/wam_lua_target), halt"`

## Notes

This closes the first Lua external fact-source parity gap against Haskell/R-style file-backed fact streaming. The slice intentionally stays dependency-free and limited to binary CSV/TSV facts; LMDB or richer fact-source adapters can follow separately.
