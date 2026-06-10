# Haskell Hybrid WAM Feature Parity — Session Summary

## Branch: claude/haskell-hybrid-wam-features-Qih27

## PRs shipped (17 total)

| PR | Feature |
|----|---------|
| #2509 | Lowered emitter: inline optimizations (get_constant, get_value, put_structure, put_list, set_*, deallocate, cut) + Phase I instructions (PutStructureDyn, Arg, NotMemberList, etc.) |
| #2510 | ISO error handling: WamException, throw/1, catch/3, is_iso/2, 6 ISO comparison variants, succ variants, isIsoMetaBuiltin routing |
| #2512 | WAM instructions: GetStructure, GetList, UnifyVariable, UnifyValue, UnifyConstant + ReadArgs builder + pushBuilderIfActive |
| #2514 | Bidirectional ancestor kernel template (A*-pruned frontier search) |
| #2517 | CSR reverse-index reader template + codegen wiring (csr_path option) |
| #2518 | Bidirectional kernel upgrade + CSR auto-resolution |
| #2519 | Edge store + LMDB materialisation auto-resolvers (6-resolver composed chain) |
| #2521 | stepST handlers for GetStructure/GetList/Unify* |
| #2522 | Runtime parser support (compiled prolog_term_parser, 50 predicates) |
| #2523 | Fix detect_kernels for module-qualified predicates |
| #2524 | Fix module qualifier stripping in wam_haskell_predicate_wamcode |
| #2526 | stepST handlers for ISO builtins (15 of 18 native, 3 on pure bridge) |
| #2527 | CSR benchmark validation test suite (15 tests) |
| #2528 | Cross-target benchmark script |

## Key technical decisions

- **ISO exceptions**: Haskell uses lazy `throw` (works in pure/ST) + `unsafePerformIO` + `try` for catch/3
- **Builder state**: Added `ReadArgs ![Value]` + `wsBuilderStack :: ![Builder]` for nested get_structure
- **CSR reader**: Binary search on in-memory .idx ByteString, positioned Handle reads for .val
- **Runtime parser**: 45 parser + 4 wrapper predicates appended via haskell_project_predicates/3
- **stepST ISO**: `throwIsoErrorPure` uses PureWamState bindings; `succLaxST` reads registers directly

## Known issues

- Main.hs always includes LMDB types (MDB_env) even when LMDB isn't used — blocks GHC compilation without lmdb package
- Runtime parser end-to-end: works but generation takes ~60s for 50 predicates
- succ_iso/2, throw/1, catch/3 still use pure bridge in stepST (appropriate due to exception semantics)

## Testing verified with

- SWI-Prolog 9.0.4
- GHC 9.4.7
- Test suites: Phase 1 (11), Phase 3 (8), Phase 4 (31), ISO (9), CSR (15), parser capability (39)
