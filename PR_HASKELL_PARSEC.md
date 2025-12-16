# feat: Add Parsec parser combinator support and standard interface to Haskell target

## Summary

Extends the Haskell target with:
1. **Standard interface** (`target_info/1`, `compile_predicate/3`) for `target_module/2` dispatch
2. **Parsec support** - Compile Prolog DCGs to Haskell parser combinators

## New Features

### Standard Interface
- `target_info/1` - Returns metadata (family: functional, features, parser_support: parsec)
- `compile_predicate/3` - Wrapper enabling unified `target_module/2` dispatch

### Parsec (DCG → Parser Combinators)

```prolog
?- compile_dcg_to_parsec((digit --> [d]), [module_name('Parser')], Code).
```

Generates:

```haskell
digit :: Parser String
digit = string "d"
```

### DCG Pattern Mapping

| Prolog DCG | Haskell Parsec |
|------------|----------------|
| `[char]` | `string "char"` |
| `a, b` | `a *> b` |
| `a ; b` | `try a <|> b` |
| `{goal}` | `pure ()` |
| Non-terminal | Recursive call |

## Tests: 11/11 Pass

```
[PASS] compile_recursion
[PASS] compile_rules (ancestor)
[PASS] compile_module
[PASS] factorial pattern
[PASS] list_fold pattern
[PASS] list_tail_recursion pattern
[PASS] target_info (standard interface) ✨
[PASS] compile_predicate (standard interface) ✨
[PASS] DCG to Parsec (simple) ✨
[PASS] DCG to Parsec (sequence) ✨
[PASS] Grammar to Parsec (multiple rules) ✨
```

## Files Changed

### Implementation
```
src/unifyweaver/targets/haskell_target.pl   [MODIFIED] +205 lines
- Added target_info/1, compile_predicate/3
- Added compile_dcg_to_parsec/3, compile_grammar_to_parsec/3
- Added dcg_body_to_parsec/2 (terminal, sequence, alternative, non-terminal)
- Added parsec_common_helpers/1

tests/test_haskell_target.pl                [MODIFIED] +60 lines
- Added 5 new tests for standard interface and Parsec
```

### Documentation
```
docs/HASKELL_TARGET.md                      [MODIFIED]
- Added Parsec section with DCG pattern mapping
- Added compile_dcg_to_parsec/3 example

education/other-books/book-haskell-target/
- 03_parsec.md                              [NEW] 120 lines
- README.md                                 [MODIFIED] Added Chapter 3
```

## Technical Notes

- Parsec is GHC's standard parser combinator library (bundled with GHC)
- DCG → Parsec enables declarative parser generation from Prolog grammars
- Standard interface enables unified target dispatch via `target_module/2`
- Added cuts to `dcg_body_to_parsec/2` clauses to prevent ambiguous matching

## Related

- PR #352: JavaScript Family and TypeScript Target (introduced `target_module/2`)
- PR #351: Haskell fold patterns
