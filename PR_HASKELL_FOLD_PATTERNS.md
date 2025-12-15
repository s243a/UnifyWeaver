# feat: Add list_fold and list_tail_recursion patterns to Haskell target

## Summary

Extends the Haskell target with two new pattern types for list operations, enabling more idiomatic Haskell code generation.

## New Pattern Types

| Pattern | Generated Haskell | Description |
|---------|-------------------|-------------|
| `list_fold` | `listSum = foldr (+) 0` | Uses `foldr` for list operations |
| `list_tail_recursion` | `sumAcc [] !acc = acc` | Tail-recursive with strictness |

## Tests: 6/6 Pass

```
[PASS] compile_recursion
[PASS] compile_rules (ancestor)
[PASS] compile_module
[PASS] factorial pattern
[PASS] list_fold pattern ✨
[PASS] list_tail_recursion pattern ✨
```

## Usage

```prolog
?- compile_module_to_haskell(
       [pred(listSum, 2, list_fold),
        pred(sumAcc, 3, list_tail_recursion)],
       [module_name('ListOps')],
       Code).
```

## Generated Code Examples

### list_fold
```haskell
listSum :: [Int] -> Int
listSum = foldr (+) 0
```

### list_tail_recursion
```haskell
{-# LANGUAGE BangPatterns #-}
sumAcc :: [Int] -> Int -> Int
sumAcc [] !acc = acc
sumAcc (h:t) !acc = sumAcc t (acc + h)
```

## Files Changed

### Implementation
```
src/unifyweaver/targets/haskell_target.pl   [MODIFIED] +17 lines
tests/test_haskell_target.pl                [MODIFIED] +27 lines
```

### Documentation
```
docs/HASKELL_TARGET.md                      [MODIFIED]
- Added list_fold, list_tail_recursion to Function Types table
- Added compile_predicate_to_haskell/3 API section
```

### Education Book
```
education/other-books/book-haskell-target/02_recursion.md
- Fixed tail_recursion → list_tail_recursion
- Added "Alternative: Generic Predicate" section
```

## Commits

| Commit | Description |
|--------|-------------|
| `b6f3498` | feat: Add patterns to haskell_target.pl |
| `e87bf32` | docs: Update HASKELL_TARGET.md |
| `ca119be` | docs: Fix education book Ch2 |

## Related

- PR #349: Initial Haskell target
- PR #350: Extended Targets table
