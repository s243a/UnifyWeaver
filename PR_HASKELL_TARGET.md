# feat: Add Haskell target for functional programming integration

## Summary

This PR adds a Haskell code generation target to UnifyWeaver, enabling Prolog predicates to be compiled to type-safe Haskell code with pattern matching and tail call optimization.

## Integration Test Results ✅

```
[PASS] sumTo 10 0 = 55
[PASS] sumTo 100 0 = 5050
[PASS] factorial 5 = 120
[PASS] factorial 10 = 3628800
[PASS] fib 10 = 55
[PASS] fib 15 = 610
```

**6/6 tests PASS** with GHC 8.6.5

## New Features

### Predicates

| Predicate | Purpose |
|-----------|---------|
| `compile_facts_to_haskell/3` | Facts → list of tuples |
| `compile_recursion_to_haskell/3` | Tail recursion with BangPatterns |
| `compile_rules_to_haskell/3` | Transitive closure (ancestor) |
| `compile_module_to_haskell/3` | Multiple predicates in one module |

### Generated Code

```haskell
{-# LANGUAGE BangPatterns #-}

sumTo :: Int -> Int -> Int
sumTo 0 !acc = acc
sumTo n !acc = sumTo (n - 1) (acc + n)

factorial :: Int -> Int
factorial 0 = 1
factorial n = n * factorial (n - 1)

fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)
```

## Unit Tests: 4/4 Pass

- compile_recursion (BangPatterns)
- compile_rules (ancestor)
- compile_module
- factorial pattern

## Files

### Implementation
```
src/unifyweaver/targets/haskell_target.pl   [NEW] 250 lines
tests/test_haskell_target.pl                [NEW] 80 lines
docs/proposals/haskell_target_design.md     [NEW] design doc
docs/HASKELL_TARGET.md                      [NEW] main doc
```

### Examples
```
examples/haskell/
├── PrologMath.hs      # Generated Haskell module
├── Main.hs            # Integration test
├── haskell-test       # Compiled binary (GHC)
├── build.sh           # Build script
└── README.md
```

### Education Book
```
education/other-books/book-haskell-target/
├── README.md
├── 01_introduction.md    # Why Haskell, architecture
└── 02_recursion.md       # BangPatterns, TCO
```

## Why Haskell?

| Feature | Benefit |
|---------|---------|
| Pattern matching | Natural fit for Prolog clauses |
| Lazy evaluation | Good for streams/backtracking |
| Type safety | Catch errors at compile time |
| BangPatterns | Tail call optimization via strictness |

## Dependencies

```bash
sudo apt install ghc
```

## Usage

```bash
cd examples/haskell
./build.sh  # Generates, compiles, and tests
```
