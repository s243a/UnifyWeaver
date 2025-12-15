# Haskell Target

Compile Prolog predicates to Haskell for type-safe functional programming integration.

## Overview

| Feature | Status |
|---------|--------|
| Facts | ✅ List of tuples |
| Recursion | ✅ BangPatterns + TCO |
| Rules | ✅ Pattern matching |
| Modules | ✅ Multiple predicates |

## Quick Start

```prolog
?- use_module('src/unifyweaver/targets/haskell_target').

?- compile_module_to_haskell(
       [pred(sum, 2, tail_recursion),
        pred(factorial, 1, factorial)],
       [module_name('PrologMath')],
       Code),
   write_haskell_module(Code, 'PrologMath.hs').
```

```bash
ghc -O2 PrologMath.hs -o program
./program
```

## API Reference

### `compile_recursion_to_haskell/3`
Tail recursion with strict accumulator:
```prolog
compile_recursion_to_haskell(+Pred/Arity, +Options, -HaskellCode)
```

Generated:
```haskell
{-# LANGUAGE BangPatterns #-}
sumTo :: Int -> Int -> Int
sumTo 0 !acc = acc
sumTo n !acc = sumTo (n - 1) (acc + n)
```

### `compile_rules_to_haskell/3`
Transitive closure (ancestor pattern):
```prolog
compile_rules_to_haskell(ancestor/2, [base_pred(parent)], Code)
```

### `compile_module_to_haskell/3`
Multiple predicates in one module:
```prolog
compile_module_to_haskell(
    [pred(sum, 2, tail_recursion),
     pred(factorial, 1, factorial),
     pred(fib, 2, linear_recursion)],
    [module_name('PrologMath')],
    Code).
```

## Function Types

| Type | Description | Haskell Pattern |
|------|-------------|-----------------|
| `tail_recursion` | O(1) stack | `f n !acc = ...` |
| `factorial` | Simple recursion | `f n = n * f (n-1)` |
| `linear_recursion` | e.g., Fibonacci | `f n = f(n-1) + f(n-2)` |

## Dependencies

```bash
# Ubuntu/Debian
sudo apt install ghc

# Or GHCup
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

## See Also

- [haskell_target_design.md](./proposals/haskell_target_design.md) - Design doc
- [examples/haskell/](../examples/haskell/) - Working examples
