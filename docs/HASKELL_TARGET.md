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
     pred(fib, 2, linear_recursion),
     pred(listSum, 2, list_fold),
     pred(sumAcc, 3, list_tail_recursion)],
    [module_name('PrologMath')],
    Code).
```

### `compile_predicate_to_haskell/3`
Generic dispatcher with type option:
```prolog
compile_predicate_to_haskell(parent/2, [type(facts)], Code)
compile_predicate_to_haskell(ancestor/2, [type(rules)], Code)
compile_predicate_to_haskell(sum/3, [type(recursion)], Code)
```

## Function Types

| Type | Description | Haskell Pattern |
|------|-------------|-----------------|
| `tail_recursion` | O(1) stack | `f n !acc = ...` |
| `factorial` | Simple recursion | `f n = n * f (n-1)` |
| `linear_recursion` | e.g., Fibonacci | `f n = f(n-1) + f(n-2)` |
| `list_fold` | List sum using foldr | `f = foldr (+) 0` |
| `list_tail_recursion` | List with accumulator | `f [] !acc = acc` |

## Parsec: DCG to Parser Combinators

Compile Prolog DCGs to Haskell Parsec parsers:

```prolog
?- compile_dcg_to_parsec((digit --> [d]), [module_name('DigitParser')], Code).
```

Generates:

```haskell
{-# LANGUAGE OverloadedStrings #-}
module DigitParser where

import Text.Parsec
import Text.Parsec.String (Parser)

digit :: Parser String
digit = string "d"
```

### Multiple DCG Rules

```prolog
?- compile_grammar_to_parsec(
       [(expr --> term),
        (term --> digit),
        (digit --> [x])],
       [module_name('ExprParser'), start_symbol(expr)],
       Code).
```

### DCG Pattern Mapping

| DCG | Parsec | Description |
|-----|--------|-------------|
| `[char]` | `string "char"` | Terminal |
| `a, b` | `a *> b` | Sequence |
| `a ; b` | `try a <|> b` | Alternative |
| `{goal}` | `pure ()` | Prolog goal |
| Non-terminal | Recursive call | Rule reference |
| `*(A)` | `many A` | Zero or more |
| `+(A)` | `some A` | One or more |
| `?(A)` | `optional A` | Optional |
| `letter` | `letter` | Any letter |
| `digit` | `digit` | Any digit |
| `alpha_num` | `alphaNum` | Letter or digit |
| `space` | `space` | Whitespace |
| `not(A)` | `notFollowedBy A` | Negation |
| `lookahead(A)` | `lookAhead A` | Positive lookahead |

## Dependencies

```bash
# Ubuntu/Debian
sudo apt install ghc

# Parsec (if not bundled)
cabal install parsec

# Or GHCup
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

## See Also

- [haskell_target_design.md](./proposals/haskell_target_design.md) - Design doc
- [examples/haskell/](../examples/haskell/) - Working examples
