# Haskell Target Proposal

## Overview

Compile Prolog predicates to Haskell for type-safe functional programming integration.

## Why Haskell?

| Feature | Benefit for Prolog |
|---------|-------------------|
| **Pattern matching** | Prolog clauses → Haskell patterns |
| **Lazy evaluation** | Natural for backtracking/streams |
| **Type safety** | Add types to predicates |
| **Purity** | Prolog facts are pure |
| **GHC optimizer** | Excellent performance |

## Architecture

```
┌───────────────────────────────────────┐
│        UnifyWeaver (Prolog)           │
│  compile_predicate_to_haskell/3       │
└─────────────────┬─────────────────────┘
                  │
                  ▼
            Module.hs (Haskell source)
                  │
                  ▼
         ghc -O2 Module.hs
                  │
                  ▼
           Native binary or library
```

## Proposed API

```prolog
:- module(haskell_target, [
    compile_predicate_to_haskell/3,    % +Pred/Arity, +Options, -HaskellCode
    compile_facts_to_haskell/3,        % +Pred, +Arity, -HaskellCode
    compile_rules_to_haskell/3,        % +Pred/Arity, +Options, -HaskellCode
    compile_recursion_to_haskell/3,    % +Pred/Arity, +Options, -HaskellCode
    write_haskell_module/2,            % +Code, +Filename
    init_haskell_target/0
]).
```

## Translation Examples

### Facts → Data + List

```prolog
parent(tom, bob).
parent(bob, jim).
```

→

```haskell
module Parent where

data Person = Tom | Bob | Jim
  deriving (Eq, Show, Enum, Bounded)

parent :: [(Person, Person)]
parent = [(Tom, Bob), (Bob, Jim)]

isParent :: Person -> Person -> Bool
isParent x y = (x, y) `elem` parent
```

### Rules → Functions

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

→

```haskell
ancestor :: Person -> Person -> Bool
ancestor x y = isParent x y
            || any (\z -> isParent x z && ancestor z y) allPersons

allPersons :: [Person]
allPersons = [minBound..maxBound]
```

### Tail Recursion

```prolog
sum(0, Acc, Acc).
sum(N, Acc, Result) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, Result).
```

→

```haskell
sum :: Int -> Int -> Int
sum 0 acc = acc
sum n acc = sum (n - 1) (acc + n)  -- GHC optimizes to loop
```

## Type Inference

| Prolog | Inferred Haskell |
|--------|------------------|
| `foo(1, 2)` | `Int -> Int` |
| `foo(a, b)` | `Atom -> Atom` (data type) |
| `foo("hi")` | `String` |
| `foo(X, Y) :- X > Y` | `Ord a => a -> a` |

## Options

```prolog
compile_predicate_to_haskell(ancestor/2, [
    module_name('Ancestor'),       % Haskell module name
    type_hints([person]),          % Use custom types
    export(true),                  % Export from module
    strict(true)                   % Use strict evaluation
], Code).
```

## Generated Module Structure

```haskell
{-# LANGUAGE DeriveGeneric #-}
module Ancestor
    ( Person(..)
    , parent
    , ancestor
    ) where

import GHC.Generics (Generic)
import Data.List (nub)

-- Data types
data Person = Tom | Bob | Jim
  deriving (Eq, Show, Enum, Bounded, Generic)

-- Facts
parent :: [(Person, Person)]
parent = [(Tom, Bob), (Bob, Jim)]

-- Rules
ancestor :: Person -> Person -> Bool
ancestor x y = ...
```

## Build Commands

```bash
# Compile single module
ghc -O2 Ancestor.hs -o ancestor

# Create library
ghc -O2 -c Ancestor.hs
ar rcs libancestor.a Ancestor.o

# Use with Stack/Cabal
stack build
cabal build
```

## Implementation Phases

| Phase | Scope |
|-------|-------|
| **1** | Facts → List of tuples |
| **2** | Rules → Functions with pattern matching |
| **3** | Recursion → Tail-recursive functions |
| **4** | Type inference + custom types |

## Dependencies

```bash
# Ubuntu/Debian
sudo apt install ghc

# Or use GHCup (recommended)
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

## Advantages Over Other Targets

| Feature | Haskell | Go | Rust |
|---------|---------|-----|------|
| Pattern matching | ✅ Native | ❌ Switch | ❌ Match |
| Lazy evaluation | ✅ Default | ❌ | ❌ |
| Type inference | ✅ HM | ❌ | ✅ |
| TCO guarantee | ✅ GHC | ❌ | ❌ |
| Purity | ✅ | ❌ | ❌ |

## Status

- [ ] Phase 1: Facts
- [ ] Phase 2: Rules
- [ ] Phase 3: Recursion
- [ ] Phase 4: Types
