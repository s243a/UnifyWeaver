# Haskell Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Haskell code generation target testing

## Overview

This test plan covers the Haskell target for UnifyWeaver, which generates pure functional Haskell code with lazy evaluation and strong typing.

## Prerequisites

### System Requirements

- GHC 9.2+ (Glasgow Haskell Compiler)
- Cabal 3.6+ or Stack 2.9+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify GHC installation
ghc --version
ghci --version

# Verify build tool
cabal --version
# or
stack --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_haskell_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `data_types` | data declarations | ADT syntax |
| `type_signatures` | :: annotations | Type signatures |
| `function_definitions` | Pattern matching | Function clauses |
| `guards` | \| conditions | Guard syntax |
| `where_clauses` | Local bindings | where blocks |

#### 1.2 Type System Tests

```bash
swipl -g "use_module('tests/core/test_haskell_types'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `parametric_types` | Maybe, Either | Parametric ADTs |
| `type_classes` | Eq, Ord, Show | Instance derivation |
| `record_syntax` | Named fields | Record accessors |

### 2. Compilation Tests

#### 2.1 GHC Compilation

```bash
./tests/integration/test_haskell_ghc.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `ghc_compile` | ghc -c | Compiles |
| `warnings_clean` | -Wall | No warnings |
| `type_check` | Type inference | Types correct |

### 3. Generated Code Structure

```haskell
{-# LANGUAGE DeriveGeneric #-}

module GeneratedQuery where

import Data.Set (Set)
import qualified Data.Set as Set
import Data.Hashable
import GHC.Generics

data Fact = Fact
  { relation :: String
  , args :: [String]
  } deriving (Eq, Ord, Show, Generic)

instance Hashable Fact

initialFacts :: Set Fact
initialFacts = Set.fromList
  [ Fact "parent" ["john", "mary"]
  , Fact "parent" ["mary", "susan"]
  ]

applyRules :: Set Fact -> Set Fact -> Set Fact
applyRules total delta = Set.fromList
  [ Fact "ancestor" [x, z]
  | Fact "ancestor" [x, y] <- Set.toList delta
  , Fact "parent" [y', z] <- Set.toList total
  , y == y'
  ]

solve :: Set Fact
solve = go initialFacts initialFacts
  where
    go total delta
      | Set.null newFacts = total
      | otherwise = go total' newFacts
      where
        newFacts = applyRules total delta `Set.difference` total
        total' = total `Set.union` newFacts

main :: IO ()
main = mapM_ print $ Set.toList solve
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/haskell_target'),
    compile_to_haskell(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_haskell_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GHC_PATH` | GHC compiler path | (system) |
| `SKIP_HASKELL_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Lazy evaluation**: May cause space leaks
2. **String performance**: Use Text for large strings
3. **Cabal hell**: Version conflicts possible
