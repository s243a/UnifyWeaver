# Scala Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Scala code generation target testing

## Overview

This test plan covers the Scala target for UnifyWeaver, which generates functional Scala code with immutable data structures and pattern matching.

## Prerequisites

### System Requirements

- Scala 3.3+ (or Scala 2.13 for compatibility)
- sbt 1.9+ or Mill
- JDK 17+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Scala installation
scala --version
scalac --version

# Verify build tool
sbt --version

# Verify JDK
java --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_scala_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `case_class` | case class | Correct syntax |
| `object_generation` | object/companion | Singleton objects |
| `pattern_matching` | match/case | Pattern syntax |
| `for_comprehension` | for/yield | Comprehension syntax |
| `trait_generation` | trait definitions | Trait syntax |

#### 1.2 Scala 3 Features

```bash
swipl -g "use_module('tests/core/test_scala3_features'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `enum_types` | enum keyword | Scala 3 enums |
| `extension_methods` | extension keyword | Extension syntax |
| `given_using` | given/using | Context parameters |
| `union_types` | A \| B | Union types |

### 2. Compilation Tests

#### 2.1 Scala Compiler

```bash
./tests/integration/test_scala_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `scalac_success` | scalac compiles | No errors |
| `sbt_build` | sbt compile | Build succeeds |
| `jar_packaging` | sbt assembly | Fat JAR created |

### 3. Generated Code Structure

```scala
package com.unifyweaver.generated

import scala.collection.immutable.Set

case class Fact(relation: String, args: List[String])

object GeneratedQuery:
  private var facts: Set[Fact] = Set.empty
  private var delta: Set[Fact] = Set.empty

  def initFacts(): Unit =
    facts = Set(
      Fact("parent", List("john", "mary")),
      Fact("parent", List("mary", "susan"))
    )
    delta = facts

  def applyRules(total: Set[Fact], delta: Set[Fact]): Set[Fact] =
    for
      Fact("ancestor", List(x, y)) <- delta
      Fact("parent", List(y2, z)) <- total
      if y == y2
    yield Fact("ancestor", List(x, z))

  def solve(): Set[Fact] =
    initFacts()
    while delta.nonEmpty do
      val newFacts = applyRules(facts, delta)
      delta = newFacts -- facts
      facts = facts ++ delta
    facts

  @main def run(): Unit =
    solve().foreach(println)
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/scala_target'),
    compile_to_scala(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_scala_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCALA_HOME` | Scala installation | (system) |
| `SBT_OPTS` | sbt JVM options | (default) |
| `SKIP_SCALA_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Scala 2 vs 3**: Syntax differences
2. **Compilation speed**: Can be slow
3. **Binary compatibility**: Across Scala versions
