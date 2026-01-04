# Kotlin Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Kotlin code generation target testing

## Overview

This test plan covers the Kotlin target for UnifyWeaver, which generates idiomatic Kotlin code for JVM, Android, and multiplatform projects.

## Prerequisites

### System Requirements

- Kotlin 1.9+ compiler
- JDK 17+ (for JVM target)
- Gradle 8.0+ or Maven 3.8+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Kotlin installation
kotlinc -version

# Verify JDK
java --version

# Verify build tool
gradle --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_kotlin_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `data_class` | data class | Correct syntax |
| `null_safety` | ? and !! | Nullable types |
| `extension_functions` | fun T.ext() | Extension syntax |
| `lambda_expressions` | { } syntax | Lambda handling |
| `when_expression` | when {} | Pattern matching |

#### 1.2 Kotlin Idioms

```bash
swipl -g "use_module('tests/core/test_kotlin_idioms'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `scope_functions` | let, run, apply | Scope function usage |
| `sequences` | Sequence<T> | Lazy sequences |
| `coroutines` | suspend fun | Coroutine syntax |

### 2. Compilation Tests

#### 2.1 Kotlin Compiler

```bash
./tests/integration/test_kotlin_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `kotlinc_success` | kotlinc compiles | No errors |
| `gradle_build` | Gradle build | Build succeeds |
| `jar_packaging` | JAR creation | Runnable JAR |

### 3. Generated Code Structure

```kotlin
package com.unifyweaver.generated

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
data class Fact(
    val relation: String,
    val args: List<String>
)

class GeneratedQuery {
    private val facts = mutableSetOf<Fact>()
    private var delta = mutableSetOf<Fact>()

    init {
        initFacts()
    }

    private fun initFacts() {
        facts += Fact("parent", listOf("john", "mary"))
        delta = facts.toMutableSet()
    }

    fun solve(): Set<Fact> {
        while (delta.isNotEmpty()) {
            val newFacts = applyRules()
            delta = (newFacts - facts).toMutableSet()
            facts += delta
        }
        return facts
    }

    private fun applyRules(): Set<Fact> = buildSet {
        // Rule implementations
    }
}

fun main() {
    val result = GeneratedQuery().solve()
    result.forEach { println(Json.encodeToString(Fact.serializer(), it)) }
}
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/kotlin_target'),
    compile_to_kotlin(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_kotlin_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KOTLIN_HOME` | Kotlin installation | (system) |
| `JAVA_HOME` | JDK path | (system) |
| `SKIP_KOTLIN_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Compilation speed**: Slower than Java
2. **Kotlin/Native**: Different from JVM target
3. **Multiplatform**: expect/actual needs care
