# Java Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Java code generation target testing

## Overview

This test plan covers the Java target for UnifyWeaver, which generates Java programs with semi-naive fixpoint evaluation, suitable for enterprise integration and JVM deployment.

## Prerequisites

### System Requirements

- Java JDK 17+ (LTS recommended)
- Maven 3.8+ or Gradle 8.0+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Java installation
java --version
javac --version

# Verify build tool
mvn --version
# or
gradle --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

These tests verify Java code generation without compiling the generated code.

#### 1.1 Basic Generator Tests

```bash
# Run Java generator tests
swipl -g "use_module('tests/core/test_java_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `class_generation` | Main class structure | Public class with main() |
| `fact_compilation` | Prolog facts | Static initializer block |
| `rule_compilation` | Prolog rules | Method generation |
| `fixpoint_loop` | Semi-naive evaluation | While loop with delta tracking |
| `type_inference` | Variable typing | Proper Java types |

#### 1.2 Data Structure Tests

```bash
swipl -g "use_module('tests/core/test_java_data_structures'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `hashset_usage` | Fact storage | `HashSet<Fact>` generation |
| `record_types` | Java records | `record Fact(...)` syntax |
| `stream_api` | Java Streams | Stream operations for queries |
| `collections` | List/Map usage | Proper generics |

### 2. Integration Tests (Compilation + Execution)

These tests compile Prolog to Java, build with Maven/Gradle, and execute.

#### 2.1 Maven Build Tests

```bash
./tests/integration/test_java_maven_build.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `compile_success` | javac passes | Exit code 0 |
| `test_execution` | JUnit tests pass | All tests green |
| `jar_packaging` | JAR creation | Executable JAR |
| `dependency_resolution` | Maven deps | Dependencies resolved |

#### 2.2 Execution Tests

```bash
swipl -g "use_module('tests/test_java_execution'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `transitive_closure` | Ancestor query | Correct ancestors found |
| `aggregation` | Sum/count/avg | Correct aggregates |
| `json_input` | JSON parsing | Jackson/Gson parsing works |
| `json_output` | JSON generation | Valid JSON output |

### 3. Code Quality Tests

#### 3.1 Static Analysis

```bash
# Run with SpotBugs
mvn spotbugs:check -f /tmp/generated_project/pom.xml

# Run with Checkstyle
mvn checkstyle:check -f /tmp/generated_project/pom.xml
```

**Quality Gates**:
| Check | Threshold |
|-------|-----------|
| SpotBugs bugs | 0 high priority |
| Checkstyle violations | 0 errors |
| Compiler warnings | 0 |

#### 3.2 Generated Code Structure

Verify generated code follows Java conventions:

```java
package com.unifyweaver.generated;

import java.util.*;
import java.util.stream.*;

public record Fact(String relation, List<String> args) {}

public class GeneratedQuery {
    private static final Set<Fact> initialFacts = Set.of(
        new Fact("parent", List.of("john", "mary")),
        // ...
    );

    public static Set<Fact> solve() {
        Set<Fact> total = new HashSet<>(initialFacts);
        Set<Fact> delta = new HashSet<>(initialFacts);

        while (!delta.isEmpty()) {
            Set<Fact> newFacts = new HashSet<>();
            // Apply rules...
            delta = newFacts;
            delta.removeAll(total);
            total.addAll(delta);
        }
        return total;
    }

    public static void main(String[] args) {
        solve().forEach(System.out::println);
    }
}
```

### 4. Performance Tests

#### 4.1 JVM Warmup Tests

```bash
# Run with JMH benchmark
java -jar target/benchmarks.jar
```

**Benchmarks**:
| Test | Cold Start | Warmed Up |
|------|------------|-----------|
| Simple query | < 500ms | < 10ms |
| 1000 facts | < 1s | < 50ms |
| 10000 facts | < 5s | < 200ms |

#### 4.2 Memory Usage

```bash
# Monitor heap usage
java -Xmx256m -XX:+PrintGCDetails -jar generated.jar
```

**Memory Targets**:
| Test | Max Heap |
|------|----------|
| Simple query | < 64MB |
| 10000 facts | < 128MB |
| 100000 facts | < 512MB |

### 5. JVM Version Compatibility

#### 5.1 Multi-Version Testing

```bash
# Test with different JDKs
JAVA_HOME=/usr/lib/jvm/java-17 ./tests/integration/test_java_compat.sh
JAVA_HOME=/usr/lib/jvm/java-21 ./tests/integration/test_java_compat.sh
```

**Compatibility Matrix**:
| Feature | Java 17 | Java 21 |
|---------|---------|---------|
| Records | Yes | Yes |
| Pattern matching | Preview | Yes |
| Virtual threads | No | Yes |
| Sealed classes | Yes | Yes |

## Test Commands Reference

### Quick Smoke Test

```bash
# Generate Java code
swipl -g "
    use_module('src/unifyweaver/targets/java_target'),
    compile_to_java(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
# Run all Java tests
./tests/run_java_tests.sh

# Or individually:
swipl -g "use_module('tests/core/test_java_generator'), run_tests" -t halt
./tests/integration/test_java_maven_build.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JAVA_HOME` | JDK installation path | (system) |
| `MAVEN_HOME` | Maven installation path | (system) |
| `SKIP_JAVA_EXECUTION` | Skip runtime tests | `0` |
| `JAVA_TARGET_VERSION` | Target Java version | `17` |
| `KEEP_JAVA_ARTIFACTS` | Preserve generated code | `0` |

## Known Issues

1. **Java 8 incompatible**: Uses records (Java 14+)
2. **Large fact sets**: Memory pressure with >1M facts
3. **GraalVM native-image**: Reflection config needed

## Related Documentation

- [Java Target Implementation](../../architecture/targets/java_target.md)
- [JVM Optimization Guide](../../architecture/jvm_optimization.md)
- [Cross-Runtime Pipelines](../../architecture/cross_runtime_pipelines.md)
