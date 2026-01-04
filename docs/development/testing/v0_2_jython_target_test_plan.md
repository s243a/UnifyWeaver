# Jython Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Jython code generation target testing

## Overview

This test plan covers the Jython target for UnifyWeaver, which generates Python code that runs on the Java Virtual Machine via Jython. This enables seamless Java library integration while using Python syntax.

## Prerequisites

### System Requirements

- Jython 2.7+ (standalone JAR)
- JDK 11+ (JDK 17+ recommended)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Jython installation
java -jar jython.jar --version
# or if installed
jython --version

# Verify JDK
java --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_jython_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `def_generation` | def keyword | Function syntax |
| `class_generation` | class keyword | Class syntax |
| `import_java` | from java.* | Java imports |
| `list_comprehension` | [x for x in y] | Comprehension |
| `dict_literal` | {...} | Dictionary syntax |

#### 1.2 Java Interop

```bash
swipl -g "use_module('tests/core/test_jython_java_interop'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `java_import` | Java class import | Correct import |
| `java_call` | Java method call | Method invocation |
| `java_collections` | ArrayList, HashMap | Java collections |
| `type_conversion` | Jython↔Java types | Proper conversion |

### 2. Compilation Tests

#### 2.1 Jython Execution

```bash
./tests/integration/test_jython_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `jython_run` | jython script.py | Code runs |
| `java_interop` | Use Java classes | Integration works |
| `standalone_jar` | jythonc packaging | JAR created |

### 3. Generated Code Structure

```python
from java.util import ArrayList, HashSet
from java.lang import System

class Fact:
    def __init__(self, relation, args):
        self.relation = relation
        self.args = args

    def __hash__(self):
        return hash((self.relation, tuple(self.args)))

    def __eq__(self, other):
        return (self.relation == other.relation and
                self.args == other.args)

class GeneratedQuery:
    def __init__(self):
        self.facts = HashSet()
        self.delta = HashSet()

    def init_facts(self):
        self.facts.add(Fact("parent", ["john", "mary"]))
        self.facts.add(Fact("parent", ["mary", "susan"]))
        self.delta = HashSet(self.facts)

    def apply_rules(self, total, delta):
        new_facts = HashSet()
        # Rule implementations
        return new_facts

    def solve(self):
        self.init_facts()
        while self.delta.size() > 0:
            new_facts = self.apply_rules(self.facts, self.delta)
            self.delta = HashSet()
            for f in new_facts:
                if not self.facts.contains(f):
                    self.delta.add(f)
                    self.facts.add(f)
        return self.facts

if __name__ == "__main__":
    query = GeneratedQuery()
    for fact in query.solve():
        System.out.println("%s(%s)" % (fact.relation, ", ".join(fact.args)))
```

### 4. Java Library Integration Tests

#### 4.1 Common Libraries

```bash
./tests/integration/test_jython_libraries.sh
```

**Test Cases**:
| Library | Description | Expected |
|---------|-------------|----------|
| java.util | Collections | Works |
| java.io | File I/O | Works |
| java.sql | JDBC | Connections work |
| javax.xml | XML processing | Works |

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/jython_target'),
    compile_to_jython(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_jython_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JYTHON_HOME` | Jython installation | (system) |
| `JAVA_HOME` | JDK location | (system) |
| `CLASSPATH` | Java classpath | (default) |
| `SKIP_JYTHON_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Python 2 only**: Jython is Python 2.7 compatible
2. **Performance**: Slower startup than CPython
3. **C extensions**: No support for CPython C extensions
4. **Version lag**: Lags behind CPython features

## Related Documentation

- [Jython Documentation](https://www.jython.org/docs/)
- [Java Integration Guide](https://jython.readthedocs.io/en/latest/JythonAndJavaIntegration/)
