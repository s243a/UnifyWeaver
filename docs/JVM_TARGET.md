# JVM Target Family

UnifyWeaver provides JVM (Java Virtual Machine) targets for generating pipelines in Java, Scala, Kotlin, Clojure, and Jython.

## Status

| Target | Module | Status |
|--------|--------|--------|
| Java | `java_target.pl` | ✅ Initial (pipeline mode) |
| Jython | `jython_target.pl` | ✅ Initial (pipeline mode) |
| Kotlin | `kotlin_target.pl` | ✅ Initial (pipeline + generator modes) |
| Scala | `scala_target.pl` | ✅ Initial (pipeline + generator modes) |
| Clojure | — | 📋 Planned |

## Transport Support

JVM targets use `direct` transport for in-process communication:

```prolog
?- infer_transport_from_targets(java, scala, T).
T = direct.

?- infer_transport_from_targets(java, python, T).
T = pipe.
```

### JVM Glue Module

The `jvm_glue.pl` module provides:
- **Runtime detection** - JVM, Java version, Jython availability
- **Transport selection** - `direct` for JVM-to-JVM, `pipe` for others
- **Bridge generation** - Java ↔ Jython via PythonInterpreter
- **Launcher generation** - Shell scripts with classpath management
- **Mixed pipeline orchestration** - Combine Java and Jython steps

## Java Target

### Quick Start

```prolog
:- use_module('src/unifyweaver/targets/java_target').

% Compile predicate to Java
?- compile_predicate_to_java(my_pred/2, [pipeline_input(true)], Code).

% Generate Gradle build
?- generate_gradle_build([java_version('21')], GradleCode).
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `package(Name)` | `generated` | Java package name |
| `class_name(Name)` | Predicate name | Output class name |
| `pipeline_input(Bool)` | `false` | Enable streaming JSONL |
| `main_class(Bool)` | `true` | Generate main method |

### Build System

Gradle is the default build system. Generated projects include:
- `build.gradle` with Gson dependency
- Fat JAR task for standalone execution
- JUnit 5 test support

## Roadmap

### Phase 1: Java Foundation (Current)
- [x] Basic predicate compilation
- [x] Pipeline mode with JSONL streaming
- [x] Gradle build generation
- [x] Body translation (goals → Java)
- [x] Recursion detection
- [x] Arithmetic expressions (+, -, *, /, mod)
- [x] Comparison operators (>, <, >=, =<, =:=, =\=)
- [x] get_dict/3 support

### Phase 2: Kotlin & Scala
- [ ] Kotlin target with coroutine support
- [ ] Scala target with functional idioms
- [ ] Shared JVM glue for in-process calls

### Phase 3: Scripting Languages
- [ ] Jython target (Python on JVM)
- [ ] Clojure target

### Phase 4: Advanced Features
- [ ] Maven build option
- [ ] GraalVM native-image support
- [ ] Inter-JVM-language bridges
