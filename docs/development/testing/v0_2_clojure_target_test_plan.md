# Clojure Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Clojure code generation target testing

## Overview

This test plan covers the Clojure target for UnifyWeaver, which generates Lisp-style Clojure code with persistent data structures and functional idioms.

## Prerequisites

### System Requirements

- Clojure 1.11+
- Leiningen 2.10+ or deps.edn (tools.deps)
- JDK 17+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Clojure installation
clj --version
# or
lein version

# Verify JDK
java --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_clojure_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `def_generation` | def/defn | Definition syntax |
| `let_bindings` | let blocks | Binding syntax |
| `vector_literal` | [...] | Vector syntax |
| `map_literal` | {...} | Map syntax |
| `keyword_usage` | :keyword | Keyword syntax |

#### 1.2 Functional Idioms

```bash
swipl -g "use_module('tests/core/test_clojure_functional'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `map_filter_reduce` | Collection ops | HOF usage |
| `threading_macros` | -> and ->> | Threading syntax |
| `destructuring` | Pattern binding | Destructure syntax |

### 2. Compilation Tests

#### 2.1 Clojure Evaluation

```bash
./tests/integration/test_clojure_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `clj_eval` | clj evaluation | Code runs |
| `lein_run` | lein run | Project runs |
| `uberjar` | lein uberjar | Standalone JAR |

### 3. Generated Code Structure

```clojure
(ns com.unifyweaver.generated
  (:require [clojure.set :as set]
            [clojure.data.json :as json]))

(def initial-facts
  #{["parent" ["john" "mary"]]
    ["parent" ["mary" "susan"]]})

(defn apply-rules [total delta]
  (set
    (for [["ancestor" [x y]] delta
          ["parent" [y2 z]] total
          :when (= y y2)]
      ["ancestor" [x z]])))

(defn solve []
  (loop [facts initial-facts
         delta initial-facts]
    (if (empty? delta)
      facts
      (let [new-facts (set/difference (apply-rules facts delta) facts)]
        (recur (set/union facts new-facts) new-facts)))))

(defn -main [& args]
  (doseq [fact (solve)]
    (println (json/write-str {:relation (first fact)
                              :args (second fact)}))))
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/clojure_target'),
    compile_to_clojure(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_clojure_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLJ_CONFIG` | deps.edn location | (default) |
| `LEIN_HOME` | Leiningen home | ~/.lein |
| `SKIP_CLOJURE_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Startup time**: JVM startup overhead
2. **Parentheses**: S-expression density
3. **ClojureScript**: Different from JVM Clojure
