# Python Accelerators Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Python accelerator code generation targets testing (Codon, Cython, mypyc, Nuitka, Numba)

## Overview

This test plan covers the Python accelerator targets for UnifyWeaver. These targets generate Python code optimized for various compilation/acceleration frameworks that provide near-native performance.

### Accelerator Targets

| Target | Description | Output |
|--------|-------------|--------|
| `python_codon_target` | Codon AOT compiler | Type-annotated Python |
| `python_cython_target` | Cython transpiler | .pyx files |
| `python_mypyc_target` | mypyc compiler | Type-annotated Python |
| `python_nuitka_target` | Nuitka compiler | Optimized Python |
| `python_numba_target` | Numba JIT | @jit decorated Python |

## Prerequisites

### System Requirements

- Python 3.10+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

#### Per-Accelerator Requirements

**Codon:**
```bash
pip install codon
# or from source
```

**Cython:**
```bash
pip install cython
```

**mypyc:**
```bash
pip install mypy
```

**Nuitka:**
```bash
pip install nuitka
```

**Numba:**
```bash
pip install numba
```

### Verification

```bash
# Verify Python
python3 --version

# Verify accelerators
codon --version 2>/dev/null || echo "Codon not installed"
cython --version 2>/dev/null || echo "Cython not installed"
mypyc --version 2>/dev/null || echo "mypyc not installed"
python3 -c "import nuitka; print(nuitka.__version__)" 2>/dev/null || echo "Nuitka not installed"
python3 -c "import numba; print(numba.__version__)" 2>/dev/null || echo "Numba not installed"

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Codon Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_codon_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `type_annotations` | : type hints | Static types |
| `codon_primitives` | Codon builtins | Codon-specific |
| `parallel_pragma` | @par decorator | Parallelism |

#### 1.2 Cython Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_cython_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `cdef_types` | cdef int x | C types |
| `cpdef_functions` | cpdef func | Hybrid functions |
| `memoryview` | typed memoryviews | Buffer access |
| `pxd_headers` | .pxd generation | Header files |

#### 1.3 mypyc Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_mypyc_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `strict_typing` | Full annotations | Type coverage |
| `final_classes` | @final decorator | Optimization hints |
| `dataclasses` | @dataclass | Compiled dataclass |

#### 1.4 Nuitka Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_nuitka_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `standard_python` | Regular Python | Compiles |
| `module_structure` | __main__ | Entry point |
| `plugin_compat` | Plugin annotations | Plugin hints |

#### 1.5 Numba Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_numba_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `jit_decorator` | @jit | JIT compilation |
| `njit_decorator` | @njit | No-python mode |
| `vectorize` | @vectorize | SIMD ops |
| `numpy_usage` | NumPy operations | Array ops |

### 2. Compilation Tests

#### 2.1 Codon Compilation

```bash
./tests/integration/test_codon_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `codon_build` | codon build | Binary created |
| `codon_run` | codon run | Execution works |
| `parallel_execution` | @par functions | Parallelism |

#### 2.2 Cython Compilation

```bash
./tests/integration/test_cython_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `cythonize` | cythonize -i | .so created |
| `setup_build` | python setup.py | Build succeeds |
| `import_module` | import compiled | Module works |

#### 2.3 mypyc Compilation

```bash
./tests/integration/test_mypyc_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `mypyc_compile` | mypyc module.py | .so created |
| `type_check` | mypy module.py | No errors |
| `import_compiled` | import module | Works |

#### 2.4 Nuitka Compilation

```bash
./tests/integration/test_nuitka_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `nuitka_compile` | nuitka --module | Module compiled |
| `nuitka_standalone` | --standalone | Binary created |
| `nuitka_onefile` | --onefile | Single binary |

#### 2.5 Numba Compilation

```bash
./tests/integration/test_numba_compilation.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `jit_warmup` | First call | JIT compiles |
| `cached_execution` | Subsequent calls | Fast execution |
| `aot_compilation` | AOT mode | .so created |

### 3. Generated Code Examples

#### 3.1 Codon Code

```python
from typing import Set, Tuple

def solve() -> Set[Tuple[str, str]]:
    facts: Set[Tuple[str, str]] = {("parent", "john", "mary"), ("parent", "mary", "susan")}
    delta: Set[Tuple[str, str]] = set(facts)

    while delta:
        new_facts: Set[Tuple[str, str]] = set()
        for rel, x, y in delta:
            if rel == "ancestor":
                for rel2, y2, z in facts:
                    if rel2 == "parent" and y == y2:
                        new_facts.add(("ancestor", x, z))
        delta = new_facts - facts
        facts |= delta

    return facts

if __name__ == "__main__":
    for fact in solve():
        print(fact)
```

#### 3.2 Cython Code (.pyx)

```cython
# cython: language_level=3
from cpython.set cimport PySet_Add, PySet_Contains

cdef class Fact:
    cdef str relation
    cdef tuple args

    def __init__(self, str relation, tuple args):
        self.relation = relation
        self.args = args

    def __hash__(self):
        return hash((self.relation, self.args))

    def __eq__(self, other):
        return self.relation == other.relation and self.args == other.args

cpdef set solve():
    cdef set facts = set()
    cdef set delta = set()
    cdef Fact f

    facts.add(Fact("parent", ("john", "mary")))
    facts.add(Fact("parent", ("mary", "susan")))
    delta = set(facts)

    while delta:
        new_facts = apply_rules(facts, delta)
        delta = new_facts - facts
        facts |= delta

    return facts

cdef set apply_rules(set facts, set delta):
    cdef set result = set()
    # Rule implementations
    return result
```

#### 3.3 Numba Code

```python
import numpy as np
from numba import jit, njit
from numba.typed import List, Dict

@njit
def apply_rules_numba(facts_array: np.ndarray, delta_indices: np.ndarray) -> np.ndarray:
    """Apply rules on array-based fact representation."""
    new_facts = []
    for i in delta_indices:
        rel = facts_array[i, 0]
        if rel == 0:  # ancestor relation
            x, y = facts_array[i, 1], facts_array[i, 2]
            for j in range(len(facts_array)):
                if facts_array[j, 0] == 1 and facts_array[j, 1] == y:  # parent
                    new_facts.append([0, x, facts_array[j, 2]])
    return np.array(new_facts, dtype=np.int32) if new_facts else np.empty((0, 3), dtype=np.int32)

def solve():
    # Encode facts as numeric arrays for Numba
    facts = np.array([[1, 0, 1], [1, 1, 2]], dtype=np.int32)  # parent facts
    # ... fixpoint loop
    return facts
```

### 4. Performance Tests

#### 4.1 Benchmark Comparison

```bash
./tests/perf/test_accelerator_performance.sh
```

**Expected Speedups (vs CPython)**:
| Accelerator | Simple | 1000 Facts | 10000 Facts |
|-------------|--------|------------|-------------|
| Codon | 10-50x | 20-100x | 50-200x |
| Cython | 5-20x | 10-50x | 20-100x |
| mypyc | 3-10x | 5-20x | 10-40x |
| Nuitka | 1.5-3x | 2-5x | 2-5x |
| Numba | 10-100x | 50-200x | 100-500x |

## Test Commands Reference

### Quick Smoke Tests

```bash
# Codon
swipl -g "use_module('src/unifyweaver/targets/python_codon_target'), compile_to_codon(test, C), format('~w~n', [C])" -t halt

# Cython
swipl -g "use_module('src/unifyweaver/targets/python_cython_target'), compile_to_cython(test, C), format('~w~n', [C])" -t halt

# mypyc
swipl -g "use_module('src/unifyweaver/targets/python_mypyc_target'), compile_to_mypyc(test, C), format('~w~n', [C])" -t halt

# Nuitka
swipl -g "use_module('src/unifyweaver/targets/python_nuitka_target'), compile_to_nuitka(test, C), format('~w~n', [C])" -t halt

# Numba
swipl -g "use_module('src/unifyweaver/targets/python_numba_target'), compile_to_numba(test, C), format('~w~n', [C])" -t halt
```

### Full Test Suite

```bash
./tests/run_accelerator_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CODON_PATH` | Codon installation | (system) |
| `CYTHON_ANNOTATE` | Generate HTML annotations | `0` |
| `MYPYC_OPT_LEVEL` | Optimization level | `3` |
| `NUITKA_JOBS` | Parallel compilation jobs | `auto` |
| `NUMBA_THREADING_LAYER` | Threading backend | `tbb` |
| `SKIP_ACCELERATOR_TESTS` | Skip all accelerator tests | `0` |

## Known Issues

### Codon
1. **Python compatibility**: Not 100% Python compatible
2. **Library support**: Limited third-party library support
3. **Dynamic features**: No support for some dynamic Python features

### Cython
1. **Build complexity**: Requires C compiler setup
2. **Type annotations**: Manual type hints needed for speedup
3. **Debugging**: Harder to debug compiled code

### mypyc
1. **Type coverage**: Requires comprehensive type hints
2. **Dynamic code**: No support for highly dynamic code
3. **Incremental compilation**: Can be slow

### Nuitka
1. **Compilation time**: Slow for large projects
2. **Binary size**: Large standalone binaries
3. **Plugin issues**: Some plugins have compatibility issues

### Numba
1. **Limited Python**: Only supports a subset of Python
2. **NumPy required**: Best with NumPy-based code
3. **JIT overhead**: First call is slow

## Related Documentation

- [Codon Documentation](https://docs.exaloop.io/codon/)
- [Cython Documentation](https://cython.readthedocs.io/)
- [mypyc Documentation](https://mypyc.readthedocs.io/)
- [Nuitka User Manual](https://nuitka.net/doc/user-manual.html)
- [Numba Documentation](https://numba.readthedocs.io/)
