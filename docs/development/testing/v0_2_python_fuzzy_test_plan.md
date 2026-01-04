# Python Fuzzy Logic Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Python fuzzy logic code generation target testing

## Overview

This test plan covers the Python fuzzy logic target for UnifyWeaver, which generates Python code that implements fuzzy Datalog semantics. Instead of binary true/false values, fuzzy Datalog uses confidence values in [0,1] range, enabling probabilistic reasoning and approximate matching.

## Prerequisites

### System Requirements

- Python 3.10+
- NumPy (for fuzzy operations)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Optional Libraries

- scikit-fuzzy (for advanced fuzzy operations)
- matplotlib (for visualization)

### Verification

```bash
# Verify Python
python3 --version

# Verify NumPy
python3 -c "import numpy; print(numpy.__version__)"

# Optional: scikit-fuzzy
python3 -c "import skfuzzy; print(skfuzzy.__version__)" 2>/dev/null || echo "scikit-fuzzy not installed"

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_fuzzy_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `fuzzy_fact` | Fact with confidence | (fact, 0.8) |
| `fuzzy_rule` | Rule with t-norm | min/product |
| `threshold_filter` | Confidence threshold | Filtering |
| `aggregation` | Multiple derivations | max/sum |

#### 1.2 Fuzzy Operations

```bash
swipl -g "use_module('tests/core/test_fuzzy_operations'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `t_norm_min` | min(a, b) | Gödel t-norm |
| `t_norm_product` | a * b | Product t-norm |
| `t_norm_lukasiewicz` | max(0, a+b-1) | Łukasiewicz |
| `s_norm_max` | max(a, b) | Gödel s-norm |
| `negation` | 1 - a | Standard negation |

### 2. Compilation Tests

#### 2.1 Python Execution

```bash
./tests/integration/test_fuzzy_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `python_run` | python script.py | Executes |
| `numpy_ops` | NumPy fuzzy ops | Correct values |
| `threshold_output` | Filter by threshold | Correct filtering |

### 3. Generated Code Structure

```python
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Callable
import numpy as np

@dataclass(frozen=True)
class FuzzyFact:
    relation: str
    args: tuple
    confidence: float

    def __hash__(self):
        return hash((self.relation, self.args))

    def __eq__(self, other):
        # Equality ignores confidence for set membership
        return self.relation == other.relation and self.args == other.args

class FuzzyTNorm:
    """T-norm implementations for fuzzy conjunction."""

    @staticmethod
    def minimum(a: float, b: float) -> float:
        """Gödel t-norm: min(a, b)"""
        return min(a, b)

    @staticmethod
    def product(a: float, b: float) -> float:
        """Product t-norm: a * b"""
        return a * b

    @staticmethod
    def lukasiewicz(a: float, b: float) -> float:
        """Łukasiewicz t-norm: max(0, a + b - 1)"""
        return max(0.0, a + b - 1.0)

class FuzzySNorm:
    """S-norm implementations for fuzzy disjunction."""

    @staticmethod
    def maximum(a: float, b: float) -> float:
        """Gödel s-norm: max(a, b)"""
        return max(a, b)

    @staticmethod
    def probabilistic_sum(a: float, b: float) -> float:
        """Probabilistic sum: a + b - a*b"""
        return a + b - a * b

class FuzzyQuery:
    def __init__(self,
                 t_norm: Callable[[float, float], float] = FuzzyTNorm.minimum,
                 s_norm: Callable[[float, float], float] = FuzzySNorm.maximum,
                 threshold: float = 0.0):
        self.t_norm = t_norm
        self.s_norm = s_norm
        self.threshold = threshold
        self.facts: Dict[Tuple[str, tuple], float] = {}

    def add_fact(self, relation: str, args: tuple, confidence: float):
        """Add or update a fuzzy fact."""
        key = (relation, args)
        if key in self.facts:
            # Aggregate using s-norm for multiple derivations
            self.facts[key] = self.s_norm(self.facts[key], confidence)
        else:
            self.facts[key] = confidence

    def get_confidence(self, relation: str, args: tuple) -> float:
        """Get confidence of a fact, 0.0 if not present."""
        return self.facts.get((relation, args), 0.0)

    def init_facts(self):
        # Base facts with confidence values
        self.add_fact("parent", ("john", "mary"), 1.0)
        self.add_fact("parent", ("mary", "susan"), 0.95)
        self.add_fact("similar", ("john", "jon"), 0.85)

    def apply_rules(self):
        """Apply fuzzy rules and return new derivations."""
        new_derivations = []

        # ancestor(X, Z) :- parent(X, Y), parent(Y, Z).
        # Confidence = t_norm(conf(parent(X,Y)), conf(parent(Y,Z)))
        for (rel1, args1), conf1 in self.facts.items():
            if rel1 == "parent":
                x, y = args1
                for (rel2, args2), conf2 in self.facts.items():
                    if rel2 == "parent" and args2[0] == y:
                        z = args2[1]
                        derived_conf = self.t_norm(conf1, conf2)
                        if derived_conf >= self.threshold:
                            new_derivations.append(("ancestor", (x, z), derived_conf))

        return new_derivations

    def solve(self) -> Dict[Tuple[str, tuple], float]:
        """Compute fuzzy fixpoint."""
        self.init_facts()

        changed = True
        while changed:
            changed = False
            for relation, args, confidence in self.apply_rules():
                key = (relation, args)
                old_conf = self.facts.get(key, 0.0)
                new_conf = self.s_norm(old_conf, confidence)
                if abs(new_conf - old_conf) > 1e-9:
                    self.facts[key] = new_conf
                    changed = True

        # Filter by threshold
        return {k: v for k, v in self.facts.items() if v >= self.threshold}

def main():
    # Use product t-norm and probabilistic s-norm
    query = FuzzyQuery(
        t_norm=FuzzyTNorm.product,
        s_norm=FuzzySNorm.probabilistic_sum,
        threshold=0.5
    )

    results = query.solve()

    print("Fuzzy Datalog Results (threshold >= 0.5):")
    for (relation, args), confidence in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {relation}({', '.join(args)}): {confidence:.3f}")

if __name__ == "__main__":
    main()
```

### 4. Fuzzy Semantics Tests

#### 4.1 T-Norm Variations

```bash
./tests/integration/test_fuzzy_tnorms.sh
```

**Test Cases**:
| T-Norm | Input (0.8, 0.6) | Expected |
|--------|------------------|----------|
| min | min(0.8, 0.6) | 0.6 |
| product | 0.8 * 0.6 | 0.48 |
| Łukasiewicz | max(0, 1.4-1) | 0.4 |

#### 4.2 S-Norm Variations

```bash
./tests/integration/test_fuzzy_snorms.sh
```

**Test Cases**:
| S-Norm | Input (0.8, 0.6) | Expected |
|--------|------------------|----------|
| max | max(0.8, 0.6) | 0.8 |
| prob_sum | 0.8+0.6-0.48 | 0.92 |
| bounded_sum | min(1, 1.4) | 1.0 |

### 5. Performance Tests

#### 5.1 Fuzzy Computation Performance

```bash
./tests/perf/test_fuzzy_performance.sh
```

**Benchmarks**:
| Test | Facts | Expected Time |
|------|-------|---------------|
| Simple | 100 | < 50ms |
| Medium | 1000 | < 500ms |
| Large | 10000 | < 5s |

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/python_fuzzy_target'),
    compile_to_fuzzy_python(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_fuzzy_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FUZZY_T_NORM` | Default t-norm | `min` |
| `FUZZY_S_NORM` | Default s-norm | `max` |
| `FUZZY_THRESHOLD` | Default threshold | `0.0` |
| `SKIP_FUZZY_EXECUTION` | Skip runtime tests | `0` |

## Known Issues

1. **Floating point**: Precision issues near boundaries
2. **Convergence**: Some t-norms may converge slowly
3. **Memory**: Storing all confidence values can be expensive
4. **Semantics**: Different t-norms give different results

## Related Documentation

- [Fuzzy Logic Overview](https://en.wikipedia.org/wiki/Fuzzy_logic)
- [T-norm Reference](https://en.wikipedia.org/wiki/T-norm)
- [scikit-fuzzy Documentation](https://pythonhosted.org/scikit-fuzzy/)
