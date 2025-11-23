# Python Target: Generator-Based Evaluation Mode

## Overview
Add an alternative evaluation strategy to the Python target that uses **semi-naive fixpoint iteration** like the C# query engine, complementing the existing procedural mode.

## Motivation

### Current: Procedural Mode
```python
# Factorial with recursion/memoization
@functools.cache
def _factorial_worker(arg):
    if arg == 0: return 1
    return arg * _factorial_worker(arg - 1)
```

**Pros**: Simple, direct, fast for deterministic recursion  
**Cons**: Deep recursion limits, harder to compose, no fixpoint

### Proposed: Generator Mode  
```python
# Semi-naive evaluation with delta/total sets
def _factorial_fixpoint(records):
    total = set()  # Facts discovered so far
    delta = set(records)  # New facts this iteration
    
    while delta:
        new_delta = set()
        for record in delta:
            # Apply rules, add new facts
            for result in _apply_rules(record, total):
                if result not in total:
                    new_delta.add(result)
                    total.add(result)
                    yield result
        delta = new_delta
```

**Pros**: Composable, supports transitive closure, no recursion limit, matches C# semantics  
**Cons**: Slower for simple cases, more memory

## Use Cases

| Pattern | Procedural | Generator |
|---------|------------|-----------|
| Factorial/Fibonacci | ✅ Best | ⚠️ Overkill |
| Tail recursion | ✅ Best (loops) | ⚠️ Inefficient |
| **Transitive closure** | ❌ Stack overflow | ✅ **Ideal** |
| **Graph queries** | ❌ Needs memoization | ✅ **Ideal** |
| **Recursive joins** | ⚠️ Complex | ✅ **Natural** |

## Architecture

### Mode Selection
```prolog
compile_predicate_to_python(Pred, Options, Code) :-
    option(mode(Mode), Options, procedural),  % Default: procedural
    (   Mode == generator
    ->  compile_generator_mode(Pred, Options, Code)
    ;   compile_procedural_mode(Pred, Options, Code)  % Current implementation
    ).
```

### Generator Mode Components

**1. Delta/Total Sets**
```python
def process_stream_generator(records: Iterator[Dict]) -> Iterator[Dict]:
    total: Set[FrozenDict] = set()  # All facts
    delta: Set[FrozenDict] = set(freeze_dict(r) for r in records)  # New facts
    
    while delta:
        new_delta = set()
        for fact in delta:
            # Apply each rule
            for new_fact in _apply_rule_1(fact, total):
                if new_fact not in total:
                    total.add(new_fact)
                    new_delta.add(new_fact)
                    yield unfreeze_dict(new_fact)
            # ... more rules
        delta = new_delta
```

**2. Rule Application**
```python
def _apply_rule_1(fact: FrozenDict, total: Set[FrozenDict]) -> Iterator[FrozenDict]:
    # Example: path(X,Z) :- edge(X,Y), path(Y,Z)
    if 'x' in fact and 'y' in fact:  # edge pattern
        # Join with existing path facts
        for other in total:
            if other.get('y') == fact.get('x'):  # path(Y,Z)
                yield freeze_dict({'x': fact['x'], 'z': other['z']})
```

**3. Frozen Dicts** (for set membership)
```python
class FrozenDict:
    def __init__(self, d):
        self._d = d
        self._hash = hash(frozenset(d.items()))
    def __hash__(self):
        return self._hash
    def __eq__(self, other):
        return self._d == other._d
```

## Implementation Plan

### Phase 1: Basic Infrastructure
- [ ] Add `mode` option to compile_predicate_to_python
- [ ] Create `compile_generator_mode/3`  
- [ ] Implement FrozenDict class
- [ ] Generate fixpoint iteration skeleton

### Phase 2: Rule Translation
- [ ] Translate Prolog clauses to Python rule functions
- [ ] Join detection (identify variables shared across goals)
- [ ] Pattern matching code generation
- [ ] Handle built-ins (is, >, <, etc.)

### Phase 3: Optimization
- [ ] Index delta sets by key for faster joins
- [ ] Semi-naive optimization (only join with delta, not total)
- [ ] Stratified negation support
- [ ] Magic sets transformation (optional)

### Phase 4: Testing
- [ ] Transitive closure test (ancestor/2)
- [ ] Graph reachability test
- [ ] Performance comparison vs procedural

## Example: Transitive Closure

**Prolog**:
```prolog
edge(a, b).
edge(b, c).
edge(c, d).

path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).
```

**Generated Python (Generator Mode)**:
```python
def process_stream_generator(records):
    total = set()
    delta = set(freeze_dict(r) for r in records)
    
    while delta:
        new_delta = set()
        for fact in delta:
            # Rule 1: path(X,Y) :- edge(X,Y)  
            if 'x' in fact and 'y' in fact and fact.get('_pred') == 'edge':
                yield from [{'x': fact['x'], 'y': fact['y'], '_pred': 'path'}]
           
            # Rule 2: path(X,Z) :- edge(X,Y), path(Y,Z)
            if 'x' in fact and 'y' in fact and fact.get('_pred') == 'edge':
                for other in total:
                    if (other.get('_pred') == 'path' and 
                        other.get('x') == fact['y']):  # Join condition
                        new_fact = freeze_dict({
                            'x': fact['x'],
                            'z': other['z'],
                            '_pred': 'path'
                        })
                        if new_fact not in total:
                            total.add(new_fact)
                            new_delta.add(new_fact)
                            yield unfreeze_dict(new_fact)
        delta = new_delta
```

## Decision Matrix

**Use Procedural Mode When:**
- Simple recursion (factorial, Fibonacci)
- Tail recursion (can use loops)
- Single-solution queries
- Performance critical

**Use Generator Mode When:**
- Transitive closure
- Graph algorithms  
- Recursive joins
- Need composability
- Unknown recursion depth

## Compatibility

- Both modes support same input/output: JSONL streaming
- Can mix modes in same system (different predicates)
- Generator mode is **opt-in** via mode option

## Future: Hybrid Mode

Automatically choose best mode:
```prolog
compile_predicate_to_python(Pred, Options, Code) :-
    option(mode(Mode), Options, auto),
    (   Mode == auto
    ->  (   is_transitive_closure(Pred)
        ->  ActualMode = generator
        ;   is_tail_recursive(Pred)  
        ->  ActualMode = procedural
        ;   ActualMode = procedural  % Default
        ),
        compile_with_mode(Pred, ActualMode, Options, Code)
    ;   compile_with_mode(Pred, Mode, Options, Code)
    ).
```

## References
- C# Query Engine (`csharp_query_target.pl`) - Similar semi-naive approach
- Datalog evaluation strategies
- Magic sets transformation papers
