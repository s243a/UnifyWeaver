# Transitive Closure Tutorial

## Introduction

This tutorial demonstrates **transitive closure** computation using UnifyWeaver's Python target in **generator mode**. We'll compute ancestor relationships from parent facts, showing how generator mode naturally handles recursive graph queries.

## What is Transitive Closure?

**Transitive closure** finds all reachable paths in a graph. If A→B and B→C, then A→C is in the transitive closure.

**Real-world examples:**
- **Ancestor trees**: Who are all my ancestors?
- **Organization charts**: Who reports to CEO (directly or indirectly)?
- **Package dependencies**: What packages does my app transitively depend on?

## The Problem

Given parent relationships:
```
john → mary
mary → sue  
sue → alice
john → bob
```

**Query**: Who are john's ancestors (children, grandchildren, etc.)?

**Answer**: mary, sue, alice, bob (all 4 descendants)

## Why Generator Mode?

**Procedural mode** would stack overflow on deep family trees.  
**Generator mode** uses fixpoint iteration - no recursion depth limit!

## Step 1: Define the Prolog Rules

Create `family.pl`:

```prolog
% Facts - direct parent relationships
parent(john, mary).
parent(mary, sue).
parent(sue, alice).
parent(john, bob).

% Rules - transitive closure
ancestor(X, Y) :- parent(X, Y).                    % Base case
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).    % Recursive case
```

## Step 2: Compile to Python (Generator Mode)

```prolog
:- use_module('src/unifyweaver/targets/python_target').

compile_ancestor :-
    % Load facts and rules
    consult('family.pl'),
    
    % Compile in generator mode
    compile_predicate_to_python(
        ancestor/2,
        [mode(generator), record_format(jsonl)],
        PythonCode
    ),
    
    % Save to file
    open('ancestor.py', write, Stream),
    write(Stream, PythonCode),
    close(Stream),
    
    writeln('✓ Compiled ancestor.py in generator mode').

?- compile_ancestor.
```

**Run:**
```bash
swipl -l family_compile.pl
```

**Output:**
```
✓ Compiled ancestor.py in generator mode
```

## Step 3: Understand the Generated Code

**Key components in `ancestor.py`:**

### 1. FrozenDict (for set membership)
```python
@dataclass(frozen=True)
class FrozenDict:
    '''Hashable dictionary for use in sets'''
    items: tuple
```

### 2. Rule 1 - Base Case
```python
def _apply_rule_1(fact, total):
    '''ancestor(X,Y) :- parent(X,Y)'''
    if 'arg0' in fact and 'arg1' in fact:
        yield FrozenDict.from_dict({
            'arg0': fact.get('arg0'),
            'arg1': fact.get('arg1')
        })
```

### 3. Rule 2 - Recursive Join
```python
def _apply_rule_2(fact, total):
    '''ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z)'''
    if 'arg0' in fact and 'arg1' in fact:
        for other in total:
            if other.get('arg0') == fact.get('arg1'):  # Join on Y
                yield FrozenDict.from_dict({
                    'arg0': fact.get('arg0'),
                    'arg1': other.get('arg1')
                })
```

### 4. Fixpoint Loop
```python
def process_stream_generator(records):
    total = set()
    delta = set(FrozenDict.from_dict(r) for r in records)
    
    while delta:  # Iterate until no new facts
        new_delta = set()
        for fact in delta:
            for new_fact in _apply_rule_1(fact, total):
                if new_fact not in total:
                    total.add(new_fact)
                    new_delta.add(new_fact)
                    yield new_fact.to_dict()
            # ... same for rule 2
        delta = new_delta
```

## Step 4: Prepare Input Data

Create `parents.jsonl` (parent facts):
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
{"arg0": "sue", "arg1": "alice"}
{"arg0": "john", "arg1": "bob"}
```

## Step 5: Run the Python Code

```bash
python3 ancestor.py < parents.jsonl
```

**Output:**
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
{"arg0": "sue", "arg1": "alice"}
{"arg0": "john", "arg1": "bob"}
{"arg0": "john", "arg1": "sue"}
{"arg0": "mary", "arg1": "alice"}
{"arg0": "john", "arg1": "alice"}
```

**Interpretation:**
- First 4 lines: Direct parents (echoed from input)
- Line 5: `john → sue` (grandchild)
- Line 6: `mary → alice` (grandchild)
- Line 7: `john → alice` (great-grandchild)

**✓ Complete transitive closure computed!**

## Step 6: Visualize the Iterations

**Iteration 0** (Input):
```
parent(john, mary)
parent(mary, sue)
parent(sue, alice)
parent(john, bob)
```

**Iteration 1** (Rule 1 - copy to ancestor):
```
ancestor(john, mary)    ← from parent(john, mary)
ancestor(mary, sue)     ← from parent(mary, sue)
ancestor(sue, alice)    ← from parent(sue, alice)
ancestor(john, bob)     ← from parent(john, bob)
```

**Iteration 2** (Rule 2 - join parent × ancestor):
```
ancestor(john, sue)     ← parent(john, mary) + ancestor(mary, sue)
ancestor(mary, alice)   ← parent(mary, sue) + ancestor(sue, alice)
```

**Iteration 3** (Rule 2 - another level):
```
ancestor(john, alice)   ← parent(john, mary) + ancestor(mary, alice)
```

**Iteration 4**: No new facts → **Fixpoint reached!** ✓

## Comparison: Procedural vs Generator

### Procedural Mode (Would Fail)

```prolog
compile_predicate_to_python(ancestor/2, [mode(procedural)], Code).
```

**Problem:**
- Deep family trees → stack overflow
- Python recursion limit: ~1000 levels
- Not suitable for arbitrary graphs

### Generator Mode (Works Perfectly)

```prolog
compile_predicate_to_python(ancestor/2, [mode(generator)], Code).
```

**Benefits:**
- ✅ No recursion depth limit
- ✅ Guaranteed termination (fixpoint)
- ✅ Handles cycles gracefully
- ✅ Natural for graph algorithms

## Real-World Example: Organization Chart

**Scenario**: Who reports to the CEO?

```prolog
% Direct reports
reports_to(alice, bob).
reports_to(bob, carol).
reports_to(carol, dave_ceo).
reports_to(eve, bob).

% Transitive closure
reports_chain(X, Y) :- reports_to(X, Y).
reports_chain(X, Z) :- reports_to(X, Y), reports_chain(Y, Z).
```

**Query**: `reports_chain(alice, dave_ceo)?`  
**Answer**: Yes! alice → bob → carol → dave_ceo

**Generator mode finds all indirect reporting relationships efficiently!**

## Performance Characteristics

| Metric | Procedural | Generator |
|--------|------------|-----------|
| Max depth | ~1000 | Unlimited |
| Memory | Low (streaming) | Medium (stores total set) |
| Speed (shallow) | Fast | Slower |
| Speed (deep) | ❌ Crashes | ✓ Works |
| Cycles | ❌ Infinite loop | ✓ Handles gracefully |

**Rule of thumb:**
- Shallow recursion (< 100 levels) → Procedural (faster)
- Deep/unknown depth → Generator (reliable)

## Advanced: Query Specific Nodes

Want only john's descendants?

**Filter input:**
```bash
echo '{"arg0": "john", "arg1": "mary"}
{"arg0": "mary", "arg1": "sue"}
{"arg0": "sue", "arg1": "alice"}
{"arg0": "john", "arg1": "bob"}' | python3 ancestor.py | jq 'select(.arg0 == "john")'
```

**Output:**
```json
{"arg0": "john", "arg1": "mary"}
{"arg0": "john", "arg1": "bob"}
{"arg0": "john", "arg1": "sue"}
{"arg0": "john", "arg1": "alice"}
```

**john's descendants: mary, bob, sue, alice** ✓

## Troubleshooting

### Issue: "No output"

**Cause**: Input format mismatch.

**Solution**: Verify JSONL format (newline-delimited, not array).

### Issue: "Incorrect results"

**Cause**: Rules not loaded properly.

**Solution**: Check `consult('family.pl')` ran before compilation.

### Issue: "Python error"

**Cause**: Python < 3.7.

**Solution**: Upgrade to Python 3.7+ (need type hints support).

## Summary

You've learned:
1. ✅ What transitive closure is
2. ✅ When to use generator mode
3. ✅ How to compile Prolog to Python (generator)
4. ✅ How fixpoint iteration works
5. ✅ Real-world applications

**Generator mode makes graph algorithms easy!**

## Next Steps

**Try it yourself:**
1. Modify the family tree (add more people)
2. Change to organization chart
3. Try package dependencies
4. Experiment with cycles (add `parent(alice, john)`)

**Advanced topics:**
- N-way joins (3+ predicates)
- Stratified negation
- Aggregation
- Optimization (indexing)

## Resources

- [Python Target Guide](../guides/python_target.md) - Complete API reference
- [Generator Mode Design](../proposals/python_generator_mode.md) - Implementation details
- [C# Query Target](../guides/csharp_query_target.md) - Similar semi-naive approach

---

**Congratulations!** You've mastered transitive closure with UnifyWeaver! 🎉
