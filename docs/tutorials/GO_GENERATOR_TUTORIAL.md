# Go Generator Mode Tutorial

This tutorial walks through using Go generator mode for Datalog evaluation.

## Prerequisites

- Go installed (`go version`)
- SWI-Prolog with UnifyWeaver loaded

---

## Example 1: Transitive Closure (Ancestor)

The classic recursive Datalog example.

### Define the Program

```prolog
% Load UnifyWeaver
:- use_module('src/unifyweaver/targets/go_target').

% Define facts
parent(john, mary).
parent(mary, sue).
parent(sue, alice).

% Define rules
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile to Go
:- compile_predicate_to_go(ancestor/2, [mode(generator)], Code),
   open('ancestor.go', write, S),
   write(S, Code),
   close(S).
```

### Run

```bash
go run ancestor.go
```

### Expected Output

```json
{"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
{"relation":"parent","args":{"arg0":"mary","arg1":"sue"}}
{"relation":"parent","args":{"arg0":"sue","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"sue","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"mary","arg1":"sue"}}
{"relation":"ancestor","args":{"arg0":"mary","arg1":"alice"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"mary"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"sue"}}
{"relation":"ancestor","args":{"arg0":"john","arg1":"alice"}}
```

---

## Example 2: Graph Reachability with Negation

Find paths that aren't blocked.

```prolog
edge(a, b).
edge(b, c).
edge(c, d).
blocked(b, c).

path(X, Y) :- edge(X, Y), \+ blocked(X, Y).
path(X, Z) :- edge(X, Y), \+ blocked(X, Y), path(Y, Z).

:- compile_predicate_to_go(path/2, [mode(generator)], Code), ...
```

---

## Example 3: Aggregation with HAVING

Calculate department totals, filter high earners.

```prolog
salary(eng, 1000).
salary(eng, 1500).
salary(sales, 800).
salary(sales, 1700).
salary(hr, 600).

% Sum by department, only where total > 1000
dept_high(Dept, Total) :- 
    aggregate_all(sum(S), salary(Dept, S), Dept, Total),
    Total > 1000.

:- compile_predicate_to_go(dept_high/2, [mode(generator)], Code), ...
```

### Output

```json
{"relation":"dept_high","args":{"arg0":"eng","arg1":2500}}
{"relation":"dept_high","args":{"arg0":"sales","arg1":2500}}
```

(hr excluded because 600 < 1000)

---

## Example 4: Loading Data from Stdin

Instead of hardcoding facts, pipe them in.

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

:- compile_predicate_to_go(ancestor/2, [mode(generator), json_input(true)], Code), ...
```

### Run with Piped Data

```bash
echo '{"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
{"relation":"parent","args":{"arg0":"mary","arg1":"sue"}}' | go run ancestor.go
```

---

## Example 5: Parallel Execution

Speed up large computations with goroutines.

```prolog
:- compile_predicate_to_go(ancestor/2, [mode(generator), workers(4)], Code), ...
```

The fixpoint loop distributes facts across 4 workers.

---

## Example 6: Incremental Computation with Database

Persist results to bbolt database.

```prolog
:- compile_predicate_to_go(ancestor/2, [mode(generator), db_backend(bbolt), 
                                         db_file('ancestry.db')], Code), ...
```

### Workflow

```bash
# First run - computes and saves to ancestry.db
./ancestor

# Add more facts via stdin, continue from saved state
echo '{"relation":"parent","args":{"arg0":"alice","arg1":"bob"}}' | ./ancestor
```

---

## Combining Options

All options can be combined:

```prolog
:- compile_predicate_to_go(ancestor/2, [
    mode(generator),
    json_input(true),      % Load facts from stdin
    workers(4),            % 4 parallel workers
    db_backend(bbolt),     % Persist to database
    db_file('data.db')
], Code), ...
```

---

## Tips

1. **Filter output by relation**:
   ```bash
   go run ancestor.go | jq 'select(.relation == "ancestor")'
   ```

2. **Pretty print**:
   ```bash
   go run ancestor.go | jq .
   ```

3. **Count results**:
   ```bash
   go run ancestor.go | wc -l
   ```

4. **Performance**: For large datasets, use `workers(N)` where N = number of CPU cores.
