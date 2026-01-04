# Prolog Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Prolog code generation target testing

## Overview

This test plan covers the Prolog target for UnifyWeaver, which generates optimized Prolog code from UnifyWeaver queries. This target supports multiple Prolog dialects and generates efficient fixpoint computations.

## Prerequisites

### System Requirements

- SWI-Prolog 9.0+ (primary dialect)
- Optional: GNU Prolog, YAP, ECLiPSe for dialect tests
- UnifyWeaver repository cloned

### Verification

```bash
# Verify SWI-Prolog
swipl --version

# Optional dialect verification
gprolog --version 2>/dev/null || echo "GNU Prolog not installed"
yap --version 2>/dev/null || echo "YAP not installed"
eclipse --version 2>/dev/null || echo "ECLiPSe not installed"
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_prolog_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `fact_generation` | fact(a, b). | Fact syntax |
| `rule_generation` | head :- body. | Rule syntax |
| `query_generation` | ?- goal. | Query syntax |
| `dcg_rules` | phrase/2 | DCG syntax |
| `module_declaration` | :- module(...) | Module syntax |

#### 1.2 Dialect Support

```bash
swipl -g "use_module('tests/core/test_prolog_dialects'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `swipl_specific` | SWI extensions | SWI syntax |
| `iso_compatible` | ISO Prolog | Standard syntax |
| `gnu_prolog` | GNU Prolog | GProlog syntax |
| `constraint_handling` | CLP(FD) | Constraint syntax |

### 2. Compilation Tests

#### 2.1 Prolog Loading

```bash
./tests/integration/test_prolog_loading.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `consult_file` | [file]. | File loads |
| `use_module` | use_module/1 | Module loads |
| `syntax_check` | Compilation | No errors |
| `goal_execution` | ?- goal. | Goal succeeds |

### 3. Generated Code Structure

```prolog
:- module(generated_query, [
    solve/1,
    fact/2,
    rule/1
]).

:- use_module(library(ordsets)).
:- use_module(library(apply)).

%% fact(?Relation, ?Args) is nondet.
%  Base facts for the query.
fact(parent, [john, mary]).
fact(parent, [mary, susan]).
fact(parent, [bob, alice]).

%% derived(?Relation, ?Args) is nondet.
%  Derived facts via rule application.
:- dynamic derived/2.

%% apply_ancestor_rule(+Facts, +Delta, -NewFacts) is det.
%  Apply ancestor rule: ancestor(X,Z) :- parent(X,Y), ancestor(Y,Z).
apply_ancestor_rule(Facts, Delta, NewFacts) :-
    findall([X, Z],
        (   member([ancestor, [X, Y]], Delta),
            (   member([parent, [Y, Z]], Facts)
            ;   fact(parent, [Y, Z])
            )
        ),
        NewList),
    list_to_ord_set(NewList, NewFacts).

%% solve(-Results) is det.
%  Compute fixpoint of all rules.
solve(Results) :-
    findall([R, A], fact(R, A), InitFacts),
    list_to_ord_set(InitFacts, Facts0),
    fixpoint_loop(Facts0, Facts0, Results).

fixpoint_loop(Facts, Delta, Results) :-
    (   Delta == []
    ->  Results = Facts
    ;   apply_all_rules(Facts, Delta, NewFacts),
        ord_subtract(NewFacts, Facts, ActualNew),
        ord_union(Facts, ActualNew, Facts1),
        fixpoint_loop(Facts1, ActualNew, Results)
    ).

apply_all_rules(Facts, Delta, NewFacts) :-
    apply_ancestor_rule(Facts, Delta, New1),
    % Add more rule applications here
    New1 = NewFacts.
```

### 4. Dialect-Specific Tests

#### 4.1 SWI-Prolog Specifics

```bash
./tests/integration/test_swi_prolog.sh
```

**Test Cases**:
| Feature | Description | Expected |
|---------|-------------|----------|
| `tabling` | table/1 | Memoization works |
| `threads` | thread_create/3 | Threading works |
| `dicts` | _{key: val} | Dict syntax |
| `strings` | "string" | String handling |

#### 4.2 Constraint Tests

```bash
./tests/integration/test_prolog_constraints.sh
```

**Test Cases**:
| Constraint | Description | Expected |
|------------|-------------|----------|
| `clpfd` | #=/2, ins/2 | Integer constraints |
| `clpr` | {}/1 | Real constraints |
| `chr` | constraint/N | CHR rules |

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/prolog_target'),
    compile_to_prolog(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_prolog_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROLOG_DIALECT` | Target dialect | `swi` |
| `SWIPL_HOME` | SWI-Prolog home | (system) |
| `SKIP_PROLOG_EXECUTION` | Skip runtime tests | `0` |
| `PROLOG_OPTIMIZATION` | Optimization level | `1` |

## Known Issues

1. **Dialect differences**: Syntax varies between implementations
2. **Module systems**: Not standardized across dialects
3. **Constraint libraries**: Not available everywhere
4. **String handling**: Varies by implementation

## Related Documentation

- [SWI-Prolog Manual](https://www.swi-prolog.org/pldoc/)
- [ISO Prolog Standard](https://www.iso.org/standard/21413.html)
