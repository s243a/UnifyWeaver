# Prolog Constraint Handling

**Status:** ✅ Implemented in v0.1
**Module:** `src/unifyweaver/targets/prolog_constraints.pl`

## Overview

The Prolog target now supports **constraint enforcement** through dialect-specific code transformations. When you specify constraints like `unique(true)`, the system generates appropriate Prolog code to honor them.

## Key Features

### 1. Tabling for SWI-Prolog (Native)

SWI-Prolog's **tabling** provides automatic memoization and deduplication:

```prolog
% Input: Compile with unique constraint
generate_prolog_script(
    [ancestor/2],
    [dialect(swi), constraints([unique(true)])],
    Code
).

% Generated Output:
:- table ancestor/2.

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

**Benefits:**
- ✅ Native Prolog feature (efficient)
- ✅ Automatic deduplication
- ✅ Memoization (avoids recomputation)
- ✅ Idiomatic and clean

### 2. Wrapper for GNU Prolog

GNU Prolog doesn't have tabling, so we generate a wrapper using `setof`:

```prolog
% Generated Output:
ancestor_impl(X, Y) :- parent(X, Y).
ancestor_impl(X, Y) :- parent(X, Z), ancestor_impl(Z, Y).

ancestor(X, Y) :-
    setof([X, Y], ancestor_impl(X, Y), Solutions),
    member([X, Y], Solutions).
```

**Trade-offs:**
- ✅ Guarantees uniqueness
- ❌ Collects all solutions before returning (not lazy)
- ❌ Won't work for infinite predicates

### 3. Configurable Failure Modes

Control what happens when a constraint can't be satisfied:

```prolog
% Default: Fail compilation (safest)
:- set_constraint_failure_mode(fail).

% Alternative: Warn but continue
:- set_constraint_failure_mode(warn).

% Alternative: Throw error
:- set_constraint_failure_mode(error).

% Alternative: Silently ignore
:- set_constraint_failure_mode(ignore).
```

**Recommended:** `fail` (default) - explicit is better than implicit

## Configuration API

### Constraint Modes

```prolog
% Use native features (tabling for SWI, wrapper for GNU)
:- set_constraint_mode(unique, native).

% Always use wrapper (even on SWI)
:- set_constraint_mode(unique, wrapper).

% Don't enforce (document only)
:- set_constraint_mode(unique, ignore).

% Fail if can't satisfy
:- set_constraint_mode(unique, fail).
```

### Failure Modes

```prolog
% Set global failure handling
:- set_constraint_failure_mode(fail).    % Fail compilation
:- set_constraint_failure_mode(warn).    % Warn and continue
:- set_constraint_failure_mode(error).   % Throw error
:- set_constraint_failure_mode(ignore).  % Silent ignore
```

## Usage Examples

### Example 1: Transitive Closure with Uniqueness

```prolog
:- use_module('src/unifyweaver/targets/prolog_target').

% Define source predicate
parent(tom, bob).
parent(bob, ann).

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Compile with unique constraint
?- generate_prolog_script(
      [ancestor/2],
      [dialect(swi), constraints([unique(true)])],
      Code
   ),
   write_prolog_script(Code, 'output/ancestor.pl').

% Generated file uses tabling:
% :- table ancestor/2.
% ancestor(X, Y) :- parent(X, Y).
% ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

### Example 2: Cross-Dialect Consistency

```prolog
% Same source, different dialects

% For SWI-Prolog (development)
generate_prolog_script(
    [ancestor/2],
    [dialect(swi), constraints([unique(true)])],
    SwiCode
).
% Uses: tabling

% For GNU Prolog (production binary)
generate_prolog_script(
    [ancestor/2],
    [dialect(gnu), constraints([unique(true)])],
    GnuCode
).
% Uses: setof wrapper
```

### Example 3: Safe Compilation with Failure Mode

```prolog
% Ensure constraints are satisfied or fail
:- set_constraint_failure_mode(fail).

% This succeeds (tabling available on SWI)
?- generate_prolog_script(
      [ancestor/2],
      [dialect(swi), constraints([unique(true)])],
      Code
   ).
Code = ...  % Contains tabling directive

% This would also succeed (wrapper for GNU)
?- generate_prolog_script(
      [ancestor/2],
      [dialect(gnu), constraints([unique(true)])],
      Code
   ).
Code = ...  % Contains setof wrapper

% Hypothetical: unsupported constraint would FAIL
% (with current implementation, unique is always satisfiable)
```

### Example 4: Warn Mode for Flexibility

```prolog
% Want to continue even if constraint can't be satisfied
:- set_constraint_failure_mode(warn).

% Compilation proceeds with warning
?- generate_prolog_script(
      [my_pred/2],
      [dialect(unknown), constraints([unique(true)])],
      Code
   ).
% [Warning] Cannot satisfy unique(true) for my_pred/2 on dialect unknown
% Continuing without constraint enforcement.
Code = ...  % Verbatim copy without constraint
```

## Supported Constraints

| Constraint | SWI-Prolog | GNU Prolog | Notes |
|-----------|-----------|-----------|-------|
| `unique(true)` | Tabling | setof wrapper | Deduplicates results |
| `unique(false)` | Natural | Natural | No deduplication needed |
| `unordered(true)` | Natural | Natural | Prolog naturally unordered |
| `unordered(false)` | Natural | Natural | Order preserved |
| `optimization(_)` | Hint only | Hint only | Not enforced |

## Constraint Satisfaction Logic

```prolog
% Check if constraint can be satisfied for a dialect
?- constraint_satisfied(unique(true), swi, Method).
Method = tabling.

?- constraint_satisfied(unique(true), gnu, Method).
Method = wrapper.

?- constraint_satisfied(unordered(true), swi, Method).
Method = yes.  % Naturally satisfied
```

## Integration with Preferences

Constraints work seamlessly with the preference system:

```prolog
% Global default: prefer Bash, fall back to Prolog with constraints
:- assertz(preferences:preferences_default([
    prefer([bash]),
    fallback_order([prolog])
])).

% Predicate-specific: enforce unique constraint
:- assertz(preferences:rule_preferences(ancestor/2, [
    constraints([unique(true)])
])).

% When Bash compilation fails, falls back to Prolog with tabling/wrapper
```

## Performance Considerations

### Tabling (SWI-Prolog)
- ✅ Very efficient (native implementation)
- ✅ Lazy evaluation (solutions produced on demand)
- ✅ Memoization speeds up repeated calls
- ✅ Works with infinite predicates

### Wrapper (GNU Prolog)
- ⚠️ Collects all solutions first (eager)
- ⚠️ Memory overhead for large result sets
- ❌ Won't work for infinite predicates
- ✅ Correct for finite result sets

**Recommendation:** Use SWI-Prolog for development (tabling), GNU Prolog for production binaries (compiled).

## Testing

Run the constraint test suite:

```bash
cd /path/to/UnifyWeaver
swipl -l examples/test_prolog_constraints.pl
```

Tests include:
- ✅ Constraint satisfaction checking
- ✅ SWI-Prolog tabling generation
- ✅ GNU Prolog wrapper generation
- ✅ Verbatim copy (no constraints)
- ✅ Failure mode configuration
- ✅ Constraint mode configuration
- ✅ Multiple constraints
- ✅ Real-world examples

## Philosophy

**Default behavior: Fail-fast**

When a constraint can't be satisfied, **fail compilation by default**. This ensures:
- No silent bugs from ignored constraints
- Explicit handling of edge cases
- Consistent behavior across targets
- User awareness of limitations

**Override when needed:**
- Use `warn` mode for exploratory development
- Use `ignore` mode for documentation-only constraints
- Use `error` mode for strict validation

## See Also

- [Prolog Dialect System](PROLOG_DIALECTS.md)
- [Prolog as Fallback](PROLOG_AS_FALLBACK.md)
- [Prolog Templates Design](PROLOG_TEMPLATES_DESIGN.md)
- [Constraint System Implementation](../src/unifyweaver/targets/prolog_constraints.pl)
