# Typed R / TypR Return Types

This example shows how `uw_return_type/2` affects both the plain `r` target
and the `typr` target.

## Example Source

```prolog
:- use_module(src/unifyweaver/targets/type_declarations).

lower_name(Name, Lower) :-
    string_lower(Name, Lower),
    true.

uw_type(lower_name/2, 1, atom).
uw_type(lower_name/2, 2, atom).
uw_return_type(lower_name/2, atom).
```

## TypR Behavior

Compile:

```prolog
?- compile_predicate_to_typr(lower_name/2, [typed_mode(explicit)], Code).
```

Result:

- TypR emits a concrete return type such as `char`
- it does not need to fall back to `Any`

The important part of the generated signature becomes:

```typr
let lower_name <- fn(arg1: char, arg2: char): char {
    ...
};
```

## Plain R Behavior

Compile:

```prolog
?- compile_predicate_to_r(lower_name/2, [], Code).
```

Standard R remains syntactically untyped, but the declared return type still
matters. The compiler can use it for:

- result-shape fallback generation
- simple compatibility checks when the body's return type is inferable

## Disabling Type-Constrained R Fallbacks

If you want to inspect unconstrained R output, disable type constraints:

```prolog
?- compile_predicate_to_r(lower_name/2, [type_constraints(false)], Code).
```

This restores the older behavior where return-type metadata does not
participate in branch filtering or fallback selection.

## Optional Diagnostics

Diagnostics are optional and off by default.

Warn but continue:

```prolog
?- compile_predicate_to_r(my_pred/2, [type_diagnostics(warn)], Code).
```

Throw on violation:

```prolog
?- compile_predicate_to_r(my_pred/2, [type_diagnostics(error)], Code).
```

Recommended usage:

- `off` for normal compilation
- `warn` when auditing generated R behavior
- `error` when a return-type rule is meant to be an enforced compiler invariant

## Why This Matters

`uw_return_type/2` gives both targets a shared declarative statement about what
the predicate is supposed to produce:

- `typr` uses it for stronger signatures
- `r` uses it for safer fallback behavior

This improves the R family without making standard R depend on type metadata.
