# TypR Wrapped Fallback Boundary

## Current State

Most high-traffic raw-R seams in the TypR target have already been reduced:

- transitive closure control flow and loaders
- recursive loop bodies
- single-predicate helper bodies
- mutual wrapper/helper bodies
- fact-only boolean paths
- fixed-arity unary I/O commands such as `cat/1` and `print/1`

The main remaining generic raw-R seam is now the wrapped fallback path in:

- [src/unifyweaver/targets/typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/src/unifyweaver/targets/typr_target.pl:7334)
- [src/unifyweaver/targets/typr_target.pl](/home/s243a/Projects/UnifyWeaver/context/antigravity/UnifyWeaver/src/unifyweaver/targets/typr_target.pl:8610)

That seam is no longer “template noise.” It is the boundary for generic bodies
that still rely on the R backend to express their body semantics.

## Audit Result

The first plausible promotion candidates were generic bodies shaped like:

- `string_lower(Name, Lower), Out = Lower`
- `string_length(Name, Len), Out is Len + 1`
- `string_lower(Name, Lower), string_upper(Lower, Upper), Out = Upper`

Those look close to the existing native assignment-tail logic, but they do not
currently enter the native generic path for the right reason:

- the assignment tail itself is simple
- the earlier producer goals such as `string_lower/2` and `string_length/2`
  are not on the native generic TypR binding path

So promoting only the tail assignment is the wrong abstraction. The controlling
boundary is the earlier generic producer step, not the final `=` or `is/2`.

## Practical Boundary

For now, treat the wrapped fallback as intentional for generic bodies whose
main value-producing steps are only available through the R-backed lowering.

That includes the current string-transform/output-tail shapes above.

## Cat / Print Binding Constraint

TypR-safe I/O bindings should remain explicit about fixed arity:

- `cat/1` is fine
- `print/1` is fine
- variadic-style `cat("x =", x)` should not be modeled as a native TypR
  variadic binding

Preferred shape:

```typr
let x <- 42;
cat(paste("x =", x));
```

## Recommended Next Step

If this fallback seam is revisited, start by auditing the producer-goal layer,
not the assignment tail:

1. classify wrapped generic bodies by the first non-native producer goal
2. promote only the first producer family that can be expressed cleanly in
   native TypR
3. if the producer remains fundamentally R-shaped, keep the wrapped fallback
   and document it instead of forcing fake-native output
