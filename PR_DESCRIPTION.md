# feat(targets): Add Scala target with bindings and LazyList generator

## Summary

Adds a complete Scala target with pipeline mode (Option/flatMap), generator mode (LazyList), and 43 bindings leveraging Scala's functional features.

## Changes

### New Files

#### `src/unifyweaver/targets/scala_target.pl`
**Three compilation modes:**
- **Simple mode** - Basic predicate translation
- **Pipeline mode** - `Option[Record]` with `flatMap` for filtering
- **Generator mode** - `LazyList` with `#::` for lazy sequences

**Features:**
- Pattern matching in JSON value handling
- `@tailrec` annotation for tail-recursive predicates
- SBT build generation

#### `src/unifyweaver/bindings/scala_bindings.pl`
43 bindings across 5 categories:
- **Option/Either:** Some, None, getOrElse, map, flatMap, filter, Right, Left
- **Collections:** List, Nil, ::, Map, head, tail, foldLeft, foldRight
- **Strings:** length, substring, split, trim, toLowerCase
- **LazyList:** LazyList.from, #::, take, drop, takeWhile, dropWhile
- **Pattern matching:** match, case class, unapply

### Modified Files

#### `docs/JVM_TARGET.md`
- Updated Scala status to "pipeline + generator modes"

#### `docs/BINDING_MATRIX.md`
- Added Scala (43 bindings, 5 categories)

## Testing

```bash
# Target tests
swipl -g "use_module('src/unifyweaver/targets/scala_target'),
    test_scala_pipeline_mode, halt(0)"
# Output: 5/5 passed (pipeline, pattern matching, Option, LazyList, SBT)

# Bindings tests  
swipl -g "use_module('src/unifyweaver/bindings/scala_bindings'),
    test_scala_bindings, halt(0)"
# Output: 43 bindings registered
```

All tests pass.
