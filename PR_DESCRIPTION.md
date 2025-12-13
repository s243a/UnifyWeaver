# feat(targets): Add Java target with recursion patterns and bindings

## Summary

Adds a complete Java target for UnifyWeaver with streaming pipeline mode, body translation, recursion patterns, and binding system integration.

## Changes

### New Files

#### `src/unifyweaver/targets/java_target.pl`
- `compile_predicate_to_java/3` - compile predicates to Java classes
- `compile_java_pipeline/3` - multi-step pipeline compilation
- `generate_gradle_build/2` - Gradle build file generation
- Pipeline mode with streaming JSONL I/O
- Body/goal translation for comparisons, arithmetic, and dict access
- Tail recursion → while loop optimization
- General recursion → memoization pattern

#### `src/unifyweaver/bindings/java_bindings.pl`
- String bindings: length, toLowerCase, trim, contains, split, replace, etc.
- Math bindings: abs, max, min, sqrt, pow, floor, ceil, round, PI
- Collection bindings: List size/get/add, Map get/put/containsKey
- I/O bindings: println, eprintln
- Stream API bindings: filter, map, collect, reduce

#### `docs/JVM_TARGET.md`
- JVM target family documentation with transport support
- Implementation roadmap (Java → Kotlin → Scala → Jython → Clojure)

### Modified Files

#### `src/unifyweaver/core/target_registry.pl`
- Added Kotlin to JVM family with android/coroutines capabilities

## Technical Notes

- JVM family uses `direct` transport for in-process communication
- Tail recursion converted to while loop with 10k max iterations
- General recursion uses `__memo__` field for memoization
- Default Java version set to 25 (matching Termux OpenJDK)
- Gradle used as build system (Freenet-compatible)

## Testing

```bash
swipl -g "use_module('src/unifyweaver/targets/java_target'), test_java_pipeline_mode, halt(0)"
```

All tests pass.
