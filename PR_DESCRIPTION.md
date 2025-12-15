# feat(targets): Add generator mode to Java and Jython

## Summary

Adds generator mode to Java and Jython targets for consistency with Kotlin, Scala, and Clojure. **All 5 JVM targets now support both pipeline and generator modes!**

## Changes

### Modified Files

#### `src/unifyweaver/targets/java_target.pl`
- Added `generator_mode(true)` option
- New predicate: `compile_generator_mode_java/4`
- Uses `Stream.flatMap()` for generator semantics
- `process()` returns `Stream<Map<String, Object>>`
- `processAll()` flattens multiple streams
- `Stream.iterate()` for recursive predicates

#### `src/unifyweaver/targets/jython_target.pl`
- Added `generator_mode(true)` option
- New predicate: `compile_generator_mode_jython/4`
- Uses Python generators with `yield`
- `process()` is a generator yielding multiple results
- `process_all()` flattens nested generators
- Python 2.7 compatible (`xrange`, `print >>`)

#### `docs/JVM_TARGET.md`
- Updated Java status to "pipeline + generator modes"
- Updated Jython status to "pipeline + generator modes"

## Testing

```bash
# Java generator mode
swipl -g "use_module('src/unifyweaver/targets/java_target'),
    compile_predicate_to_java(test/2, [generator_mode(true)], C),
    sub_atom(C, _, _, _, 'flatMap'), halt(0)"
# PASS: Uses flatMap

# Jython generator mode
swipl -g "use_module('src/unifyweaver/targets/jython_target'),
    compile_predicate_to_jython(test/2, [generator_mode(true)], C),
    sub_atom(C, _, _, _, 'yield'), halt(0)"
# PASS: Uses yield
```

## JVM Family Complete! 🎉

| Target | Pipeline | Generator |
|--------|----------|-----------|
| Java | ✅ | ✅ |
| Jython | ✅ | ✅ |
| Kotlin | ✅ | ✅ |
| Scala | ✅ | ✅ |
| Clojure | ✅ | ✅ |
