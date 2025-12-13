# feat(glue): Add JVM glue module for direct transport

## Summary

Adds the JVM glue module (`jvm_glue.pl`) that enables direct in-process communication between JVM targets (Java, Jython, Scala, Kotlin, Clojure).

## Changes

### New Files

#### `src/unifyweaver/glue/jvm_glue.pl`

**Runtime Detection:**
- `detect_jvm_runtime/1` - Detect JDK, JRE, or GraalVM
- `detect_java_version/1` - Parse Java version number
- `detect_jython/1` - Check Jython availability

**Transport Selection:**
- `jvm_transport_type/3` - Select `direct` vs `pipe` transport
- `can_use_direct/2` - Check if JVM-to-JVM direct works

**Bridge Generation:**
- `generate_java_jython_bridge/3` - Java calling Jython via PythonInterpreter
- `generate_jython_java_bridge/3` - Jython calling Java classes directly

**Process Management:**
- `generate_jvm_launcher/3` - Shell script with classpath management
- `generate_classpath/2` - Build classpath from options
- `generate_jvm_pipeline/3` - Mixed Java/Jython orchestration

### Modified Files

#### `docs/JVM_TARGET.md`
- Added "JVM Glue Module" section with feature list

## Technical Notes

- JVM targets (java, jython, scala, kotlin, clojure) use `direct` transport
- Non-JVM targets use `pipe` transport
- Java → Jython uses embedded `PythonInterpreter` for in-process calls
- Jython → Java imports Java classes directly

## Testing

```bash
# Unit tests
swipl -g "use_module('src/unifyweaver/glue/jvm_glue'), test_jvm_glue, halt(0)"

# End-to-end Jython execution
echo '{"name": "test", "value": 42}' | jython generated_pipeline.py
# Output: {"name": "test", "value": 42}
```

All tests pass.
