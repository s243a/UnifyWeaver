# ILAsm Target — Philosophy

## Why ILAsm

ILAsm (IL Assembler) is the .NET equivalent of Krakatau/Jamaica for
the JVM. It produces CIL (Common Intermediate Language) text that the
.NET runtime assembles into PE executables. Adding ILAsm completes
the "assembly target family" across all three major managed runtimes:

| Runtime | Assembly target | Bytecode layer | Status |
|---------|----------------|----------------|--------|
| JVM | Jamaica (symbolic) | jvm_bytecode.pl | Done |
| JVM | Krakatau (numeric) | jvm_bytecode.pl | Done |
| .NET | ILAsm | **cil_bytecode.pl** (new) | Planned |
| Android | smali/baksmali | (future) | Not started |

## Architecture Decision: Shared CIL Layer

Following the JVM pattern where Jamaica and Krakatau share
`jvm_bytecode.pl` for expression compilation and guard translation,
ILAsm should have a shared `cil_bytecode.pl` module. Even though
ILAsm is currently the only .NET assembly target, this:

1. Keeps CIL instruction generation separate from output formatting
2. Enables future targets (e.g., a Mono-specific format, or
   direct PE emission) to reuse the instruction logic
3. Follows the proven pattern from JVM

## Relationship to C#

C# (`csharp_target.pl`, `csharp_native_target.pl`) already exists
as a .NET target. ILAsm is NOT a replacement — it's a lower-level
alternative, like Jamaica vs Java:

| Level | JVM | .NET |
|-------|-----|------|
| High-level language | Java, Kotlin, Scala | C# |
| Assembly text | Jamaica, Krakatau | **ILAsm** |
| Binary | .class files | PE/DLL files |

ILAsm is useful when:
- You need precise control over stack layout and local variables
- You're generating code for environments without a C# compiler
- You want to bypass C# syntax constraints (e.g., tail calls,
  explicit stack manipulation)
- As a fallback target in the WAM compilation chain

## The Recursive Template-Then-Lower Advantage

Previous assembly targets were built with templates and basic
`clause_guard_output_split`. The new architecture gives ILAsm
`classify_goal_sequence` support from day one:

1. **compile_expression hooks** — register 4 multifile renderers
   and get ite output, disjunction, guarded tail for free
2. **Shared clause_body_analysis** — the classification pipeline
   handles pattern detection; ILAsm only provides rendering
3. **compile_non_recursive routing** — non-recursive predicates
   go through the full native lowering pipeline

This means ILAsm doesn't need to start with recursion patterns
and bindings — the shared framework handles non-recursive
predicates immediately. Recursion patterns and bindings add
depth incrementally.

## Implementation Priority Order

Traditional approach (used for most targets):
```
1. Recursion patterns (TC, tail, linear, tree) via templates
2. Bindings
3. Native lowering (if at all)
```

New approach (leveraging the shared framework):
```
1. Scaffold: module + compile_predicate_to_ilasm + basic output
2. compile_expression hooks (4 renderers) → immediate ite/disj/guard support
3. classify_goal_sequence integration → full non-recursive coverage
4. Component registration → custom IL injection
5. Bindings (cil_asm_bindings.pl) → .NET BCL mappings
6. Recursion patterns (multifile dispatch) → iterative loops, memoization
7. Transitive closure template → composable mustache
```

Steps 1-3 give a working target for non-recursive predicates.
Steps 4-7 add depth. This is the reverse of the traditional order
because the shared framework now provides the non-recursive
infrastructure that previously required target-specific code.

## CIL vs JVM Differences

Key differences that affect code generation:

| Aspect | JVM | CIL |
|--------|-----|-----|
| Stack | Operand stack | Evaluation stack (similar) |
| Locals | `iload`/`istore` by index | `ldloc`/`stloc` by index |
| Strings | `ldc` pushes constant | `ldstr` pushes string |
| Method calls | `invokestatic`/`invokevirtual` | `call`/`callvirt` |
| Branching | `goto`/`if_icmpXX` | `br`/`bge`/`bgt`/`beq` |
| Returns | `ireturn`/`lreturn` | `ret` (single instruction) |
| Type system | Separate int/long/float/double | Unified `int32`/`int64`/`float32`/`float64` |
| Tail calls | No native support | `.tail` prefix |

CIL's `.tail` prefix is significant — it means tail recursion can
be expressed directly in IL without loop transformation, unlike JVM
where we must transform to while loops.
