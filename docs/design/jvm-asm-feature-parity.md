# Design: Jamaica & Krakatau JVM Assembly Feature Parity

## Philosophy

**Bytecode is the shared language; syntax is the skin.** Jamaica and Krakatau emit identical JVM bytecodes (`iadd`, `iload`, `if_icmpge`, `invokestatic`, etc.). The only difference is the textual framing. All recursion logic, memoization patterns, and binding operations are expressed as bytecode instruction sequences in the shared `jvm_bytecode.pl` layer, with only the final class/method wrapping being target-specific.

**Maximize shared code, minimize duplication.** The existing JVM high-level targets (Java, Kotlin, Scala) duplicate entire class templates per recursion type. We avoid this: `jvm_bytecode.pl` generates instruction lists, each target wraps them in its syntax envelope. This is the same pattern established by VarStyle (symbolic vs numeric).

**Reuse established protocols exactly.** The multifile hook protocol (`tail_recursion:compile_tail_pattern/9`, etc.) is used by 8 targets. Jamaica and Krakatau will follow the identical pattern — thin hook clauses that call shared bytecode generators.

**Bindings operate at the bytecode level.** Unlike `java_bindings.pl` which maps to Java source method calls (`.toLowerCase()`), JVM assembly bindings map to `invokevirtual`/`invokestatic` bytecodes directly. A single `jvm_asm_bindings.pl` file serves both targets since the bytecodes are identical.

## Current State

| Feature | Jamaica | Krakatau | Mature targets |
|---------|:---:|:---:|:---:|
| Native clause lowering | YES | YES | YES |
| Recursion patterns (6) | NO | NO | YES |
| Bindings | NO | NO | YES |
| Component system | NO | NO | YES (4/10) |

## Specification

### 1. Recursion Patterns

Six patterns, each with a shared bytecode generator and thin target wrappers:

#### 1a. Tail Recursion
JVM strategy: `goto` loop (no stack growth, no recursive call).
```
LOOP:
  iload n / iconst_0 / if_icmple DONE    ;; exit when n <= 0
  iload acc / iload n / iadd / istore acc ;; acc += n
  iload n / iconst_1 / isub / istore n   ;; n -= 1
  goto LOOP
DONE:
  iload acc / ireturn
```
Plus an entry-point wrapper: `f_entry(n) { return f(n, 0); }`

#### 1b. Linear Recursion
JVM strategy: recursive `invokestatic` self-call with memoization.
Memo via local arrays: `sipush 1024 / newarray int / astore memo`

#### 1c. Tree Recursion
Two recursive calls: `f(n-1) + f(n-2)`, memoized.

#### 1d. Multicall Linear Recursion
Same as tree but formalized for the multicall dispatch hook.

#### 1e. Direct Multicall Recursion
Clause body analysis drives the call pattern.

#### 1f. Mutual Recursion
Multiple methods in one class, each calling the others via `invokestatic ClassName.otherMethod(I)I`.

#### Memoization Strategy
Local-array approach (simpler, avoids `<clinit>`):
- Entry method allocates `int[1024]` and `boolean[1024]` arrays
- Passes them to the core method as extra parameters
- Core method checks `memo_valid[n]`, returns `memo[n]` if cached
- After computing, stores in both arrays

### 2. Shared JVM Assembly Bindings

`jvm_asm_bindings.pl` registers bindings for **both** `jamaica` and `krakatau` atoms using a helper `declare_dual_binding/5`:

| Category | Bindings |
|----------|----------|
| Arithmetic | iadd, isub, imul, idiv, irem, ineg |
| Math | Math.abs, Math.max, Math.min (invokestatic) |
| I/O | System.out.println (getstatic + invokevirtual) |
| Comparison | if_icmpgt, if_icmplt, etc. (formalized from existing guards) |
| Bitwise | iand, ior, ixor, ishl, ishr |

### 3. Component System

Both targets register with `component_registry` following the WAT pattern:
```prolog
:- initialization((
    catch(register_component_type(source, custom_jamaica, custom_jamaica, [...]), _, true)
), now).
```

## Implementation Plan

### Phase 1: Shared Bytecode Recursion Generators
**File**: `src/unifyweaver/core/jvm_bytecode.pl`
- Add `jvm_tail_recursion_bytecode/4`
- Add `jvm_linear_recursion_bytecode/5`
- Add `jvm_tree_recursion_bytecode/5`
- Add `jvm_multicall_recursion_bytecode/5`
- Add `jvm_direct_multicall_bytecode/5`
- Add `jvm_mutual_recursion_bytecode/3`
- Add memoization helpers: `jvm_memo_check_bytecode/3`, `jvm_memo_store_bytecode/3`

### Phase 2: Target Multifile Hooks
**Files**: `jamaica_target.pl`, `krakatau_target.pl`
- Add `use_module` for 6 advanced recursion modules
- Add 6 `:- multifile` declarations per target
- Add 6 hook clauses per target (thin wrappers → shared generators)
- Add `jamaica_wrap_class/4`, `krakatau_wrap_class/5` helpers

### Phase 3: JVM Assembly Bindings
**File**: `src/unifyweaver/bindings/jvm_asm_bindings.pl` (new)
- `declare_dual_binding/5` — registers for both jamaica and krakatau
- Arithmetic, math, I/O, comparison, bitwise categories
- Wire into `init_jamaica_target/0` and `init_krakatau_target/0`

### Phase 4: Component System
**Files**: `jamaica_target.pl`, `krakatau_target.pl`, `target_registry.pl`
- Add component registration to both targets
- Update registry capabilities

### Phase 5: Tests
**File**: `tests/core/test_jvm_asm_native_lowering.pl` (extend)
- 6 recursion generator tests (shared bytecode)
- 12 target wrapping tests (6 per target)
- Binding registration tests
- Component registration tests

## Files Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/unifyweaver/core/jvm_bytecode.pl` | Modify | +250 |
| `src/unifyweaver/targets/jamaica_target.pl` | Modify | +200 |
| `src/unifyweaver/targets/krakatau_target.pl` | Modify | +200 |
| `src/unifyweaver/bindings/jvm_asm_bindings.pl` | Create | ~120 |
| `src/unifyweaver/core/target_registry.pl` | Modify | +2 |
| `tests/core/test_jvm_asm_native_lowering.pl` | Modify | +150 |

## Risks

1. **Memoization complexity**: JVM has no flat linear memory. Mitigation: use local-array approach, avoid `<clinit>`.
2. **String type support**: Current `jvm_bytecode.pl` is int-only. Mitigation: keep string bindings as declaration-only for now.
3. **Termux OOM**: Per project memory, don't run JVM tests in parallel.
