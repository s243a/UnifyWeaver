# Type Declaration System — Specification

## 1. Scope

This specification covers:

- The syntax and semantics of Prolog-level type declarations for UnifyWeaver predicates.
- The syntax and semantics of type-emission mode declarations.
- The type vocabulary (primitive and composite types).
- How type information flows from declaration → target Prolog layer → codegen context → rendered code.
- Typed code-generation rules for Java, Haskell, Rust, C#, Kotlin, TypeScript, and F# targets.
- Compatibility rules for untyped targets.

---

## 2. Type Declaration Syntax

Type declarations are Prolog facts using the `uw_type/3` directive:

```prolog
% uw_type(+PredicateName/Arity, +ArgumentIndex, +TypeTerm)
uw_type(edge/2, 1, atom).        % arg 1 (from-node) is an atom
uw_type(edge/2, 2, atom).        % arg 2 (to-node) is an atom
uw_type(weighted_edge/3, 1, atom).
uw_type(weighted_edge/3, 2, atom).
uw_type(weighted_edge/3, 3, float). % arg 3 (weight) is float
```

For predicates that are *derived* (e.g., `tc/2` is the transitive closure of
`edge/2`), the derived predicate inherits the node type of its base unless
explicitly overridden:

```prolog
% Explicit override:
uw_type(tc/2, 1, integer).
uw_type(tc/2, 2, integer).
```

Type emission mode is controlled separately:

```prolog
% uw_typed_mode(+PredicateName/Arity, +Mode)
uw_typed_mode(tc/2, infer).
uw_typed_mode(weighted_edge/3, explicit).
```

Return types are declared separately:

```prolog
% uw_return_type(+PredicateName/Arity, +TypeTerm)
uw_return_type(tc/2, boolean).
uw_return_type(lower_name/2, atom).
```

Supported modes:

- `off`
- `infer`
- `explicit`

### 2.2 Annotation Presence Semantics

Type declaration presence is semantically significant:

- **No `uw_type/3` declaration for an argument**: target should omit the
  annotation for that argument and allow target-native inference/fallback.
- **Explicit `uw_type(..., any)` declaration**: target should emit its top type
  (`Any`, `object`, etc.) when the target supports it.

This distinction avoids conflating unknown types with intentional polymorphism.

Return type declarations are also optional:

- **No `uw_return_type/2` declaration**: target uses existing fallback behavior.
- **Explicit `uw_return_type/2` declaration**: typed targets may annotate
  returns, and `r`-family targets may use it for validation or fallback logic.

### 2.4 Optional Type Diagnostics

Targets that use return-type constraints for compilation decisions may support:

- `type_diagnostics(off)` — silent fallback/filtering
- `type_diagnostics(warn)` — emit diagnostics but continue
- `type_diagnostics(error)` — throw on a type-constraint violation
- `type_diagnostics_report(Report)` — bind a structured report of recorded
  violations without forcing warning or error mode

Default should remain `off` so normal Prolog-style fallback behavior is
preserved unless the caller explicitly requests stricter reporting.

When `type_diagnostics_report(Report)` is present, `Report` should be bound to a
list of dicts describing recorded violations. Current `r`-target fields are:

- `target`
- `predicate`
- `action`
- `expected`
- `inferred`
- `body`

### 2.3 Typed Mode Precedence

`typed_mode` may be set at more than one level. Resolution order is:

1. per-predicate declaration via `uw_typed_mode/2`
2. per-call compile option
3. global compiler setting
4. target default

This allows a project-wide default while preserving local predicate-level
control where needed.

### 2.1 Type Vocabulary

#### Primitive Types

| UnifyWeaver type | Meaning |
|---|---|
| `atom` | A Prolog atom; rendered as `String` in most targets |
| `integer` | Whole number |
| `float` | IEEE 754 double |
| `number` | Integer or float (use when not constrained) |
| `boolean` | True/false |
| `string` | UTF-8 string (distinct from atom in some targets) |
| `any` | No type constraint — fall back to most general target type |

#### Composite Types

| UnifyWeaver type | Meaning |
|---|---|
| `list(T)` | Homogeneous list of `T` |
| `pair(A, B)` | Two-tuple |
| `maybe(T)` | Optional/nullable `T` |
| `map(K, V)` | Key-value map |
| `set(T)` | Unordered unique collection |
| `record(Name, Fields)` | Named struct; `Fields` is a list of `field(Name, Type)` |

#### User-Defined Types

A user may register a domain type via `uw_domain_type/2`:

```prolog
uw_domain_type(employee_id, integer).
uw_domain_type(department, atom).
```

When a target encounters `employee_id`, it resolves it to `integer` for
primitive-only targets, or emits a type alias/newtype where supported
(e.g., Haskell `newtype EmployeeId = EmployeeId Int`).

---

## 3. Type Resolution Pipeline

```
[Prolog source: uw_type/3 facts + uw_return_type/2 facts + uw_typed_mode/2 facts]
        │
        ▼
[Target .pl layer]
  - Looks up uw_type for pred/arity/arg
  - Looks up uw_return_type for pred/arity
  - Resolves typed_mode using precedence rules
  - Falls back to uw_domain_type
  - Falls back to omission for undeclared types
  - Maps abstract type → target-language type string
        │
        ▼
[Codegen context]
  node_type, edge_type, weight_type,
  node_type_decl (import/using if needed),
  typed_mode, type_preamble (when supported), etc.
        │
        ▼
[Template or direct emitter renders output]
```

The key predicate to implement in each `*_target.pl`:

```prolog
%% resolve_type(+AbstractType, +TargetLang, -ConcreteTypeString)
resolve_type(atom,    haskell, "String").
resolve_type(integer, haskell, "Int").
resolve_type(float,   haskell, "Double").
resolve_type(boolean, haskell, "Bool").
resolve_type(atom,    java,    "String").
resolve_type(integer, java,    "Integer").
resolve_type(float,   java,    "Double").
resolve_type(any,     _,       fallback). % handled per-target
```

---

## 4. Codegen Context Extensions

The following new keys are added to the template rendering context when type
information is present. All are optional — templates that do not reference them
continue to work unchanged, and direct emitters may consume the same logical
values without going through Mustache.

| Key | Example value (Haskell) | Example value (Java) |
|---|---|---|
| `node_type` | `String` | `String` |
| `edge_type` | `(String, String)` | `Map.Entry<String,String>` |
| `weight_type` | `Double` | `Double` |
| `typed` | `true`/`false` | `true`/`false` |
| `typed_mode` | `infer` | `explicit` |
| `return_type` | `bool` | `bool` / declared concrete type |
| `node_type_import` | *(empty for primitives)* | `java.util.Map` |
| `rel_type` | `Map String [String]` | `Map<String, List<String>>` |
| `type_preamble` | `type EmployeeId = Int` | `record EmployeeId(...)` |

Mustache sections allow conditional inclusion:

```mustache
{{#typed}}
type Rel = Map.Map {{node_type}} [{{node_type}}]
{{/typed}}
{{^typed}}
type Rel = Map.Map String [String]
{{/typed}}
```

---

## 5. Per-Target Type Mapping Tables

### 5.1 Haskell

| Abstract | Haskell type | Notes |
|---|---|---|
| `atom` | `String` | |
| `integer` | `Int` | Use `Integer` for arbitrary precision |
| `float` | `Double` | |
| `boolean` | `Bool` | |
| `list(T)` | `[T]` | |
| `maybe(T)` | `Maybe T` | |
| `map(K,V)` | `Map.Map K V` | Requires `Data.Map.Strict` |
| `record(N,Fs)` | `data N = N { ... }` | Emit data declaration |

### 5.2 Java

| Abstract | Java type | Notes |
|---|---|---|
| `atom` | `String` | |
| `integer` | `int` / `Integer` | Boxed in generic contexts |
| `float` | `double` / `Double` | |
| `boolean` | `boolean` / `Boolean` | |
| `list(T)` | `List<T>` | `java.util.List` |
| `maybe(T)` | `Optional<T>` | `java.util.Optional` |
| `map(K,V)` | `Map<K,V>` | `java.util.Map` |
| `record(N,Fs)` | `record N(...)` | Java 16+ record |

### 5.3 Rust

| Abstract | Rust type | Notes |
|---|---|---|
| `atom` | `String` | Or `&str` in non-owning positions |
| `integer` | `i64` | Configurable via `uw_type_hint` |
| `float` | `f64` | |
| `boolean` | `bool` | |
| `list(T)` | `Vec<T>` | |
| `maybe(T)` | `Option<T>` | |
| `map(K,V)` | `HashMap<K,V>` | `std::collections::HashMap` |
| `record(N,Fs)` | `struct N { ... }` | Emit struct definition |

### 5.4 C# / F# / Kotlin / TypeScript

See Appendix A (to be expanded per target).

### 5.5 R / TypR

| Abstract | R | TypR (`typed_mode=infer`) | TypR (`typed_mode=explicit`) |
|---|---|---|---|
| `atom` | no annotation | optional annotation / inferred | explicit annotation |
| `integer` | no annotation | optional annotation / inferred | explicit annotation |
| `float` | no annotation | optional annotation / inferred | explicit annotation |
| `boolean` | no annotation | optional annotation / inferred | explicit annotation |
| `list(T)` | no annotation | may annotate if declared | explicit annotation |
| `map(K,V)` | no annotation | may annotate if declared | explicit annotation |
| `any` | no annotation | emit only when explicitly declared | emit only when explicitly declared |

`r` remains syntactically untyped by default. `typr` is a separate target in
the same runtime family and consumes `uw_type/3` optionally based on
`typed_mode`.

When `uw_return_type/2` is present:

- `typr` should use it to avoid falling back to `Any` where a concrete return
  type is known, both on native-lowered generic paths and on wrapped fallbacks.
- `r` should consume it by default for validation and result-shape fallback
  generation, while remaining usable with no type metadata at all.
- `r` should allow this behavior to be disabled per compile call via
  `type_constraints(false)`.

When no explicit return type is present, targets may still use shallow
return-type inference from inferable binding-shaped bodies to improve signatures
or internal compatibility checks. This inference should be conservative:

- it should prefer known binding output types
- it should ignore non-inferable bodies rather than guessing
- it should not change the requirement that plain `r` remains usable with no
  type declarations

Initial implementation may keep R and TypR templates/code paths separate even
when they share the same underlying type-resolution layer. Later convergence is
allowed once TypR output shape stabilizes.

TypR code generation should also follow TypR binding discipline:

- use `let` when first introducing a name
- use plain assignment for later updates to that name

Example:

```typr
let visited <- [start];
visited <- c(visited, next_node);
```

This is preferable to relying on the compiler to infer whether a plain
assignment is an introduction or an update.

Current TypR lowering policy is intentionally mixed:

- simple fact predicates lower directly to TypR
- supported binding-shaped rule bodies lower directly to TypR
- guard-style zero-output command bindings may be folded into TypR clause
  conditions
- multi-step native TypR control-flow chains may keep later guards and outputs
  in native TypR when those guards depend on earlier bound values
- simple comparison and boolean guard expressions over already-bound
  intermediates may also stay native in those control-flow chains
- structured fan-out chains may also stay native when one earlier bound value
  feeds multiple later derived outputs or conditions
- structured split-and-recombine chains may also stay native when those guarded
  derived values later feed a combined output
- guarded disjunction-style alternative-assignment chains may also stay native
  when each alternative may introduce different branch-local intermediates
  before binding either the same later intermediate or the final output
  directly, and later native steps continue from that selected result
- guarded disjunction-style multi-result chains may also stay native when each
  alternative may introduce different branch-local intermediates before
  binding the same later variables, and later native steps continue from those
  selected results
- multiple sequential branch/rejoin segments may also stay native when each
  rejoin preserves the later result variables needed by subsequent native
  steps, including repeated multi-result selection and rejoin
- asymmetric partial-rejoin chains may also stay native when an earlier rejoin
  preserves only part of the later state, subsequent native steps derive more
  shared values, and a later guarded rejoin continues from that expanded state
- Prolog `if -> then ; else` chains may also stay native when the then/else
  branches bind either the same later intermediate, the final output directly,
  the same later result set, or guard-only control flow before later native
  steps continue from the selected values, including cases where one branch
  uses additional branch-local intermediates before producing that shared
  later result set
- Prolog `if -> then` chains may also stay native when the then branch either
  contributes guard-only control flow for later native steps or binds a later
  intermediate, the final output directly, or the later result set needed by
  subsequent native steps
- accumulator-style tail-recursive predicates may also lower to TypR-valid
  functions when they match the conservative tail-recursion shape currently
  supported by the shared recursion compiler, using raw-expression loop bodies
  inside TypR rather than relabeled standalone R output
- two-level nested guarded alternatives may also stay native when a supported
  semicolon branch contains another guarded alternative whose nested branch may
  itself contain one more guarded alternative, provided those branches select
  the same later result set, including nested multi-result selections
- supported literal-headed branch bodies may also stay native when those chains
  fit inside a TypR `if` branch with `let`-introduced intermediate locals
- supported dataframe helpers such as `filter/3`, `sort_by/3`, and `group_by/3`
  may lower directly to TypR raw-expression assignments
- literal-guarded multi-clause predicates may lower to TypR `if` chains
- more complex generic bodies may still fall back to wrapped R expressions

This is acceptable as long as the shared type metadata and validation rules stay
consistent across both paths.

---

## 6. Fallback / Untyped Targets

Targets in this list emit no type annotations and silently ignore `uw_type`
facts:

- Python (base), Ruby, Lua, Perl, Bash, AWK, R

Targets that support optional type annotations emit them when `uw_type` is
present, and omit them otherwise:

- Python (mypy/mypyc/Codon variants), TypeScript, TypR

---

## 7. Backward Compatibility Guarantee

- Any UnifyWeaver `.pl` source file that does not contain `uw_type/3` facts
  generates **identical output** to the current codebase.
- The `typed` Mustache key is `false` / absent when no type annotations exist,
  so `{{^typed}}...{{/typed}}` guards emit the legacy hardcoded types.
- No existing test should require modification.
