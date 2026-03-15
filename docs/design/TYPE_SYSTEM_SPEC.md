# Type Declaration System — Specification

## 1. Scope

This specification covers:

- The syntax and semantics of Prolog-level type declarations for UnifyWeaver predicates.
- The type vocabulary (primitive and composite types).
- How type information flows from declaration → target Prolog layer → Mustache context → rendered code.
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
[Prolog source: uw_type/3 facts]
        │
        ▼
[Target .pl layer: resolve_type/3]
  - Looks up uw_type for pred/arity/arg
  - Falls back to uw_domain_type
  - Falls back to `any`
  - Maps abstract type → target-language type string
        │
        ▼
[Mustache context dict]
  node_type, edge_type, weight_type,
  node_type_decl (import/using if needed), etc.
        │
        ▼
[Mustache template renders typed output]
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

## 4. Mustache Context Extensions

The following new keys are added to the template rendering context when type
information is present. All are optional — templates that do not reference them
continue to work unchanged.

| Key | Example value (Haskell) | Example value (Java) |
|---|---|---|
| `node_type` | `String` | `String` |
| `edge_type` | `(String, String)` | `Map.Entry<String,String>` |
| `weight_type` | `Double` | `Double` |
| `typed` | `true`/`false` | `true`/`false` |
| `node_type_import` | *(empty for primitives)* | `java.util.Map` |
| `rel_type` | `Map String [String]` | `Map<String, List<String>>` |

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

---

## 6. Fallback / Untyped Targets

Targets in this list emit no type annotations and silently ignore `uw_type`
facts:

- Python (base), Ruby, Lua, Perl, Bash, AWK

Targets that support optional type annotations emit them when `uw_type` is
present, and omit them otherwise:

- Python (mypy/mypyc/Codon variants), TypeScript

---

## 7. Backward Compatibility Guarantee

- Any UnifyWeaver `.pl` source file that does not contain `uw_type/3` facts
  generates **identical output** to the current codebase.
- The `typed` Mustache key is `false` / absent when no type annotations exist,
  so `{{^typed}}...{{/typed}}` guards emit the legacy hardcoded types.
- No existing test should require modification.
