# Code Generation Philosophy — Templates vs Native Lowering

## The Spectrum

UnifyWeaver generates code for 20+ target languages from Prolog
predicate definitions. There are three distinct strategies for
turning a Prolog predicate into target-language code, and they
form a spectrum:

```
Templates ←────────── Hybrid ──────────→ Native Lowering
(idiomatic)          (TypR today)        (robust)
```

### Templates

A template is a chunk of target-language code with `{{holes}}` that
the compiler fills in. The transitive closure templates are pure
examples: 300+ lines of idiomatic BFS code per language, with only
`{{pred}}`, `{{base}}`, and `{{seed_code}}` varying.

**Strengths:**
- Generated code looks hand-written — idiomatic naming, formatting,
  comments, error handling
- Easy to review and audit — the template IS the output
- Language experts can contribute templates without knowing Prolog
- Fast to add a new language — copy a template, adapt syntax

**Weaknesses:**
- One template per pattern per language — O(patterns × languages)
- Edge cases require template variants or conditionals
- Structural changes (like adding input modes) require splitting
  every template
- No type awareness — the template doesn't know if nodes are
  strings or integers

### Native Lowering

Native lowering translates the Prolog clause structure directly into
target-language constructs. TypR's `native_typr_clause_body` is the
clearest example: it walks the goal sequence, matches each goal
against known patterns (guards, outputs, if-then-else, dataframe
ops), and emits TypR code piece by piece.

**Strengths:**
- Handles arbitrary predicate structure — not limited to pre-defined
  patterns like "transitive closure" or "tail recursion"
- Type-aware — can emit `character()` vs `integer()` based on
  inferred types
- Extensible via specification predicates — new patterns added
  without changing the core
- One implementation covers all predicates that fit the supported
  goal shapes

**Weaknesses:**
- Generated code is less idiomatic — mechanical translation rather
  than hand-crafted style
- Complex implementation — TypR's native lowering is 1000+ lines
  of pattern matching
- Debugging is harder — the generated code doesn't map clearly to
  a readable template
- Language-specific idioms are lost — a native-lowered Rust function
  won't use `impl` blocks or trait patterns the way a Rust developer
  would

### Wrapped Fallback

When native lowering fails, TypR compiles the predicate to R first
(via `compile_predicate_to_r`), then wraps the R code in a raw
expression IIFE: `@{ (function(...) { ... })(...) }@`. This gives
100% coverage at the cost of bypassing the TypR type system.

## The TypR Hybrid

TypR is the only target that implements all three strategies with
a clear priority chain:

1. **Template** — for transitive closure (detected pattern, renders
   `transitive_closure.mustache` with type-aware variables like
   `{{empty_nodes_expr}}`)
2. **Native lowering** — for predicates matching supported goal
   shapes (guards, outputs, if-then-else, comparisons, dataframe
   ops, single-clause and multi-clause dispatch)
3. **Wrapped R** — for everything else (compile to R, wrap in IIFE)

This hybrid exists because TypR sits between two languages: it has
its own syntax and type system, but R is always available as an
escape hatch. The question is: **can other targets benefit from this
same architecture?**

## Why Not Just Templates?

Templates work brilliantly for well-defined patterns — transitive
closure, tail recursion, linear recursion, tree recursion. These
are the "named patterns" that UnifyWeaver detects via
`classify_predicate`.

But many real predicates don't fit named patterns. A predicate
with three clauses, mixed arithmetic and list operations, and a
guard condition doesn't have a template. Today, most targets simply
fail for these predicates — only bash (via the stream compiler) and
TypR (via native lowering) can handle them.

## Why Not Just Native Lowering?

Because `fmt.Println` in Go, `console.log` in JavaScript, and
`println!` in Rust all do the same thing but look completely
different. Native lowering from a common Prolog AST produces
code that works but doesn't look like code a developer would write.

Templates capture the aesthetic and idiomatic expectations of each
language community. A Rust developer expects `HashMap`, `impl`
blocks, and `match` expressions. A Python developer expects classes,
list comprehensions, and `if __name__ == "__main__"`. Templates
deliver this; native lowering doesn't.

## The Insight From TypR

TypR's architecture reveals a design principle:

> **Use templates for the scaffolding, native lowering for the logic.**

The transitive closure template provides the idiomatic structure:
graph initialization, BFS traversal, reachability check. But the
type-aware variables (`{{empty_nodes_expr}}`, `{{seed_code}}`) are
computed by the compiler, not hardcoded in the template.

This is already partially true for other targets — `{{pred}}` and
`{{base}}` are compiler-computed. But TypR goes further with
type-dependent expressions, and the composable input templates
go further with mode-dependent I/O sections.

The next step is to push this further: **templates define the
idiomatic skeleton, but more of the template variables are
computed by target-aware compiler logic.**

## Relationship to Composable Templates

The composable template system (`tc_definitions`, `tc_input_*`,
`tc_interface_cli`) is an intermediate step on this spectrum.
It's still template-based, but the compiler chooses which templates
to compose based on options. This is template-level composition
rather than AST-level lowering.

The full vision would combine both:
- **Composable templates** for structural sections (definitions,
  interface)
- **Native lowering** for the logic inside functions (goal sequences,
  guards, control flow)
- **Type-aware template variables** for expressions that depend on
  inferred types

## Implications for New Targets

When adding a new target language, the choice is:

1. **Template-only** (quickest) — write 7 composable templates,
   get transitive closure support for all input modes. This is what
   we've done for 19 targets.

2. **Template + type awareness** (medium) — like TypR, compute
   type-dependent template variables. Requires `resolve_type` and
   `empty_collection_expr` for the target.

3. **Template + native lowering** (most complete) — like TypR,
   handle arbitrary predicates via goal-by-goal translation. Requires
   a binding registry mapping Prolog goals to target expressions.

4. **Template + native lowering + wrapped fallback** (maximum
   coverage) — like TypR, with a fallback to a known-good target.
   Requires a compilation chain (e.g., TypeScript → JavaScript,
   Kotlin → Java).

Most targets should start at level 1 and progress as needed. TypR
is at level 4 because its relationship with R makes fallback natural.
