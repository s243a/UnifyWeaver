# UnifyWeaver Template Engine

UnifyWeaver's targets render generated source code through a small
mustache-compatible templating engine in
`src/unifyweaver/core/template_system.pl`. This document records what
the engine currently supports, what it deliberately leaves out, and
how to extend it.

## Why a custom engine

Every transpilation target needs to splice generated fragments into
boilerplate scaffolds (Cargo manifests, Haskell `Main.hs`, Python
modules, .NET project files, shell scripts). A full templating
language (Jinja2, ERB, Liquid) would be heavy and add a runtime
dependency. The engine in `template_system.pl` is ~200 lines of
Prolog and supports the small subset of mustache syntax we actually
use.

## Supported syntax

### Substitution

```mustache
Hello {{name}}!
```

The engine walks the dict `[Key=Value, ...]` and replaces every
occurrence of `{{Key}}` with `Value`. Unknown keys remain in the
output verbatim — a deliberate choice so missing-key bugs surface
visibly in generated code rather than being silently elided.

```prolog
?- render_template('Hello {{name}}!', [name='World'], R).
R = "Hello World!".
```

### Truthy section blocks

```mustache
{{#flag}}
This block renders only when `flag` is truthy.
{{/flag}}
```

The block is rendered if and only if `flag` is in the dict and its
value is *truthy* (see Truthiness Rules below). Block contents may
contain further substitutions and nested sections.

```prolog
?- render_template('A{{#flag}}B{{/flag}}C', [flag=true], R).
R = "ABC".

?- render_template('A{{#flag}}B{{/flag}}C', [flag=false], R).
R = "AC".
```

### Inverted section blocks

```mustache
{{^flag}}
This block renders only when `flag` is falsy or absent.
{{/flag}}
```

The complement of truthy sections. Useful for "show this fallback
when the feature isn't enabled" patterns.

```prolog
?- render_template('A{{^flag}}B{{/flag}}C', [flag=false], R).
R = "ABC".

?- render_template('A{{^flag}}B{{/flag}}C', [flag=true], R).
R = "AC".
```

### Truthiness rules

A key is *truthy* in a dict when:

- The key is present in the dict (i.e. `member(Key=Value, Dict)` succeeds), AND
- The value is not one of: `false`, `0`, `""`, `''`, `[]`.

A key not in the dict is *falsy*. Any value other than the explicit
falsy list is truthy — `true`, non-empty atoms, non-zero numbers,
non-empty strings, non-empty lists.

This matches Python/Lua/JavaScript intuitions about truthiness rather
than strict mustache semantics (which treats only `false` and `null`
as falsy). The wider falsy set is more useful when a "flag" can come
from option processing where missing means off.

### Nested sections

Sections of *different* names nest cleanly:

```prolog
?- render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=true], R).
R = "[AB]".
?- render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=true, b=false], R).
R = "[A]".
?- render_template('[{{#a}}A{{#b}}B{{/b}}{{/a}}]', [a=false, b=true], R).
R = "[]".
```

## Deliberate non-features

### Same-name nesting

```mustache
{{#x}} ... {{#x}} ... {{/x}} ... {{/x}}
```

The engine does not handle nested sections with the same name — the
first `{{/x}}` will close the outermost `{{#x}}`. If you find
yourself wanting this, restructure with two differently-named tags.

### List iteration

Mustache's `{{#list}}...{{/list}}` rendered once per element of a
list is **not** supported. The engine treats every truthy non-list
identically: render the body once. We don't need iteration in
practice — generated code with N similar fragments is built by
Prolog at codegen time and substituted as a single string.

If list iteration becomes useful, add it as a separate feature with a
distinct syntax (e.g. `{{*list}}...{{/list}}`) so the existing
truthy-section semantics stay simple.

### Partials, lambdas, set delimiters

Mustache's `{{> partial}}`, lambda functions, and the `{{=delim=}}`
delimiter-changing syntax are not implemented. Composition between
templates uses the existing `compose_templates/3` predicate. Lambdas
aren't relevant because templates are pure data. Delimiter changes
are a workaround for languages where `{{` is syntactically
significant — none of UnifyWeaver's targets need that escape hatch.

## Engine architecture

Rendering is a two-pass process:

1. **`expand_sections/3`** — walks the template, finding the earliest
   `{{#tag}}` or `{{^tag}}` and its matching `{{/tag}}`, then either
   keeps or strips the body based on truthiness. Recurses on the body
   (so substitutions and nested sections inside an expanded block get
   processed) and on the tail.
2. **`render_template_string/3`** — pure substitution pass over the
   already-section-expanded string. Walks the dict and replaces each
   `{{Key}}` with `Value`.

```
render_template/3
    │
    ├─▶ atom_string normalize
    ├─▶ expand_sections/3   ← handles {{#}} {{^}} {{/}}
    └─▶ render_template_string/3   ← handles {{key}}
```

The two passes are intentionally separate: section logic only touches
section markers, substitution logic only touches `{{key}}` placeholders.
This makes the engine easier to reason about and keeps the
substitution path identical to the pre-section behaviour, so any
existing template that doesn't use the new syntax renders byte-for-byte
identically.

## Adding new features

### Step-by-step

1. **Add the syntax detection** in `expand_sections/3` (if it's a
   block construct) or as a new pass between `expand_sections` and
   `render_template_string` (if it's a transform on the substituted
   output).
2. **Add unit tests** at `tests/core/test_template_sections.pl` (or
   start a new test file if the feature is a separate concern). Each
   test should use `copy_term/2` for variable scoping — Prolog's
   default sharing across conjuncted goals will silently break tests
   that reuse variable names.
3. **Document the feature** here, including the deliberate
   non-feature edges of the new syntax.
4. **Run the existing test suite** to confirm no regression. Templates
   that don't use the new feature must render identically.

### Sketch: list iteration (if we ever need it)

```prolog
% Pseudocode addition to expand_sections — NOT implemented:
%   {{*list}}body{{/list}} renders body once per element of list,
%   exposing each element under the magic name {{this}}.
```

This would require a third pass (or a recursive call structure
inside `expand_sections`), with a distinct syntax marker (`*` rather
than `#`/`^`) to avoid conflating with truthy-section semantics.

## Tests

Run the section test suite:

```sh
swipl -g run_tests -t halt tests/core/test_template_sections.pl
```

Expected: 18 passed, 0 failed. The suite covers truthy/falsy/inverted
sections, all six falsy values (false, 0, "", '', [], missing key),
nested sections, substitution-inside-section interaction, and
backward compatibility with pure-substitution templates.

## Related modules

- `src/unifyweaver/core/template_system.pl` — the engine itself
- `tests/core/test_template_sections.pl` — section feature tests
- `templates/targets/*/` — actual template files used by the various
  transpilation targets

## Status

| Phase | Status |
|-------|--------|
| `{{key}}` substitution | Complete (original feature) |
| `{{#flag}}/{{/flag}}` truthy sections | Complete |
| `{{^flag}}/{{/flag}}` inverted sections | Complete |
| Same-name nesting | Not supported (deliberate) |
| `{{> partial}}` | Not supported |
| List iteration | Not supported |
| Lambdas / delimiter changes | Not supported |
