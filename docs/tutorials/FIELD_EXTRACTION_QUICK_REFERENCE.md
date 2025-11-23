# Field Extraction Quick Reference

**Fast lookup for declarative field extraction syntax**

---

## Basic Syntax

```prolog
:- source(xml, PredicateName, [
    file('path/to/file.xml'),
    tag('ElementTag'),
    fields([
        field1: 'xmlTag1',
        field2: 'xmlTag2'
    ])
]).
```

---

## Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `file(Path)` | String | Required | Path to XML file |
| `tag(Tag)` | Atom | Required | XML element tag name |
| `fields([...])` | List | Required for field extraction | Field specifications |
| `engine(Engine)` | `awk_pipeline`, `iterparse`, `xmllint`, `xmlstarlet` | `awk_pipeline` | Extraction engine |
| `output(Format)` | `dict`, `list`, `compound(F)` | `dict` | Output format |
| `field_compiler(Strategy)` | `modular`, `inline`, `prolog` | `modular` | Implementation strategy |
| `case_insensitive(Bool)` | `true`, `false` | `false` | Case-insensitive tag match (awk_pipeline) |

### Field compilers (naming)
- `modular` (AWK via `xml_field_compiler`): faster, external awk/bash; better for larger files; limited namespace handling.
- `inline` (AWK regex in `xml_source.pl`): self-contained awk script; very fast but brittle (regex-based, minimal namespace support).
- `prolog` (library(sgml)): pure Prolog parsing; more robust for namespaces/structure; slower; no external tools.

### Engine applicability (field extraction)
- **awk_pipeline (inline/modular)**: fast, regex-based; limited namespace handling; respects `case_insensitive/1`; uses awk/bash (firewall: system tools).
- **prolog (sgml)**: pure Prolog parsing; more robust namespaces; slower; avoids external tools.
- **iterparse/xmllint/xmlstarlet**: streaming/parsing alternatives (availability/constraints may vary; verify before use).

### Preference & sandbox notes
- Default engine is `awk_pipeline`; override with `engine/1` if you need Prolog/SGML or other parsers.
- awk_pipeline invokes system awk/bash; ensure firewall/sandbox policies allow these tools.
- Prolog/SGML stays in-process (no external binaries) but may be slower.

---

## Output Formats

### Dict (Default)
```prolog
output(dict)
% Returns: [_{field1: 'val1', field2: 'val2'}, ...]
% Access: Item.field1
```

### List
```prolog
output(list)
% Returns: [['val1', 'val2'], ...]
% Access: Item = [Field1, Field2], ...
```

### Compound
```prolog
output(compound(item))
% Returns: [item('val1', 'val2'), ...]
% Pattern match: item(Field1, Field2)
```

---

## Examples

### Extract Two Fields
```prolog
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([
        id: 'productId',
        name: 'productName'
    ])
]).

?- products(Items).
Items = [_{id: '123', name: 'Laptop'}, ...].
```

### Case-insensitive Tag Match (awk)
```prolog
:- source(xml, products_ci, [
    file('products.xml'),
    tag('product'),
    fields([id: 'productId', name: 'productName']),
    engine(awk_pipeline),
    case_insensitive(true)
]).
```

### With Custom Output
```prolog
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([id: 'productId', name: 'productName']),
    output(compound(product))
]).

?- products(Items).
Items = [product('123', 'Laptop'), ...].
```

### Choose Implementation
```prolog
% Modular (default) - AWK via xml_field_compiler module
:- source(xml, products_mod, [
    file('products.xml'),
    tag('product'),
    fields([id: 'productId']),
    field_compiler(modular)
]).

% Inline - AWK self-contained in xml_source.pl
:- source(xml, products_inline, [
    file('products.xml'),
    tag('product'),
    fields([id: 'productId']),
    field_compiler(inline)
]).

% Prolog - Pure Prolog using library(sgml)
:- source(xml, products_prolog, [
    file('products.xml'),
    tag('product'),
    fields([id: 'productId']),
    field_compiler(prolog)
]).
```

---

## Field Specifications

### Simple Tag
```prolog
fields([
    title: 'title'
])
% Extracts: <title>content</title>
```

### With Namespace
```prolog
fields([
    title: 'dcterms:title'
])
% Extracts: <dcterms:title>content</dcterms:title>
```

### CDATA Content
```prolog
fields([
    title: 'title'
])
% Handles automatically:
% <title><![CDATA[content]]></title>
% Returns: 'content' (CDATA stripped)
```

---

## Common Patterns

### Extract and Filter
```prolog
:- source(xml, all_items, [
    file('data.xml'),
    tag('item'),
    fields([category: 'category', name: 'name'])
]).

get_electronics(Electronics) :-
    all_items(All),
    include(is_electronics, All, Electronics).

is_electronics(Item) :-
    Item.category = 'Electronics'.
```

### Extract and Transform
```prolog
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([id: 'id', price: 'price'])
]).

get_total_value(Total) :-
    products(All),
    maplist(get_price, All, Prices),
    sumlist(Prices, Total).

get_price(Product, Price) :-
    atom_number(Product.price, Price).
```

### Multiple Sources
```prolog
:- source(xml, trees, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId', title: 'title'])
]).

:- source(xml, pearls, [
    file('data.xml'),
    tag('Pearl'),
    fields([id: 'pearlId', parent: 'parentId'])
]).
```

---

## Performance Tips

### ✅ Do This
```prolog
% Extract once
:- source(xml, data, [
    file('data.xml'),
    tag('item'),
    fields([id: 'id', name: 'name'])
]).

% Filter in Prolog
?- data(All),
   include(filter1, All, Set1),
   include(filter2, All, Set2).
```

### ❌ Avoid This
```prolog
% Don't extract whole elements then parse
extract_elements('data.xml', 'item', awk_pipeline, All),
maplist(manual_parse, All, Parsed).
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Fields return empty | Check tag names (case-sensitive) |
| CDATA not working | Already automatic - check XML structure |
| Multiple tags | Define separate sources (not supported yet) |
| XPath needed | Use direct tag names with awk_pipeline |
| Memory issues | Use `engine(awk_pipeline)` - constant memory |

---

## Implementation Comparison

| Feature | Modular | Inline | Prolog |
|---------|---------|--------|--------|
| Location | `xml_field_compiler` module | `xml_source.pl` | `xml_source.pl` |
| Engine | AWK | AWK | library(sgml) |
| Dependencies | External module | Self-contained (AWK) | SWI-Prolog SGML |
| Code size | 335 bytes | 333 bytes | ~105 bytes bash wrapper |
| Performance | Fast (streaming) | Fast (streaming) | Slower (DOM parsing) |
| Memory | Constant (~20KB) | Constant (~20KB) | Loads full XML to memory |
| Features | Full | Full | Full |
| Default | Yes | No | No |

**Recommendations:**
- **modular** (default): Best for most use cases - clean separation, easy to maintain
- **inline**: Use if you need zero external module dependencies
- **prolog**: Use for educational purposes, debugging, or when you need pure Prolog (no AWK dependency)

---

## See Also

- `DECLARATIVE_FIELD_EXTRACTION_TUTORIAL.md` - Full tutorial
- `XML_EXTRACTION_TUTORIAL.md` - Element extraction basics
- `PEARLTREES_QUERY_TUTORIAL.md` - Pearltrees-specific helpers
- `docs/proposals/DECLARATIVE_FIELD_EXTRACTION.md` - Design proposal

---

## Quick Example

```prolog
% 1. Load
:- use_module('src/unifyweaver/sources').

% 2. Declare
:- source(xml, items, [
    file('data.xml'),
    tag('item'),
    fields([id: 'id', name: 'name'])
]).

% 3. Use (after compilation)
?- items(All),
   length(All, Count),
   format('Found ~w items~n', [Count]).
```

Done! 🎯
