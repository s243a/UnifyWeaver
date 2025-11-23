# Declarative Field Extraction Tutorial

**Learn how to extract specific fields from XML using declarative Prolog syntax**

---

## Overview

UnifyWeaver now supports **declarative field extraction** - you specify WHAT fields you want in Prolog, and the system compiles HOW to extract them efficiently.

**What you'll learn:**
- Extract specific fields instead of whole XML elements
- Choose between modular and inline implementations
- Use different output formats (dict, list, compound)
- Achieve 34x faster extraction with 38x less memory

---

## Quick Comparison

### Before: Manual String Parsing (Non-Declarative)

```prolog
% Extract whole elements
extract_elements('data.xml', 'product', awk_pipeline, Products),

% Manually parse each one (imperative!)
maplist(parse_product_xml, Products, ParsedProducts).

parse_product_xml(XML, product(ID, Name)) :-
    atom_codes(XML, Codes),
    extract_field(Codes, 'id', ID),      % String manipulation
    extract_field(Codes, 'name', Name).  % Imperative parsing
```

**Problems:**
- ❌ Manual string parsing
- ❌ Imperative code
- ❌ Parse entire XML twice (extract + parse)
- ❌ High memory usage

### After: Declarative Field Extraction

```prolog
% Declare WHAT you want
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([
        id: 'productId',
        name: 'productName'
    ])
]).

% Use directly!
?- products(AllProducts).
AllProducts = [
    _{id: '1001', name: 'Laptop Pro 15'},
    _{id: '1002', name: 'Wireless Mouse'},
    ...
].
```

**Benefits:**
- ✅ Fully declarative
- ✅ Single-pass extraction
- ✅ 34x faster
- ✅ 38x less memory
- ✅ No manual parsing

---

## Prerequisites

```prolog
:- use_module('src/unifyweaver/sources').
```

---

## Basic Field Extraction

### Example: Pearltrees Data

Given `pearltrees_export.rdf`:
```xml
<pt:Tree rdf:about="https://www.pearltrees.com/s243a/physics/id10647426">
   <dcterms:title><![CDATA[Physics]]></dcterms:title>
   <pt:treeId>10647426</pt:treeId>
   <pt:privacy>0</pt:privacy>
</pt:Tree>
```

**Declarative extraction:**
```prolog
:- source(xml, trees, [
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ])
]).

?- trees(AllTrees).
AllTrees = [
    _{id: '10647426', title: 'Physics'},
    _{id: '10647429', title: 'Introductory Physics'},
    ...
].
```

**What happened:**
1. System compiled field extraction to AWK
2. AWK extracts only specified fields
3. CDATA content handled automatically
4. Returns structured Prolog dicts

---

## Output Formats

### 1. Dict Format (Default)

```prolog
:- source(xml, trees_dict, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId', title: 'title']),
    output(dict)  % or omit for default
]).

?- trees_dict(Trees).
Trees = [
    _{id: '123', title: 'First'},
    _{id: '456', title: 'Second'}
].

% Access with dot notation
?- trees_dict([First|_]),
   Title = First.title.
Title = 'First'.
```

**Use when:** You want named field access with `.` notation

### 2. List Format

```prolog
:- source(xml, trees_list, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId', title: 'title']),
    output(list)
]).

?- trees_list(Trees).
Trees = [
    ['123', 'First'],
    ['456', 'Second']
].
```

**Use when:** You need minimal memory footprint

### 3. Compound Format

```prolog
:- source(xml, trees_compound, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId', title: 'title']),
    output(compound(tree))
]).

?- trees_compound(Trees).
Trees = [
    tree('123', 'First'),
    tree('456', 'Second')
].

% Pattern matching
?- trees_compound(Trees),
   member(tree(ID, 'First'), Trees).
ID = '123'.
```

**Use when:** You want pattern matching and classic Prolog style

---

## Implementation Strategies

UnifyWeaver provides three implementation approaches:

### Option B: Modular (Default)

```prolog
:- source(xml, trees, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId']),
    field_compiler(modular)  % explicit
]).
```

**Characteristics:**
- Delegates to `xml_field_compiler` module
- Uses AWK for streaming extraction
- Separation of concerns
- Easier to extend
- **Default if not specified**

### Option A: Inline

```prolog
:- source(xml, trees, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId']),
    field_compiler(inline)
]).
```

**Characteristics:**
- Self-contained in `xml_source.pl`
- Uses AWK for streaming extraction
- No external module dependencies
- Slightly smaller (333 vs 335 bytes)
- Same functionality as modular

### Option C: Pure Prolog

```prolog
:- source(xml, trees, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId']),
    field_compiler(prolog)
]).
```

**Characteristics:**
- Pure Prolog using `library(sgml)`
- No AWK dependency
- Loads XML into memory (DOM parsing)
- Ideal for educational purposes and debugging
- Slower for large files
- Useful when AWK is not available

**Which to choose?**
- Use **modular** (default) for most use cases - best performance and maintainability
- Use **inline** if you want zero external module dependencies
- Use **prolog** for educational purposes, debugging, or pure Prolog environments
- AWK strategies (modular/inline) are identical in performance (both use streaming)

---

## CDATA Content

CDATA sections are handled automatically:

```xml
<item>
    <title><![CDATA[Physics & Chemistry]]></title>
</item>
```

```prolog
:- source(xml, items, [
    file('data.xml'),
    tag('item'),
    fields([title: 'title'])
]).

?- items(Items).
Items = [_{title: 'Physics & Chemistry'}].
% CDATA delimiters stripped automatically
```

**No special configuration needed!**

---

## Real-World Example: E-Commerce

### Product Catalog XML

```xml
<catalog>
  <product id="1001">
    <name>Laptop Pro 15</name>
    <category>Electronics</category>
    <price>1299.99</price>
    <stock>45</stock>
  </product>
  <product id="1002">
    <name>Wireless Mouse</name>
    <category>Electronics</category>
    <price>29.99</price>
    <stock>200</stock>
  </product>
</catalog>
```

### Declarative Extraction

```prolog
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([
        name: 'name',
        category: 'category',
        price: 'price',
        stock: 'stock'
    ])
]).

% Find all electronics
find_electronics(Electronics) :-
    products(AllProducts),
    include(is_electronics, AllProducts, Electronics).

is_electronics(Product) :-
    Product.category = 'Electronics'.

% Find laptops in stock
find_laptops_in_stock(Laptops) :-
    products(AllProducts),
    include(is_laptop_in_stock, AllProducts, Laptops).

is_laptop_in_stock(Product) :-
    sub_atom(Product.name, _, _, _, 'Laptop'),
    atom_number(Product.stock, Stock),
    Stock > 0.
```

**Usage:**
```prolog
?- find_laptops_in_stock(Laptops),
   length(Laptops, Count).
Laptops = [_{name: 'Laptop Pro 15', category: 'Electronics', price: '1299.99', stock: '45'}],
Count = 1.
```

---

## Performance Benefits

### Memory Usage

**Before (whole element extraction):**
```prolog
% Extracts entire 19MB XML
extract_elements('data.xml', 'Tree', awk_pipeline, Trees).
% Memory: 19MB loaded into Prolog
```

**After (field extraction):**
```prolog
% Extracts only 2 fields from each element
:- source(xml, trees, [
    file('data.xml'),
    tag('Tree'),
    fields([id: 'treeId', title: 'title'])
]).
% Memory: Only extracted fields (~500KB for 5,000 trees)
% 38x reduction!
```

### Speed

**Before:**
```prolog
% Step 1: Extract (200ms)
extract_elements('data.xml', 'Tree', awk_pipeline, Trees),
% Step 2: Parse (6.8s)
maplist(parse_tree, Trees, Parsed).
% Total: ~7 seconds
```

**After:**
```prolog
% Single pass extraction + field selection (200ms)
?- trees(Trees).
% Total: 0.2 seconds
% 34x faster!
```

---

## Advanced: Combining with Filtering

```prolog
% Define filtered source
:- source(xml, physics_trees, [
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title',
        privacy: 'pt:privacy'
    ])
]).

% Post-filter in Prolog
get_public_physics_trees(Trees) :-
    physics_trees(AllTrees),
    include(is_public_physics, AllTrees, Trees).

is_public_physics(Tree) :-
    Tree.privacy = '0',
    downcase_atom(Tree.title, Lower),
    sub_atom(Lower, _, _, _, physics).
```

---

## Limitations & Workarounds

### Current Limitations

1. **Single Tag Only**
   ```prolog
   % This works
   fields([id: 'productId'])

   % This doesn't work yet
   tags(['product', 'item'])
   ```
   **Workaround:** Define multiple sources

2. **No XPath (AWK engine)**
   ```prolog
   % Simple tag names only
   fields([id: 'productId'])  % ✅

   % XPath not supported with awk_pipeline
   fields([id: xpath('.//@id')])  % ❌ (iterparse engine will support this)
   ```
   **Workaround:** Use direct tag names

3. **No Nested Fields**
   ```prolog
   % Can't extract nested structures yet
   fields([
       product: [
           name: 'name',
           price: 'price'
       ]
   ])  % ❌
   ```
   **Workaround:** Flatten fields

---

## Comparison with Design Proposals

This tutorial covers the **current implementation** (Phase 1). See design proposals for future enhancements:

- `DECLARATIVE_FIELD_EXTRACTION.md` - Future XPath support
- `STREAMING_SGML_PARSING.md` - Parse to proper Prolog terms
- `DECLARATIVE_HIERARCHICAL_QUERIES.md` - Parent-child relationships
- `OUTPUT_FORMAT_SPECIFICATION.md` - More output formats

---

## Troubleshooting

### Problem: Fields return empty

**Check:**
1. Tag names are case-sensitive
2. Tag names match exactly (including namespaces)
3. XML file path is correct

**Try:**
```prolog
% Debug: Extract whole element first
:- source(xml, debug_trees, [
    file('data.xml'),
    tag('Tree')
    % No fields() - extracts whole elements
]).

?- debug_trees([First|_]).
% Examine First to see actual tag structure
```

### Problem: CDATA not extracted

**Solution:** Already handled automatically! If still issues:

```prolog
% Check if content is truly CDATA
?- debug_trees([First|_]),
   sub_atom(First, _, _, _, 'CDATA').
% Should find <![CDATA[...]]>
```

### Problem: Which implementation to use?

**Decision tree:**
```
What's your priority?

Educational/Debugging/No AWK?
└─ field_compiler(prolog) - Pure Prolog, library(sgml)

Need zero external module dependencies (but have AWK)?
└─ field_compiler(inline) - Self-contained AWK

Default (best for production)?
└─ field_compiler(modular) - Modular AWK [default]

AWK strategies (modular/inline) have identical:
- Performance (streaming AWK pipeline)
- Features (CDATA, output formats)
- Memory usage (constant ~20KB)

Prolog strategy:
- Slower (DOM parsing, not streaming)
- Higher memory (loads full XML)
- No AWK dependency
- Great for learning/debugging
```

---

## Best Practices

### 1. Extract Fields, Not Elements

```prolog
% Good: Declare fields
:- source(xml, products, [
    file('products.xml'),
    tag('product'),
    fields([name: 'name', price: 'price'])
]).

% Avoid: Extract then parse manually
extract_elements('products.xml', 'product', awk_pipeline, Products),
maplist(manual_parse, Products, Parsed).
```

### 2. Use Appropriate Output Format

```prolog
% For dot notation access
output(dict)  % product.name

% For pattern matching
output(compound(product))  % product(Name, Price)

% For minimal memory
output(list)  % [Name, Price]
```

### 3. Filter in Prolog

```prolog
% Extract all, filter in Prolog
:- source(xml, all_products, [
    file('products.xml'),
    tag('product'),
    fields([name: 'name', category: 'category'])
]).

get_electronics(Electronics) :-
    all_products(All),
    include(is_electronics, All, Electronics).
```

---

## Summary

**Three main components:**
1. `source/3` directive - Declare your data source
2. `fields([...])` option - Specify which fields to extract
3. Output format - Choose dict/list/compound

**Three implementations:**
- `field_compiler(modular)` - Default, AWK via xml_field_compiler module (best for production)
- `field_compiler(inline)` - Self-contained AWK in xml_source.pl (no external modules)
- `field_compiler(prolog)` - Pure Prolog using library(sgml) (educational/debugging)

**Typical workflow:**
```prolog
% 1. Declare
:- source(xml, my_data, [
    file('data.xml'),
    tag('item'),
    fields([id: 'itemId', name: 'itemName'])
]).

% 2. Use
?- my_data(Items),
   include(my_filter, Items, Filtered),
   maplist(process, Filtered).
```

**Performance:**
- 34x faster than extract-then-parse
- 38x less memory
- Streaming: constant ~20KB memory
- Works great on mobile (Android/Termux)

Happy extracting! 🚀
