# XML Extraction Tutorial

**Learn how to extract and query data from any XML file using xml_query helpers**

---

## Overview

The `xml_query` module provides simple predicates for extracting elements from XML files without writing complex bash pipelines or dealing with xml_source compilation directly.

**What you'll learn:**
- Extract all elements of a specific tag
- Filter elements based on content
- Count elements efficiently
- Choose the right engine for your use case

---

## Prerequisites

```prolog
:- use_module('src/unifyweaver/helpers/xml_query').
```

---

## Example XML File

Let's work with a sample product catalog (`products.xml`):

```xml
<?xml version="1.0"?>
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
  <product id="1003">
    <name>Office Chair</name>
    <category>Furniture</category>
    <price>249.99</price>
    <stock>15</stock>
  </product>
  <product id="2001">
    <name>Gaming Laptop</name>
    <category>Electronics</category>
    <price>1899.99</price>
    <stock>12</stock>
  </product>
</catalog>
```

---

## Basic Extraction

### Extract All Products

```prolog
?- extract_elements('products.xml', 'product', awk_pipeline, Products).
Products = [
  '<product id="1001">...</product>',
  '<product id="1002">...</product>',
  '<product id="1003">...</product>',
  '<product id="2001">...</product>'
].
```

**What happened:**
1. Opened `products.xml`
2. Found all `<product>` elements
3. Returned as list of XML strings

---

## Filtering Results

### Find Products by Name

```prolog
?- find_elements('products.xml', 'product', awk_pipeline,
                 contains(name, 'laptop'), Laptops).
Laptops = [
  '<product id="1001">...</product>',    % Laptop Pro 15
  '<product id="2001">...</product>'     % Gaming Laptop
].
```

**The `contains/2` filter:**
- `contains(name, 'laptop')` - Matches elements where `<name>` contains "laptop"
- Case-insensitive: matches "Laptop", "laptop", "LAPTOP"
- Works with any XML tag name

### Find by Pattern

```prolog
?- find_elements('products.xml', 'product', awk_pipeline,
                 matches('id="2'), HighIDs).
HighIDs = [
  '<product id="2001">...</product>'
].
```

**The `matches/1` filter:**
- `matches('id="2')` - Matches elements containing the pattern anywhere
- Case-insensitive
- More flexible than field-specific matching

---

## Counting Elements

### Count All Products

```prolog
?- count_elements('products.xml', 'product', Count).
Count = 4.
```

### Count Specific Categories

```prolog
% Extract all products first
?- extract_elements('products.xml', 'product', awk_pipeline, All),
   include(contains_filter(category, 'electronics'), All, Electronics),
   length(Electronics, Count).
Count = 3.
```

---

## Engine Selection

All predicates use `awk_pipeline` by default, but you can specify others:

### 1. awk_pipeline (Default - Fastest)

```prolog
extract_elements('large_file.xml', 'record', awk_pipeline, Records).
```

- **Best for:** Large files (>10MB), streaming data
- **Speed:** Fastest (processes 19MB in ~0.2s *on Android phone/Termux*)
- **Memory:** Constant (~20KB)
- **Platform:** Runs efficiently even on mobile hardware

### 2. iterparse (Most Robust)

```prolog
extract_elements('complex.xml', 'item', iterparse, Items).
```

- **Best for:** Complex XML with namespaces, mixed content
- **Speed:** Fast
- **Memory:** Low
- **Requires:** Python with lxml

### 3. xmllint (CLI Fallback)

```prolog
extract_elements('data.xml', 'node', xmllint, Nodes).
```

- **Best for:** Portable solution, when Python unavailable
- **Speed:** Medium
- **Memory:** Low
- **Requires:** xmllint command-line tool

### 4. xmlstarlet (Small Files Only)

```prolog
extract_elements('small.xml', 'entry', xmlstarlet, Entries).
```

- **Best for:** Very small files (<1MB)
- **Speed:** Slow on large files
- **Memory:** High (loads entire file)
- **Requires:** xmlstarlet command-line tool

---

## Real-World Examples

### Example 1: E-commerce Product Search

```prolog
% Find all laptops in stock
find_laptops_in_stock(File, Laptops) :-
    find_elements(File, 'product', awk_pipeline,
                  contains(name, 'laptop'), AllLaptops),
    include(has_stock, AllLaptops, Laptops).

has_stock(ProductXML) :-
    % Check if <stock> value > 0
    atom_codes(ProductXML, Codes),
    sub_string(ProductXML, _, _, _, '<stock>0</stock>'),
    !,
    fail.
has_stock(_).
```

### Example 2: Blog Post Extraction

Given `blog.xml`:
```xml
<blog>
  <post date="2024-01-15">
    <title>Introduction to Prolog</title>
    <author>Alice</author>
    <tags>prolog,tutorial</tags>
  </post>
  <post date="2024-01-20">
    <title>Advanced Prolog Techniques</title>
    <author>Bob</author>
    <tags>prolog,advanced</tags>
  </post>
</blog>
```

Extract all posts:
```prolog
?- extract_elements('blog.xml', 'post', awk_pipeline, Posts),
   length(Posts, Count).
Posts = ['<post date="2024-01-15">...</post>', ...],
Count = 2.
```

Find posts by author:
```prolog
?- find_elements('blog.xml', 'post', awk_pipeline,
                 contains(author, 'alice'), AlicePosts).
AlicePosts = ['<post date="2024-01-15">...</post>'].
```

### Example 3: Configuration File Parsing

Given `config.xml`:
```xml
<configuration>
  <database>
    <host>localhost</host>
    <port>5432</port>
    <name>mydb</name>
  </database>
  <cache>
    <enabled>true</enabled>
    <ttl>3600</ttl>
  </cache>
</configuration>
```

Extract database config:
```prolog
?- extract_elements('config.xml', 'database', awk_pipeline, [DBConfig]).
DBConfig = '<database><host>localhost</host>...</database>'.
```

---

## Processing Extracted Data

Once you have XML elements, process them with standard Prolog:

### Count Results

```prolog
extract_elements('data.xml', 'item', awk_pipeline, Items),
length(Items, Count),
format('Found ~w items~n', [Count]).
```

### Take First N Results

```prolog
extract_elements('data.xml', 'record', awk_pipeline, AllRecords),
length(First10, 10),
append(First10, _, AllRecords).
```

### Filter Further in Prolog

```prolog
extract_elements('products.xml', 'product', awk_pipeline, All),
include(contains_text('electronics'), All, Electronics),
include(contains_text('laptop'), Electronics, Laptops).
```

### Parse Individual Elements

```prolog
% Custom predicate to extract product info
extract_product_info(ProductXML, product{id: ID, name: Name, price: Price}) :-
    atom_codes(ProductXML, Codes),
    % Extract id="..." attribute
    re_matchsub('id="(?<id>[^"]+)"', ProductXML, Match, []),
    atom_number(Match.id, ID),
    % Extract <name>...</name>
    re_matchsub('<name>(?<name>[^<]+)</name>', ProductXML, NameMatch, []),
    Name = NameMatch.name,
    % Extract <price>...</price>
    re_matchsub('<price>(?<price>[^<]+)</price>', ProductXML, PriceMatch, []),
    atom_number(PriceMatch.price, Price).

% Use it:
?- extract_elements('products.xml', 'product', awk_pipeline, Products),
   maplist(extract_product_info, Products, Infos).
Infos = [
  product{id:1001, name:'Laptop Pro 15', price:1299.99},
  product{id:1002, name:'Wireless Mouse', price:29.99},
  ...
].
```

---

## Performance Best Practices

### 1. Extract Once, Query Many Times

```prolog
% Good: Extract once
extract_elements('large_file.xml', 'record', awk_pipeline, AllRecords),
include(filter1, AllRecords, Set1),
include(filter2, AllRecords, Set2),
include(filter3, AllRecords, Set3).

% Bad: Extract multiple times
find_elements('large_file.xml', 'record', awk_pipeline, filter1, Set1),
find_elements('large_file.xml', 'record', awk_pipeline, filter2, Set2),
find_elements('large_file.xml', 'record', awk_pipeline, filter3, Set3).
```

### 2. Use awk_pipeline for Large Files

```prolog
% For 10MB+ files, always use awk_pipeline
extract_elements('huge_data.xml', 'item', awk_pipeline, Items).
```

### 3. Count Without Extracting Full Content

```prolog
% Just need count? Use count_elements
count_elements('data.xml', 'product', Count).  % Fast!

% Don't do this if you only need count:
extract_elements('data.xml', 'product', awk_pipeline, Products),
length(Products, Count).  % Slower, uses more memory
```

### 4. Filter at Extraction Time When Possible

```prolog
% Good: Filter during extraction
find_elements('data.xml', 'product', awk_pipeline,
              contains(category, 'electronics'), Products).

% Less efficient: Extract all then filter
extract_elements('data.xml', 'product', awk_pipeline, All),
include(is_electronics, All, Products).
```

---

## Common Patterns

### Pattern 1: Extract and Process

```prolog
process_products(File) :-
    extract_elements(File, 'product', awk_pipeline, Products),
    maplist(extract_product_info, Products, Infos),
    maplist(print_product, Infos).

print_product(Info) :-
    format('Product ~w: ~w ($~2f)~n',
           [Info.id, Info.name, Info.price]).
```

### Pattern 2: Conditional Extraction

```prolog
get_products_by_category(File, Category, Products) :-
    find_elements(File, 'product', awk_pipeline,
                  contains(category, Category), Products).
```

### Pattern 3: Multi-level Extraction

```prolog
% Extract outer elements, then inner elements
extract_all_items(CatalogFile, Items) :-
    extract_elements(CatalogFile, 'section', awk_pipeline, Sections),
    maplist(extract_items_from_section, Sections, ItemLists),
    append(ItemLists, Items).

extract_items_from_section(SectionXML, Items) :-
    % Parse section XML and extract <item> elements
    ...
```

---

## Troubleshooting

### Problem: No elements returned

**Check:**
1. Tag name is case-sensitive and matches exactly
2. File path is correct
3. XML is well-formed

**Try:**
```prolog
% Use iterparse for more robust parsing
extract_elements('data.xml', 'item', iterparse, Items).
```

### Problem: Slow extraction

**Solution:**
```prolog
% Make sure you're using awk_pipeline (default)
extract_elements('data.xml', 'item', awk_pipeline, Items).

% Avoid xmlstarlet for large files
```

### Problem: Memory issues

**Solution:**
```prolog
% awk_pipeline uses constant memory (~20KB)
extract_elements('huge_file.xml', 'record', awk_pipeline, Records).

% Process in batches if needed
% (though usually not necessary with awk_pipeline)
```

---

## Next Steps

- **For Pearltrees data:** See `PEARLTREES_QUERY_TUTORIAL.md` for domain-specific helpers
- **Quick lookup:** See `XML_QUERY_QUICK_REFERENCE.md` for predicate cheatsheet
- **Advanced usage:** See `XML_SOURCE_CONFIGURATION.md` for engine details
- **Build custom helpers:** Create your own domain-specific module like `pearltrees_query.pl`

---

## Summary

**Three main predicates:**
1. `extract_elements/4` - Get all elements of a tag
2. `find_elements/5` - Get elements matching a filter
3. `count_elements/3,4` - Count elements

**Two filters:**
1. `contains(Field, Text)` - Field contains text (case-insensitive)
2. `matches(Pattern)` - Element matches pattern (case-insensitive)

**Default engine:** `awk_pipeline` (fastest, constant memory)

**Typical workflow:**
```prolog
:- use_module('src/unifyweaver/helpers/xml_query').

% Extract
extract_elements('data.xml', 'item', awk_pipeline, Items),

% Filter (optional)
include(my_filter, Items, Filtered),

% Process
maplist(process_item, Filtered).
```

Happy querying!
