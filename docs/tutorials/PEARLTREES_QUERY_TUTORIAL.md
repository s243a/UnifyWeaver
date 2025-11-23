# Pearltrees Query Tutorial

**Learn how to query your Pearltrees RDF exports with simple, fast Prolog predicates.**

---

## Quick Start

### Find Trees by Title

**Before (complex):**
```prolog
xml_source:compile_source(trees/1, [
    file('context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    engine(awk_pipeline)
], [], Code),
open('extract.sh', write, S),
write(S, Code),
close(S),
shell('bash extract.sh | tr "\\0" "\\n" | grep -i physics').
```

**After (simple):**
```prolog
:- use_module('src/unifyweaver/helpers/pearltrees_query').

find_trees_by_title('physics', Trees).
% Found 18 trees in ~1 second!
```

---

## Common Queries

### 1. Find Trees by Title (Case-Insensitive)

```prolog
:- use_module('src/unifyweaver/helpers/pearltrees_query').

% Find all trees with "physics" in title
find_trees_by_title('physics', PhysicsTrees).

% Find trees with "quantum"
find_trees_by_title('quantum', QuantumTrees).

% Specify custom file
find_trees_by_title('my_export.rdf', 'biology', BioTrees).
```

### 2. Get a Specific Tree by ID

```prolog
% Get the "Physics" tree (ID from URL: .../physics/id10647426)
get_tree_by_id(10647426, PhysicsTree).

% From custom file
get_tree_by_id('my_export.rdf', 12345, MyTree).
```

### 3. Count Trees

```prolog
% Count total trees in default export
count_trees(TotalCount).
% Result: 5002 trees

% Count trees in custom file
count_trees('my_export.rdf', Count).
```

---

## General XML Queries

For non-Pearltrees XML or more advanced queries, use `xml_query`:

```prolog
:- use_module('src/unifyweaver/helpers/xml_query').

% Extract all elements of a certain tag
xml_query:extract_elements(
    'data.xml',
    'product',
    awk_pipeline,
    Products
).

% Find elements with filtering
xml_query:find_elements(
    'data.xml',
    'product',
    awk_pipeline,
    contains(title, 'laptop'),
    Laptops
).

% Count elements
xml_query:count_elements(
    'data.xml',
    'item',
    Count
).
```

---

## Performance

**Tested on 19MB Pearltrees Export (5,002 trees):**

| Operation | Time | Engine |
|-----------|------|--------|
| Extract ALL trees | 0.166s | awk_pipeline |
| Find trees with "physics" | 1.0s | awk_pipeline (includes filtering) |
| Count all trees | 0.2s | awk_pipeline |

**Memory:** Constant ~20KB (streams data, doesn't load into memory)

---

## Advanced: Engine Selection

All predicates use `awk_pipeline` by default (fastest), but you can specify others:

```prolog
:- use_module('src/unifyweaver/helpers/xml_query').

% Use lxml iterparse (most robust)
xml_query:extract_elements(
    'data.xml',
    'tree',
    iterparse,
    Trees
).

% Use xmllint (CLI fallback)
xml_query:extract_elements(
    'data.xml',
    'tree',
    xmllint,
    Trees
).

% Use xmlstarlet (small files only)
xml_query:extract_elements(
    'data.xml',
    'tree',
    xmlstarlet,
    Trees
).
```

---

## Tips

1. **Default File Location**: Pearltrees predicates default to `context/PT/pearltrees_export.rdf`
2. **Case-Insensitive**: All title searches are case-insensitive
3. **Fast**: awk_pipeline processes 19MB in < 200ms
4. **Memory-Efficient**: Streams data, uses constant memory
5. **No Temp Files**: Results returned directly as Prolog lists

---

## What's Happening Under the Hood?

When you call:
```prolog
find_trees_by_title('physics', Trees).
```

The system:
1. Compiles an xml_source with awk_pipeline engine
2. Executes: `awk -f select_xml_elements.awk -v tag="pt:Tree" file.rdf`
3. Captures null-delimited XML elements
4. Filters for elements containing "physics" in title
5. Returns as Prolog list of atoms

**All in ~1 second!**

---

## Next Steps

- See `XML_SOURCE_CONFIGURATION.md` for engine details
- See `XML_PROCESSING_GUIDE.md` for advanced patterns
- Explore `src/unifyweaver/helpers/pearltrees_query.pl` for more predicates

**Happy querying!** 🚀
