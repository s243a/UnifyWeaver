# XML Query Quick Reference

**Fast lookup for xml_query and pearltrees_query predicates**

---

## Pearltrees Queries

```prolog
:- use_module('src/unifyweaver/helpers/pearltrees_query').
```

### Find Trees by Title

```prolog
% Search in default file (context/PT/pearltrees_export.rdf)
find_trees_by_title('physics', Trees).

% Search in custom file
find_trees_by_title('my_export.rdf', 'quantum', Trees).
```

### Get Tree by ID

```prolog
% Get tree with ID 10647426
get_tree_by_id(10647426, Tree).

% From custom file
get_tree_by_id('my_export.rdf', 12345, Tree).
```

### Count Trees

```prolog
% Count in default file
count_trees(Total).

% Count in custom file
count_trees('my_export.rdf', Total).
```

### Extract Tree Info

```prolog
% Parse tree XML into structured dict
extract_tree_info(TreeXML, Info).
% Info = info{id: 10647426, title: 'Physics', url: '...', ...}
```

---

## General XML Queries

```prolog
:- use_module('src/unifyweaver/helpers/xml_query').
```

### Extract All Elements

```prolog
% Extract all <product> elements from data.xml
extract_elements('data.xml', 'product', awk_pipeline, Products).

% Using different engines
extract_elements('data.xml', 'item', iterparse, Items).
extract_elements('data.xml', 'record', xmllint, Records).
```

### Find Elements with Filter

```prolog
% Find products with "laptop" in title
find_elements('data.xml', 'product', awk_pipeline,
              contains(title, 'laptop'), Laptops).

% Find elements matching a pattern
find_elements('data.xml', 'item', awk_pipeline,
              matches('premium'), PremiumItems).
```

### Count Elements

```prolog
% Count all <item> elements (uses awk_pipeline)
count_elements('data.xml', 'item', Count).

% Count with specific engine
count_elements('data.xml', 'item', iterparse, Count).
```

---

## Engine Options

| Engine | Speed | Memory | Best For |
|--------|-------|--------|----------|
| `awk_pipeline` | Fastest | Constant (~20KB) | Large files, streaming |
| `iterparse` | Fast | Low | Robust parsing, complex XML |
| `xmllint` | Medium | Low | CLI-based, portable |
| `xmlstarlet` | Slow | High | Small files only |

**Default:** All predicates use `awk_pipeline` unless specified.

---

## Filter Syntax

### contains(Field, Text)
Case-insensitive substring match in a field:
```prolog
contains(title, 'physics')    % Match "Physics", "PHYSICS", etc.
```

### matches(Pattern)
Case-insensitive pattern match anywhere in element:
```prolog
matches('id10647')            % Match elements containing "id10647"
```

---

## Common Patterns

### Load module once, query many times

```prolog
:- use_module('src/unifyweaver/helpers/pearltrees_query').

% Multiple queries in same session
find_trees_by_title('physics', Physics),
find_trees_by_title('chemistry', Chemistry),
count_trees(Total).
```

### Process results with standard Prolog

```prolog
% Find trees, then count them
find_trees_by_title('quantum', Trees),
length(Trees, Count),
format('Found ~w quantum trees~n', [Count]).

% Extract and examine first result
find_trees_by_title('relativity', [First|_]),
extract_tree_info(First, Info),
format('First tree: ~w~n', [Info.title]).
```

### Chain queries for complex searches

```prolog
% Find physics trees, extract info from each
find_trees_by_title('physics', Trees),
maplist(extract_tree_info, Trees, Infos),
% Infos now contains structured data for all physics trees
```

---

## Performance Tips

1. **Use awk_pipeline for large files** (default for all predicates)
2. **Filter results in Prolog** rather than re-extracting
3. **Extract once, query many times** - store results in variable
4. **For repeated queries**, consider caching results

### Example: Efficient multi-search

```prolog
% Extract all trees once
extract_elements('context/PT/pearltrees_export.rdf',
                 'pt:Tree', awk_pipeline, AllTrees),

% Filter in Prolog (instant)
include(contains_text('physics'), AllTrees, PhysicsTrees),
include(contains_text('quantum'), AllTrees, QuantumTrees).
```

**Note:** The performance metrics documented here (0.166s to extract 5,002 trees from 19MB) were measured on an Android phone running Termux, demonstrating exceptional efficiency even on mobile hardware!

---

## Troubleshooting

### No results returned
- Check tag name matches XML exactly (case-sensitive)
- Try with `iterparse` engine for more robust parsing
- Verify file path is correct

### Slow performance
- Use `awk_pipeline` (default) instead of other engines
- For small files, all engines are fast
- Memory is constant regardless of file size

### Encoding issues
- All helpers handle UTF-8 correctly
- Use `type(binary)` for file reading (done automatically)

---

## File Locations

- **General helpers:** `src/unifyweaver/helpers/xml_query.pl`
- **Pearltrees helpers:** `src/unifyweaver/helpers/pearltrees_query.pl`
- **Full tutorial:** `docs/tutorials/PEARLTREES_QUERY_TUTORIAL.md`
- **Engine docs:** `docs/XML_SOURCE_CONFIGURATION.md`

---

## Quick Example Session

```prolog
?- use_module('src/unifyweaver/helpers/pearltrees_query').
true.

?- find_trees_by_title('physics', Trees), length(Trees, N).
Trees = ['<pt:Tree ...>', ...],
N = 18.

?- get_tree_by_id(10647426, Tree), extract_tree_info(Tree, Info).
Tree = '<pt:Tree rdf:about="...">...</pt:Tree>',
Info = info{id:10647426, title:'Physics', url:'http://...', ...}.

?- count_trees(Total).
Total = 5002.
```

---

**See also:** `PEARLTREES_QUERY_TUTORIAL.md` for detailed examples and explanations.
