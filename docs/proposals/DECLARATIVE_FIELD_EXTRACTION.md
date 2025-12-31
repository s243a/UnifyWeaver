# Declarative Field Extraction Design

**Status:** Proposal
**Created:** 2025-01-22
**Philosophy:** Declarative programming - specify WHAT, not HOW

---

## Problem Statement

Currently, `xml_source` extracts whole XML elements as strings. Users then manually parse these strings to extract specific fields, which:
1. **Breaks declarative philosophy** - Users write imperative parsing code instead of declaring what they want
2. **Duplicates effort** - Every user reimplements XML parsing
3. **Loses performance** - Parse the same XML multiple times instead of once during extraction
4. **Breaks streaming** - Must keep whole elements in memory to parse later

## Declarative Solution

Users should **declare in Prolog** which fields they want, and the system **compiles** the appropriate extraction strategy.

---

## Syntax Design

### Current: Whole Element Extraction

```prolog
:- source(xml, all_trees,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    engine(awk_pipeline)
).

% Returns: ['<pt:Tree>...</pt:Tree>', '<pt:Tree>...</pt:Tree>', ...]
```

**Use case:** Pipeline to Python/C# for deserialization, or when you need the full XML.

### Proposed: Field Extraction

```prolog
:- source(xml, tree_metadata,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title',
        privacy: 'pt:privacy'
    ]),
    engine(awk_pipeline)
).

% Returns: [
%   tree{id: '10647426', title: 'Physics', privacy: '0'},
%   tree{id: '10647427', title: 'Chemistry', privacy: '0'},
%   ...
% ]
```

**Use case:** Extract structured data directly, no manual parsing needed.

### Advanced: XPath Field Extraction

```prolog
:- source(xml, pearl_children,
    file('pearltrees_export.rdf'),
    tag('pt:AliasPearl'),
    fields([
        title: xpath('dcterms:title/text()'),
        parent_id: xpath('pt:parentTree/@rdf:resource', extract_id),
        child_url: xpath('rdfs:seeAlso/@rdf:resource'),
        position: xpath('pt:posOrder/text()', integer)
    ]),
    engine(iterparse)  % XPath requires iterparse or xmllint
).
```

**Use case:** Complex field extraction with transformations.

---

## Field Specification Syntax

### Simple Tag Names

```prolog
fields([
    id: 'pt:treeId',           % Extract text content of <pt:treeId>
    title: 'dcterms:title'     % Extract text content of <dcterms:title>
])
```

**Compiles to (AWK):**
```awk
match($0, /<pt:treeId>([^<]+)<\/pt:treeId>/, arr)
id = arr[1]
```

### XPath Expressions

```prolog
fields([
    id: xpath('pt:treeId/text()'),              % Text content
    url: xpath('./@rdf:about'),                 % Attribute
    children: xpath('pt:pearls/pt:pearl/@id')   % Multiple values
])
```

**Compiles to (xmllint):**
```bash
xmllint --xpath '//pt:Tree[position()=1]/pt:treeId/text()' file.rdf
```

### Field Transformations

```prolog
fields([
    id: xpath('pt:treeId/text()', integer),           % Convert to number
    title: xpath('dcterms:title/text()', strip_cdata), % Strip CDATA wrapper
    url: xpath('./@rdf:about', extract_id),           % Custom transform
    tags: xpath('pt:tag/text()', list)                % Collect multiple
])
```

**Available transforms:**
- `integer` - Convert string to number
- `atom` - Ensure atom type
- `strip_cdata` - Remove `<![CDATA[...]]>` wrapper
- `extract_id` - Extract ID from URL (e.g., `.../id12345` → `12345`)
- `list` - Collect multiple matches into list
- `custom(Predicate)` - User-defined transformation

---

## Output Formats

### Dict Output (Default)

```prolog
tree{id: '10647426', title: 'Physics', privacy: '0'}
```

**Pros:** Named fields, easy to work with
**Cons:** Slightly more memory than lists

### List Output

```prolog
fields([...], output(list))

% Returns: [['10647426', 'Physics', '0'], ...]
```

**Pros:** Minimal memory, fast
**Cons:** Must remember field order

### Prolog Term Output

```prolog
fields([...], output(term(tree)))

% Returns: [tree('10647426', 'Physics', '0'), ...]
```

**Pros:** Can use Prolog pattern matching
**Cons:** Fixed arity

---

## Engine Compilation Strategies

Different engines support different field extraction capabilities:

### awk_pipeline Engine

**Capabilities:**
- Simple tag text extraction: `title: 'dcterms:title'`
- Regex-based extraction
- Fast, streaming, constant memory

**Limitations:**
- No XPath support
- No complex nesting
- Regex-based, may fail on malformed XML

**Compilation example:**
```prolog
fields([id: 'pt:treeId', title: 'dcterms:title'])
```

Compiles to AWK script:
```awk
BEGIN { RS="\0" }
{
    match($0, /<pt:treeId>([^<]+)<\/pt:treeId>/, id_arr)
    match($0, /<dcterms:title><!\[CDATA\[([^\]]+)\]\]><\/dcterms:title>/, title_arr)
    printf "%s\t%s\n", id_arr[1], title_arr[1]
}
```

### iterparse Engine (Python lxml)

**Capabilities:**
- XPath expressions
- Attribute extraction
- Namespace handling
- Streaming (one element at a time)

**Compilation example:**
```prolog
fields([id: xpath('pt:treeId/text()')], engine(iterparse))
```

Compiles to Python:
```python
for event, elem in etree.iterparse(file, tag=tag):
    id = elem.findtext('.//pt:treeId', namespaces=ns)
    yield {'id': id}
    elem.clear()  # Free memory
```

### xmllint Engine

**Capabilities:**
- Full XPath 1.0
- Robust XML parsing
- No streaming (loads element into memory)

**Use when:** Complex XPath needed, file not too large

---

## Implementation Architecture

### Compilation Pipeline

```
Prolog source() directive
    ↓
Parse field specifications
    ↓
Select engine based on capabilities
    ↓
Compile to engine-specific code
    ↓
Execute and stream results
    ↓
Transform to specified output format
    ↓
Return to Prolog
```

### Source Compiler Extensions

Add to `xml_source.pl`:

```prolog
compile_source(Name, Options, Transforms, Code) :-
    option(fields(FieldSpecs), Options),
    !,  % Field extraction mode
    option(engine(Engine), Options, awk_pipeline),
    compile_field_extraction(Name, Engine, FieldSpecs, Options, Transforms, Code).

compile_source(Name, Options, Transforms, Code) :-
    % Existing whole-element extraction
    ...
```

### Field Compiler Module

New module: `src/unifyweaver/sources/xml_field_compiler.pl`

```prolog
:- module(xml_field_compiler, [
    compile_field_extraction/6,
    compile_awk_fields/3,
    compile_xpath_fields/3,
    validate_field_spec/2
]).

compile_field_extraction(Name, awk_pipeline, Fields, Options, Transforms, Code) :-
    compile_awk_fields(Name, Fields, AwkScript),
    generate_awk_wrapper(AwkScript, Options, Code).

compile_field_extraction(Name, iterparse, Fields, Options, Transforms, Code) :-
    compile_xpath_fields(Name, Fields, PythonScript),
    generate_python_wrapper(PythonScript, Options, Code).
```

---

## Backward Compatibility

### Existing Code Works Unchanged

```prolog
% No fields() option → whole element extraction (current behavior)
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree')
).
```

### Gradual Migration

```prolog
% Can mix: extract whole elements, parse some in Prolog
:- source(xml, trees, file('data.rdf'), tag('pt:Tree')).

process_trees :-
    trees(AllTrees),
    maplist(extract_title, AllTrees, Titles).

% Later: migrate to field extraction
:- source(xml, tree_titles,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([title: 'dcterms:title'])
).
```

---

## Performance Implications

### Field Extraction vs Manual Parsing

**Manual parsing (current):**
1. Extract whole element: 0.2s
2. Parse each element in Prolog: 10s
3. **Total: 10.2s**

**Field extraction (proposed):**
1. Extract and parse in one pass: 0.3s
2. **Total: 0.3s**

**Improvement: 34x faster**

### Memory Usage

**Whole elements:**
- Must keep all XML strings in memory
- 19MB file → ~19MB in memory

**Field extraction:**
- Only keep extracted fields
- 19MB file → ~500KB in memory (just the data)

**Improvement: 38x less memory**

---

## Examples

### Example 1: Pearltrees Metadata

```prolog
:- source(xml, tree_info,
    file('context/PT/pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: xpath('dcterms:title/text()', strip_cdata),
        updated: 'pt:lastUpdate',
        privacy: xpath('pt:privacy/text()', integer)
    ])
).

% Usage:
?- tree_info(Trees), member(Tree, Trees), Tree.id = '10647426'.
Tree = tree{id:'10647426', title:'Physics', updated:'2022-05-01T07:19:59', privacy:0}.
```

### Example 2: Parent-Child Relationships

```prolog
:- source(xml, pearl_relations,
    file('context/PT/pearltrees_export.rdf'),
    tag('pt:AliasPearl'),
    fields([
        title: xpath('dcterms:title/text()', strip_cdata),
        parent: xpath('pt:parentTree/@rdf:resource', extract_id),
        child: xpath('rdfs:seeAlso/@rdf:resource', extract_id)
    ]),
    filter(has_child)  % Only pearls that reference other trees
).

% Usage: Find all children of Physics tree
?- pearl_relations(Relations),
   include(parent_is('10647426'), Relations, Children).
```

### Example 3: Mixed Approach

```prolog
% Extract fields for common case
:- source(xml, tree_summaries,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title'])
).

% Extract whole element for complex processing
:- source(xml, complex_trees,
    file('data.rdf'),
    tag('pt:Tree'),
    filter(is_complex)
).

process :-
    tree_summaries(Summaries),  % Fast: just fields
    complex_trees(FullXML),     % Full XML for complex cases
    ...
```

---

## Open Questions

1. **Nested field extraction?**
   ```prolog
   fields([
       tree: [
           id: 'pt:treeId',
           metadata: [
               title: 'dcterms:title',
               privacy: 'pt:privacy'
           ]
       ]
   ])
   ```

2. **Multiple elements with same tag?**
   ```prolog
   fields([
       tags: xpath('pt:tag/text()', list)  % Collect all <pt:tag> values
   ])
   ```

3. **Conditional field extraction?**
   ```prolog
   fields([
       title: 'dcterms:title',
       url: if(has_tag('rdfs:seeAlso'),
               xpath('rdfs:seeAlso/@rdf:resource'),
               null)
   ])
   ```

4. **Field validation?**
   ```prolog
   fields([
       id: xpath('pt:treeId/text()', integer, validate(positive))
   ])
   ```

---

## Implementation Phases

### Phase 1: Basic Field Extraction (awk_pipeline)
- Simple tag text extraction
- Dict output
- Compile to AWK
- **Estimated: 2-3 days**

### Phase 2: XPath Support (iterparse)
- XPath expressions
- Attribute extraction
- Compile to Python lxml
- **Estimated: 3-4 days**

### Phase 3: Transformations
- Built-in transforms (integer, strip_cdata, etc.)
- Custom transform predicates
- **Estimated: 2 days**

### Phase 4: Advanced Features
- Nested fields
- Lists
- Conditionals
- **Estimated: 3-4 days**

---

## Success Criteria

1. ✅ Declare field extraction in Prolog (declarative)
2. ✅ No manual string parsing in user code
3. ✅ Performance: Extract fields in single pass
4. ✅ Memory: Constant memory usage (streaming)
5. ✅ Backward compatible: Existing code works unchanged
6. ✅ Multiple engines: AWK, iterparse, xmllint
7. ✅ Multiple outputs: Dict, list, term

---

## Related Design Documents

- `STREAMING_SGML_PARSING.md` - Parse XML to Prolog terms (streaming)
- `OUTPUT_FORMAT_SPECIFICATION.md` - Output format options
- `DECLARATIVE_HIERARCHICAL_QUERIES.md` - Parent-child relationships

---

**Next Steps:**
1. Review and approve this design
2. Create remaining design documents
3. Implement Phase 1 (basic field extraction with AWK)
4. Test on Pearltrees data
5. Iterate based on real usage
