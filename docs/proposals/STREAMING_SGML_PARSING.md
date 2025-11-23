# Streaming SGML/XML Parsing Architecture

**Status:** Proposal
**Created:** 2025-01-22
**Philosophy:** Parse one element at a time, clear memory immediately

---

## Problem Statement

Currently we have two approaches, both problematic:

1. **Extract whole XML strings** → Manual string parsing in Prolog (breaks declarative philosophy)
2. **Load entire file into XML parser** → Hits memory limits on large files

We need: **Parse XML to proper Prolog terms, one element at a time, streaming.**

---

## Design Goals

1. **Streaming:** Parse one element, yield to Prolog, clear memory, repeat
2. **Declarative:** Declare desired Prolog term structure, system compiles parser
3. **Memory-safe:** Constant memory usage regardless of file size
4. **Proper parsing:** Use real XML/SGML parsers, not regex
5. **Flexible output:** Raw terms, dicts, or custom structures

---

## Declarative Syntax

### Basic: Parse to Prolog Terms

```prolog
:- source(xml, tree_terms,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml),  % Use SGML parser
    output(prolog_term)
).

% Returns:
% [
%   element('pt:Tree', [('rdf:about'='https://...')], [
%       element('dcterms:title', [], [cdata('Physics')]),
%       element('pt:treeId', [], ['10647426']),
%       ...
%   ]),
%   ...
% ]
```

**Use case:** Full structural parsing when you need the complete DOM.

### Structured: Map to Prolog Dicts

```prolog
:- source(xml, tree_dicts,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml),
    structure([
        type: tree,
        fields: [
            id: child('pt:treeId', text),
            title: child('dcterms:title', cdata),
            url: attribute('rdf:about'),
            privacy: child('pt:privacy', text, integer)
        ]
    ])
).

% Returns:
% [
%   tree{id: 10647426, title: 'Physics', url: 'https://...', privacy: 0},
%   ...
% ]
```

**Use case:** Clean, typed data structures from XML.

### Custom: User-Defined Structure

```prolog
:- source(xml, custom_trees,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml),
    transform(parse_tree_element)  % User predicate
).

% User defines transformation:
parse_tree_element(Element, tree(ID, Title, Privacy)) :-
    Element = element('pt:Tree', Attrs, Children),
    member(element('pt:treeId', [], [ID]), Children),
    member(element('dcterms:title', [], [cdata(Title)]), Children),
    member(element('pt:privacy', [], [Privacy]), Children).
```

**Use case:** Complex custom structures, validation, enrichment.

---

## Streaming Architecture

### Conceptual Flow

```
Large XML File (19MB)
    ↓
Stream through file
    ↓
Find <pt:Tree> element
    ↓
Extract just this element
    ↓
Parse with SGML parser → Prolog term
    ↓
Apply transformations (if any)
    ↓
Yield to Prolog
    ↓
Clear element from memory ← CRITICAL
    ↓
Continue to next element
```

### Memory Profile

```
Without streaming:
[████████████████████] 19MB - Load entire file

With streaming:
[█] 20KB - One element at a time
[█] 20KB - Next element
[█] 20KB - Next element
...
```

**Result: Constant 20KB memory regardless of file size**

---

## Implementation Approaches

### Option 1: Python lxml.etree.iterparse

**Pros:**
- Built-in streaming support
- Robust XML parsing
- Good performance

**Implementation:**
```python
from lxml import etree

def stream_parse(filename, tag):
    context = etree.iterparse(filename, tag=tag, events=('end',))

    for event, elem in context:
        # Convert to Prolog term
        term = element_to_prolog(elem)

        # Yield to Prolog via stdout
        print(term)

        # CRITICAL: Clear element and free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
```

**Compilation:**
```prolog
compile_source(Name, Options, Transforms, Code) :-
    option(parse(sgml), Options),
    option(engine(Engine), Options, iterparse),
    Engine = iterparse,
    generate_iterparse_script(Name, Options, Transforms, Code).
```

### Option 2: SWI-Prolog library(sgml)

**Pros:**
- Native Prolog
- No external dependencies
- Direct term generation

**Implementation:**
```prolog
:- use_module(library(sgml)).

stream_parse_elements(Stream, Tag, Elements) :-
    stream_parse_one(Stream, Tag, Element),
    (   Element = end_of_file
    ->  Elements = []
    ;   Elements = [Element|Rest],
        stream_parse_elements(Stream, Tag, Rest)
    ).

stream_parse_one(Stream, Tag, Element) :-
    % Read until we find opening tag
    seek_to_tag(Stream, Tag),
    % Extract element content
    read_until_close_tag(Stream, Tag, Content),
    % Parse content with SGML
    load_structure(string(Content), [Element], [
        dialect(xml),
        space(remove)
    ]).
```

**Pros of this approach:**
- Pure Prolog solution
- No Python dependency
- Easier to debug

**Cons:**
- Need to implement tag seeking/extraction
- library(sgml) loads strings, not streams directly

### Option 3: Hybrid - AWK + SGML

**Best of both worlds:**

```bash
# AWK extracts elements (streaming, fast)
awk -f select_xml_elements.awk -v tag="pt:Tree" file.rdf |

# Pipe to Prolog for SGML parsing
swipl -g 'parse_stdin_elements' -t halt
```

**Prolog side:**
```prolog
parse_stdin_elements :-
    read_null_delimited_element(Element),
    (   Element = end_of_file
    ->  true
    ;   parse_element_sgml(Element, Term),
        writeln(Term),
        parse_stdin_elements
    ).

parse_element_sgml(ElementString, Term) :-
    load_structure(string(ElementString), [Term], [
        dialect(xml),
        space(remove)
    ]).
```

**Advantages:**
- ✅ AWK handles streaming (proven fast on your phone)
- ✅ library(sgml) handles proper XML parsing
- ✅ Each component does what it's best at
- ✅ Memory efficient (both AWK and SGML process one element)

---

## SGML Term Structure

### SWI-Prolog library(sgml) Format

```prolog
element(
    Name,           % Tag name (atom)
    Attributes,     % List of Name=Value pairs
    Content         % List of content (elements, text, cdata)
)
```

**Example:**
```prolog
element('pt:Tree',
    ['rdf:about'='https://www.pearltrees.com/s243a/physics/id10647426'],
    [
        element('dcterms:title', [], [cdata('Physics')]),
        element('pt:treeId', [], ['10647426']),
        element('pt:privacy', [], ['0'])
    ]
)
```

### Structure Mapping DSL

Users can declare how to map XML structure to Prolog terms:

```prolog
structure([
    type: tree,
    fields: [
        id: child('pt:treeId', text, integer),
        title: child('dcterms:title', cdata, atom),
        url: attribute('rdf:about'),
        children: children('pt:pearl', list(pearl_struct))
    ]
])
```

**Compiles to transformation predicate:**
```prolog
transform_element(
    element('pt:Tree', Attrs, Children),
    tree{id: ID, title: Title, url: URL, children: Kids}
) :-
    member('rdf:about'=URL, Attrs),
    member(element('pt:treeId', [], [IDText]), Children),
    atom_number(IDText, ID),
    member(element('dcterms:title', [], [cdata(Title)]), Children),
    findall(Pearl,
        (member(element('pt:pearl', _, PC), Children),
         transform_pearl(element('pt:pearl', _, PC), Pearl)),
        Kids).
```

---

## Declarative Structure Mapping

### Field Selectors

```prolog
% Get text content of child element
child(TagName, text)

% Get CDATA content
child(TagName, cdata)

% Get attribute value
attribute(AttrName)

% Get all children with tag
children(TagName, list)

% Nested structure
child(TagName, struct(ChildStructure))
```

### Type Conversions

```prolog
% Convert text to integer
child('pt:treeId', text, integer)

% Strip CDATA wrapper
child('dcterms:title', cdata, atom)

% Custom conversion
child('pt:lastUpdate', text, parse_date)
```

### Example: Full Structure

```prolog
:- source(xml, pearls_structured,
    file('pearltrees_export.rdf'),
    tag('pt:AliasPearl'),
    parse(sgml),
    structure([
        type: pearl,
        fields: [
            title: child('dcterms:title', cdata),
            parent_tree: attribute('pt:parentTree', extract_id),
            child_tree: child('rdfs:seeAlso', attribute('rdf:resource'), extract_id),
            added: child('pt:inTreeSinceDate', text, parse_datetime),
            position: child('pt:posOrder', text, integer)
        ]
    ])
).

% Returns:
% [
%   pearl{
%       title: 'Physics Education',
%       parent_tree: 10647426,
%       child_tree: 14682380,
%       added: datetime(2016,5,19,4,9,50),
%       position: 18
%   },
%   ...
% ]
```

---

## Performance Characteristics

### Memory Usage (19MB File, 5,002 Trees)

| Approach | Memory | Note |
|----------|---------|------|
| Load entire file into DOM | 40MB | Whole file + parser overhead |
| Extract all XML strings | 19MB | All element strings in memory |
| **Streaming SGML parse** | **20KB** | One element at a time |

### Speed Comparison

| Operation | Time | Approach |
|-----------|------|----------|
| Extract whole elements | 0.166s | AWK streaming |
| Parse 5,002 XML strings in Prolog | 8-10s | Manual string parsing |
| **Streaming SGML parse** | **0.4-0.6s** | AWK + library(sgml) hybrid |

**Result: 20x faster than manual parsing, 100x less memory**

---

## Error Handling

### Malformed XML

```prolog
parse_element_sgml(ElementString, Term) :-
    catch(
        load_structure(string(ElementString), [Term], [dialect(xml)]),
        error(E, _),
        (   log_parse_error(E, ElementString),
            Term = error(malformed_xml, E)
        )
    ).
```

### Partial Success

```prolog
% Continue processing even if some elements fail
stream_parse_all(Stream, Tag, ValidElements) :-
    stream_parse_all(Stream, Tag, [], ValidElements).

stream_parse_all(Stream, Tag, Acc, ValidElements) :-
    stream_parse_one(Stream, Tag, Element),
    (   Element = end_of_file
    ->  reverse(Acc, ValidElements)
    ;   Element = error(_, _)
    ->  stream_parse_all(Stream, Tag, Acc, ValidElements)  % Skip errors
    ;   stream_parse_all(Stream, Tag, [Element|Acc], ValidElements)
    ).
```

---

## Integration with xml_source

### Source Directive Extension

```prolog
:- source(xml, Name,
    file(File),
    tag(Tag),
    parse(sgml),           % NEW: Enable SGML parsing
    structure(StructDef),  % NEW: Optional structure mapping
    engine(Engine)         % iterparse, sgml_native, hybrid
).
```

### Compilation Strategy

```prolog
compile_source(Name, Options, Transforms, Code) :-
    option(parse(sgml), Options),
    !,
    select_sgml_engine(Options, Engine),
    compile_sgml_source(Name, Engine, Options, Transforms, Code).

select_sgml_engine(Options, Engine) :-
    option(engine(E), Options),
    member(E, [iterparse, sgml_native, hybrid]),
    !,
    Engine = E.
select_sgml_engine(_, hybrid).  % Default: hybrid AWK + SGML
```

---

## Examples

### Example 1: Simple Parsing

```prolog
:- source(xml, tree_elements,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml)
).

% Usage:
?- tree_elements([First|_]).
First = element('pt:Tree',
    ['rdf:about'='https://...'],
    [element('dcterms:title', [], [cdata('Physics')]), ...]
).
```

### Example 2: Structured Extraction

```prolog
:- source(xml, trees,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml),
    structure([
        type: tree,
        fields: [
            id: child('pt:treeId', text, integer),
            title: child('dcterms:title', cdata)
        ]
    ])
).

% Usage:
?- trees(Trees), member(T, Trees), T.id = 10647426.
T = tree{id:10647426, title:'Physics'}.
```

### Example 3: Custom Transformation

```prolog
:- source(xml, enriched_trees,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    parse(sgml),
    transform(enrich_tree)
).

% User-defined enrichment
enrich_tree(Element, EnrichedTree) :-
    element_to_tree(Element, Tree),
    fetch_metadata_from_web(Tree.url, Metadata),
    EnrichedTree = Tree.put(metadata, Metadata).
```

---

## Open Questions

1. **Namespace handling?**
   - Should we strip namespaces or preserve them?
   - Option to configure namespace prefixes?

2. **Validation?**
   - DTD/XSD validation during parse?
   - Schema-based transformation?

3. **Lazy vs Eager?**
   - Stream results one-by-one to Prolog (lazy)
   - Or batch results (eager, faster but more memory)?

4. **Caching parsed elements?**
   - LRU cache for frequently accessed elements?
   - Trade memory for speed?

---

## Implementation Phases

### Phase 1: Hybrid AWK + SGML (Recommended First)
- AWK extracts elements (proven fast)
- library(sgml) parses to Prolog terms
- Basic structure mapping
- **Estimated: 2-3 days**

### Phase 2: Structure Mapping DSL
- Field selectors
- Type conversions
- Dict output
- **Estimated: 3 days**

### Phase 3: Native Streaming (Optional)
- Pure Prolog streaming parser
- Direct file→term without intermediate string
- **Estimated: 4-5 days**

### Phase 4: Advanced Features
- Custom transformations
- Validation
- Caching
- **Estimated: 3-4 days**

---

## Success Criteria

1. ✅ Parse XML to proper Prolog terms
2. ✅ Constant memory (streaming)
3. ✅ No manual string parsing in user code
4. ✅ Declarative structure mapping
5. ✅ Fast (< 1s for 19MB file)
6. ✅ Handles malformed XML gracefully
7. ✅ Works on mobile (low memory)

---

## Related Design Documents

- `DECLARATIVE_FIELD_EXTRACTION.md` - Extract specific fields without full parsing
- `OUTPUT_FORMAT_SPECIFICATION.md` - Output format options
- `DECLARATIVE_HIERARCHICAL_QUERIES.md` - Using parsed terms for hierarchy traversal

---

**Recommendation:** Start with Phase 1 (Hybrid AWK + SGML) as it leverages existing proven components (AWK is already fast on your phone) and library(sgml) provides proper XML parsing.
