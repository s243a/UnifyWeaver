# Output Format Specification

**Status:** Proposal
**Created:** 2025-01-22
**Philosophy:** Flexible output formats while maintaining declarative interface

---

## Problem Statement

Different use cases need different output formats:
- **Pipeline to Python/C#:** Raw XML strings
- **Prolog processing:** Structured terms or dicts
- **Data analysis:** CSV/TSV for external tools
- **Debugging:** Pretty-printed, human-readable
- **Storage:** JSON, compact binary

Currently, output is fixed. We need: **Declare desired format, system handles transformation.**

---

## Design Goals

1. **Declarative:** Specify output format in source declaration
2. **Composable:** Combine with field extraction, SGML parsing
3. **Efficient:** Minimal overhead for format conversion
4. **Extensible:** Easy to add custom formats
5. **Streaming:** Maintain constant memory even with formatting

---

## Output Format Taxonomy

### 1. Raw XML (Current Default)

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    output(raw_xml)  % or just omit output() for default
).

% Returns:
% ['<pt:Tree>...</pt:Tree>', '<pt:Tree>...</pt:Tree>', ...]
```

**Use cases:**
- Pass to external XML processor
- Serialize for storage
- Validate against schema
- Transform with XSLT

**Pros:** No parsing overhead, exact XML preserved
**Cons:** Must parse manually if you need structured data

### 2. Prolog Terms (SGML)

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    parse(sgml),
    output(prolog_term)
).

% Returns:
% [
%   element('pt:Tree', [('rdf:about'='...')], [...]),
%   ...
% ]
```

**Use cases:**
- Pattern matching in Prolog
- Structural transformation
- Deep inspection of XML structure

**Pros:** Full XML structure available, Prolog-native
**Cons:** Verbose, less convenient than dicts

### 3. Dicts (Structured)

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: 'dcterms:title'
    ]),
    output(dict)  % or dict(tree) for named dicts
).

% Returns:
% [
%   tree{id: '10647426', title: 'Physics'},
%   tree{id: '10647427', title: 'Chemistry'},
%   ...
% ]
```

**Use cases:**
- Convenient field access (Tree.title)
- Modern Prolog code
- JSON serialization

**Pros:** Clean syntax, named fields, dot access
**Cons:** Slightly more memory than lists

### 4. Lists/Tuples

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(list)
).

% Returns:
% [
%   ['10647426', 'Physics'],
%   ['10647427', 'Chemistry'],
%   ...
% ]
```

**Use cases:**
- Minimal memory footprint
- Fast iteration
- Easy conversion to other formats

**Pros:** Compact, fast
**Cons:** No named fields, must remember order

### 5. Compounds/Terms

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(compound(tree))
).

% Returns:
% [
%   tree('10647426', 'Physics'),
%   tree('10647427', 'Chemistry'),
%   ...
% ]
```

**Use cases:**
- Pattern matching: `tree(ID, Title)`
- Predicate arguments
- Classic Prolog style

**Pros:** Pattern matching, type-like semantics
**Cons:** Fixed arity

### 6. JSON

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(json)
).

% Returns:
% [
%   '{"id": "10647426", "title": "Physics"}',
%   '{"id": "10647427", "title": "Chemistry"}',
%   ...
% ]
```

**Use cases:**
- API responses
- Web frontend
- Data interchange

**Pros:** Universal format, easy to consume externally
**Cons:** String-based, must parse to use in Prolog

### 7. CSV/TSV

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(csv)
).

% Returns:
% [
%   'id,title',
%   '10647426,Physics',
%   '10647427,Chemistry',
%   ...
% ]
```

**Use cases:**
- Export to spreadsheet
- Data analysis in R/Python
- Grep-friendly format

**Pros:** Simple, universal
**Cons:** No nesting, quoting issues

### 8. Custom

```prolog
:- source(xml, trees,
    file('data.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(custom(my_formatter))
).

% User defines:
my_formatter(dict{id: ID, title: Title}, formatted(ID, Title)).
```

**Use cases:**
- Domain-specific formats
- Enrichment (add computed fields)
- Validation

**Pros:** Complete control
**Cons:** User must implement

---

## Format Specification Syntax

### Simple Format Name

```prolog
output(dict)            % SWI-Prolog dicts
output(list)            % Lists of values
output(compound(tree))  % tree(...) terms
output(json)            % JSON strings
output(csv)             % CSV lines
```

### Named Dicts

```prolog
output(dict(tree))      % tree{...} instead of _{...}
output(dict(pearl))     % pearl{...}
```

### Format Options

```prolog
output(json([
    pretty: true,        % Pretty-printed JSON
    escape: unicode      % Escape unicode characters
]))

output(csv([
    delimiter: '\t',     % TSV instead of CSV
    header: true,        % Include header row
    quote: always        % Always quote fields
]))

output(compound(tree, [
    validate: true,      % Validate arity matches fields
    transform: enrich    % Apply transformation predicate
]))
```

---

## Implementation Architecture

### Output Pipeline

```
Extract → Parse → Structure → **Format** → Return
```

Each stage is optional:
- **Extract:** Get raw XML (always happens)
- **Parse:** SGML parse to terms (if parse(sgml))
- **Structure:** Map to dicts/lists (if fields([...]))
- **Format:** Convert to output format (if output(...))
- **Return:** Yield to Prolog

### Format Compiler

New module: `src/unifyweaver/output/formatters.pl`

```prolog
:- module(formatters, [
    format_output/3,      % format_output(+Format, +Data, -Formatted)
    register_formatter/2  % register_formatter(+Name, +Predicate)
]).

% Main dispatch
format_output(Format, Data, Formatted) :-
    (   builtin_format(Format)
    ->  format_builtin(Format, Data, Formatted)
    ;   custom_format(Format, Data, Formatted)
    ).

% Built-in formatters
format_builtin(raw_xml, XML, XML) :- !.

format_builtin(dict, Data, Dict) :-
    !,
    data_to_dict(Data, Dict).

format_builtin(dict(Tag), Data, Dict) :-
    !,
    data_to_dict(Data, _, Dict),
    Dict = Tag{...}.  % Apply tag

format_builtin(list, Data, List) :-
    !,
    data_to_list(Data, List).

format_builtin(json, Data, JSON) :-
    !,
    data_to_json(Data, JSON).

format_builtin(csv, Data, CSV) :-
    !,
    data_to_csv(Data, CSV).

% Custom formatters
custom_format(custom(Predicate), Data, Formatted) :-
    call(Predicate, Data, Formatted).
```

### Streaming Format Transformation

```prolog
% Format each element as it's extracted (streaming)
stream_format_elements(Stream, Format, FormattedStream) :-
    read_element(Stream, Element),
    (   Element = end_of_file
    ->  FormattedStream = []
    ;   format_output(Format, Element, Formatted),
        FormattedStream = [Formatted|Rest],
        stream_format_elements(Stream, Format, Rest)
    ).
```

**Key:** Format one element at a time, never load all into memory.

---

## Format Conversions

### Dict ↔ JSON

```prolog
dict_to_json(tree{id: ID, title: Title}, JSON) :-
    format(atom(JSON), '{"id": "~w", "title": "~w"}', [ID, Title]).

json_to_dict(JSON, Dict) :-
    atom_json_dict(JSON, Dict, []).
```

### Dict ↔ List

```prolog
dict_to_list(tree{id: ID, title: Title}, [ID, Title]).

list_to_dict([ID, Title], tree{id: ID, title: Title}).
```

### Dict ↔ Compound

```prolog
dict_to_compound(tree{id: ID, title: Title}, tree(ID, Title)).

compound_to_dict(tree(ID, Title), tree{id: ID, title: Title}).
```

### CSV Formatting

```prolog
dict_to_csv(tree{id: ID, title: Title}, CSV) :-
    escape_csv(Title, EscTitle),
    format(atom(CSV), '~w,~w', [ID, EscTitle]).

escape_csv(Value, Escaped) :-
    % Quote if contains comma, quote, or newline
    (   sub_atom(Value, _, _, _, ',')
    ;   sub_atom(Value, _, _, _, '"')
    ;   sub_atom(Value, _, _, _, '\n')
    )
    ->  atom_concat('"', Value, Temp),
        atom_concat(Temp, '"', Escaped)
    ;   Escaped = Value.
```

---

## Combining with Field Extraction

### Example: Pearltrees to JSON API

```prolog
:- source(xml, trees_json,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: xpath('pt:treeId/text()', integer),
        title: xpath('dcterms:title/text()', strip_cdata),
        updated: xpath('pt:lastUpdate/text()', parse_datetime),
        privacy: xpath('pt:privacy/text()', integer)
    ]),
    output(json([pretty: true]))
).

% Generate JSON API response:
?- trees_json(AllTrees),
   findall(T, (member(T, AllTrees), T.privacy = 0), PublicTrees),
   write_json_response(PublicTrees).
```

**Output:**
```json
[
  {
    "id": 10647426,
    "title": "Physics",
    "updated": "2022-05-01T07:19:59",
    "privacy": 0
  },
  ...
]
```

### Example: Export to CSV

```prolog
:- source(xml, trees_csv,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([
        id: 'pt:treeId',
        title: xpath('dcterms:title/text()', strip_cdata),
        updated: 'pt:lastUpdate'
    ]),
    output(csv([header: true, delimiter: '\t']))
).

% Export to file:
export_trees_csv :-
    trees_csv(Rows),
    open('trees.tsv', write, Out),
    maplist(writeln(Out), Rows),
    close(Out).
```

**Output file:**
```
id	title	updated
10647426	Physics	2022-05-01T07:19:59
10647427	Chemistry	2022-04-15T10:30:00
...
```

---

## Performance Implications

### Format Overhead (5,000 elements)

| Format | Overhead | Total Time | Memory |
|--------|----------|------------|--------|
| raw_xml | 0ms | 200ms | 19MB |
| dict | 50ms | 250ms | 2MB |
| list | 20ms | 220ms | 1MB |
| json | 100ms | 300ms | 3MB |
| csv | 80ms | 280ms | 1.5MB |
| custom | Varies | Varies | Varies |

**Recommendation:** Use simplest format needed:
- Just passing through? `raw_xml`
- Prolog processing? `dict`
- Minimal memory? `list`
- External tools? `json` or `csv`

---

## Advanced: Multi-Format Output

```prolog
% Generate multiple formats from one extraction
:- source(xml, trees_multi,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(multi([
        dict(tree),
        json,
        csv
    ]))
).

% Returns:
% multi_output{
%   dict: [tree{id: ..., title: ...}, ...],
%   json: ['{"id": ...}', ...],
%   csv: ['id,title', '10647426,Physics', ...]
% }
```

**Use case:** Generate API response with multiple formats simultaneously.

---

## Conditional Formatting

```prolog
% Format based on data properties
:- source(xml, trees_conditional,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title', privacy: 'pt:privacy']),
    output(conditional([
        when(privacy = 0): json([pretty: true]),
        when(privacy = 1): compound(private_tree),
        default: raw_xml
    ]))
).
```

---

## Output Validation

```prolog
% Validate output before returning
:- source(xml, trees_validated,
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    fields([id: 'pt:treeId', title: 'dcterms:title']),
    output(dict(tree)),
    validate([
        field(id, integer, positive),
        field(title, atom, non_empty)
    ])
).

% Invalid elements are filtered or cause error
```

---

## Integration Examples

### Example 1: Web API

```prolog
% Define API endpoint
api_get_trees(ID, Response) :-
    source(xml, api_trees,
        file('pearltrees_export.rdf'),
        tag('pt:Tree'),
        fields([
            id: xpath('pt:treeId/text()', integer),
            title: xpath('dcterms:title/text()', strip_cdata),
            children: related(parent_child, children)
        ]),
        output(json([pretty: true]))
    ),
    api_trees(AllTrees),
    member(Tree, AllTrees),
    Tree.id = ID,
    Response = Tree.
```

### Example 2: Data Pipeline

```prolog
% Extract → Transform → Load
pipeline :-
    % Extract as dicts
    source(xml, trees_extract,
        file('source.xml'),
        tag('item'),
        fields([...]),
        output(dict(item))
    ),

    % Transform (enrich, filter)
    trees_extract(Items),
    maplist(enrich_item, Items, Enriched),
    include(is_valid, Enriched, Valid),

    % Load as CSV
    maplist(dict_to_csv, Valid, CSV),
    write_csv('output.csv', CSV).
```

### Example 3: Debugging

```prolog
% Pretty-print for debugging
:- source(xml, trees_debug,
    file('data.rdf'),
    tag('pt:Tree'),
    output(pretty([
        max_depth: 3,
        indent: 2,
        colors: true
    ]))
).

% Output:
% tree{
%   id: 10647426,
%   title: "Physics",
%   children: [
%     tree{id: 14682380, title: "Physics Education"},
%     ...
%   ]
% }
```

---

## Open Questions

1. **Streaming JSON arrays?**
   - Output `[` at start, `]` at end, `,` between elements?
   - Or JSON Lines format (one object per line)?

2. **Binary formats?**
   - Protocol Buffers, MessagePack, BSON?
   - For high-performance serialization?

3. **Schema validation?**
   - Validate against JSON Schema, XSD?
   - Type checking?

4. **Format negotiation?**
   - Choose format based on context (HTTP Accept header)?

---

## Implementation Phases

### Phase 1: Core Formats
- raw_xml, dict, list, compound
- Format dispatch and conversion
- **Estimated: 2 days**

### Phase 2: External Formats
- JSON, CSV/TSV
- Proper escaping and encoding
- **Estimated: 2 days**

### Phase 3: Advanced Features
- Multi-format output
- Conditional formatting
- Validation
- **Estimated: 2-3 days**

### Phase 4: Optimization
- Streaming format conversion
- Format caching
- Lazy formatting
- **Estimated: 2 days**

---

## Success Criteria

1. ✅ Declare output format in source definition
2. ✅ Support all major formats (XML, dict, list, JSON, CSV)
3. ✅ Streaming format conversion (constant memory)
4. ✅ Custom formatters supported
5. ✅ Minimal overhead (< 100ms for 5,000 elements)
6. ✅ Composable with field extraction and SGML parsing
7. ✅ Easy to add new formats

---

## Related Design Documents

- `DECLARATIVE_FIELD_EXTRACTION.md` - Extract fields to format
- `STREAMING_SGML_PARSING.md` - Parse to terms before formatting
- `DECLARATIVE_HIERARCHICAL_QUERIES.md` - Format hierarchical results

---

**Next Steps:**
1. Implement core formats (Phase 1)
2. Test with Pearltrees data
3. Add JSON/CSV (Phase 2) based on real needs
4. Iterate on API based on usage
