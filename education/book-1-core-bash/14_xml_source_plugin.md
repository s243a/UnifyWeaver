<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 14: XML Source Plugin

## Introduction

The XML source plugin enables UnifyWeaver to extract data from XML and RDF files using streaming parsers. This is particularly useful for processing large XML documents without loading the entire file into memory.

## Key Features

- **Streaming Processing** - Memory-efficient parsing using lxml's iterparse or xmllint
- **Tag Filtering** - Extract only specific XML elements
- **Namespace Support** - Handle namespaced XML (including RDF)
- **Multiple Engines** - Auto-detect and use best available parser (lxml → xmllint → xmlstarlet)
- **Null-Delimited Output** - Clean separation of extracted elements

## Installation Requirements

The XML source works with multiple parsing engines:

**Option 1: lxml (Recommended)**
```bash
# Linux/WSL
pip3 install lxml

# Cygwin - use pre-compiled package
# Run Cygwin setup and install: python39-lxml
```

**Option 2: xmllint + Perl**
```bash
# Linux/WSL
sudo apt-get install libxml2-utils perl

# Usually pre-installed on most systems
```

**Option 3: xmlstarlet**
```bash
sudo apt-get install xmlstarlet
```

## Basic Usage

### Example: Extract Elements from RDF

**Input File:** `pearltrees.rdf`
```xml
<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:pt="http://www.pearltrees.com/resources/tree/"
         xmlns:dcterms="http://purl.org/dc/terms/">

  <pt:Tree rdf:about="https://www.pearltrees.com/t/hacktivism/id2492215">
    <dcterms:title><![CDATA[Hacktivism]]></dcterms:title>
    <pt:treeId>2492215</pt:treeId>
    <pt:lastUpdate>2011-03-14T19:00:12</pt:lastUpdate>
  </pt:Tree>

  <pt:Page rdf:about="https://www.pearltrees.com/p/article/id3456789">
    <dcterms:title><![CDATA[Article Title]]></dcterms:title>
    <pt:url>https://example.com/article</pt:url>
  </pt:Page>
</rdf:RDF>
```

**Prolog Code:**
```prolog
:- use_module(unifyweaver(sources)).

% Define XML source
:- source(xml, trees, [
    xml_file('pearltrees.rdf'),
    tags(['pt:Tree', 'pt:Page'])  % Extract these tags
]).

% Query: Get all extracted elements
get_all_elements(Element) :-
    trees(Element).
```

**Compilation:**
```prolog
?- compile_dynamic_source(trees/1, [], BashCode).
% Generates bash script that streams XML elements
```

**Generated Output:**
The script outputs XML elements separated by null bytes (`\0`):
```xml
<pt:Tree>...</pt:Tree>\0<pt:Page>...</pt:Page>\0
```

## Configuration Options

### Required Options

**`xml_file(Path)`** - Path to XML file
```prolog
xml_file('data/export.xml')
```

**`tags(List)`** - List of tag names to extract (with namespace prefixes)
```prolog
tags(['item', 'product'])           % Simple tags
tags(['rdf:Description', 'pt:Tree']) % Namespaced tags
```

### Optional Options

**`engine(Engine)`** - Force specific parser
```prolog
engine(iterparse)    % Use lxml (fastest, streaming)
engine(xmllint)      % Use xmllint+perl
engine(xmlstarlet)   % Use xmlstarlet (limited)
```

**`namespace_fix(Boolean)`** - Enable namespace repair
```prolog
namespace_fix(true)  % Fix broken namespace declarations
namespace_fix(false) % Default
```

## Advanced Examples

### Example 1: Extract Specific RDF Resources

```prolog
:- source(xml, rdf_people, [
    xml_file('people.rdf'),
    tags(['foaf:Person']),
    engine(iterparse)
]).

% Extract person data
extract_people :-
    findall(Person, rdf_people(Person), People),
    format('Found ~w people~n', [length(People)]).
```

### Example 2: Processing Large Files

For files that don't fit in memory, the streaming parser maintains constant memory usage:

```prolog
:- source(xml, large_dataset, [
    xml_file('huge_export.xml'),  % Could be gigabytes
    tags(['record'])               % Only extract these
]).

% Process one at a time
process_records :-
    large_dataset(Record),
    % Process Record here
    process_record(Record),
    fail.  % Backtrack to get next record
process_records.
```

### Example 3: Multiple Tag Types

```prolog
:- source(xml, mixed_content, [
    xml_file('document.xml'),
    tags(['chapter', 'section', 'paragraph'])
]).

% Extract all structural elements
get_structure(Type, Content) :-
    mixed_content(Element),
    % Parse element to determine type
    extract_type_and_content(Element, Type, Content).
```

## How It Works

### 1. Engine Detection

The plugin automatically selects the best available parser:

```prolog
detect_available_engine(Engine) :-
    (   check_lxml_available
    ->  Engine = iterparse      % Preferred
    ;   check_xmllint_available
    ->  Engine = xmllint          % Fallback
    ;   check_xmlstarlet_available
    ->  Engine = xmlstarlet       % Last resort
    ;   fail                      % No parser found
    ).
```

### 2. Python Code Generation (lxml)

For the `iterparse` engine, generates streaming Python code:

```python
from lxml import etree

file = "data.xml"
tags = {'pt:Tree', 'pt:Page'}
null = b'\\0'

# Streaming parse
context = etree.iterparse(file, events=('start', 'end'))
event, root = next(context)
nsmap = root.nsmap or {}

# Expand prefixed tags to full URIs
def expand(tag):
    if ':' in tag:
        pfx, local = tag.split(':', 1)
        uri = nsmap.get(pfx)
        return f'{{{uri}}}{local}' if uri else tag
    return tag

want = {expand(t) for t in tags}

# Stream elements
for event, elem in context:
    if event == 'end' and elem.tag in want:
        sys.stdout.buffer.write(etree.tostring(elem))
        sys.stdout.buffer.write(null)
        # Release memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
```

### 3. Bash Script Embedding

The Python code is embedded in a bash script using heredoc:

```bash
#!/bin/bash
xml_stream() {
    python3 /dev/fd/3 3<<'PYTHON'
# Python code here
PYTHON
}

xml_stream
```

## Processing Output

The output is null-delimited XML elements. Process with standard tools:

### Using xargs
```bash
./generated_script.sh | xargs -0 -I {} echo "Element: {}"
```

### Using while loop
```bash
./generated_script.sh | while IFS= read -r -d '' element; do
    echo "Processing: $element"
    # Further processing here
done
```

### Piping to xmllint for pretty-printing
```bash
./generated_script.sh | xargs -0 -I {} sh -c 'echo "{}" | xmllint --format -'
```

## Performance Characteristics

### Memory Usage
- **lxml iterparse**: ~10-20MB constant (file size independent)
- **xmllint**: Variable, typically 2-3x element size
- **xmlstarlet**: Loads entire file into memory

### Speed
```
Small files (<1MB):   lxml ≈ xmllint > xmlstarlet
Medium files (1-100MB): lxml >> xmllint > xmlstarlet
Large files (>100MB):  lxml >>> xmllint (xmlstarlet fails)
```

### Recommendations
- **< 10MB**: Any engine works fine
- **10-100MB**: Prefer lxml
- **> 100MB**: Use lxml only

## Common Use Cases

### 1. RDF Data Extraction
```prolog
:- source(xml, rdf_triples, [
    xml_file('knowledge_base.rdf'),
    tags(['rdf:Description'])
]).
```

### 2. RSS/Atom Feeds
```prolog
:- source(xml, feed_items, [
    xml_file('feed.xml'),
    tags(['item', 'entry'])
]).
```

### 3. Configuration Files
```prolog
:- source(xml, config_settings, [
    xml_file('config.xml'),
    tags(['setting'])
]).
```

### 4. Document Processing
```prolog
:- source(xml, document_sections, [
    xml_file('document.xml'),
    tags(['section'])
]).
```

## Troubleshooting

### Parser Not Found
```
Error: No XML parsing engine available
```
**Solution:** Install lxml or xmllint:
```bash
pip3 install lxml  # Recommended
# OR
sudo apt-get install libxml2-utils
```

### Namespace Issues
If tags aren't being extracted, check namespace prefixes:

```prolog
% Wrong - missing namespace prefix
tags(['Tree'])

% Correct - with namespace prefix
tags(['pt:Tree'])
```

### Memory Issues with Large Files
If using xmlstarlet with large files causes memory errors, switch to lxml:
```prolog
engine(iterparse)  % Force lxml
```

## Next Steps

- Review `data_sources_pipeline_guide.md` for ETL examples
- Combine XML source with other sources in pipelines
- Explore namespace handling in complex RDF documents

## See Also

- Chapter 13: Partitioning and Parallel Execution
- Data Sources Pipeline Guide
- Example: `examples/xml_source_demo.pl` (if available)
