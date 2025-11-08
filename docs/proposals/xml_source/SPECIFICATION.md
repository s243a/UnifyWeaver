# XML Source Plugin Specification

## 1. Overview

This document provides a detailed specification for the `xml_source` plugin. This plugin allows users to extract XML elements from a file based on a list of specified tags.

## 2. Configuration

The `xml_source` plugin is configured using the `source/3` predicate, with the source type `xml`. The following options are available in the configuration list:

*   `xml_file(FilePath)`: (Required) The path to the input XML or RDF file.
*   `tags(TagList)`: (Required) A list of XML tag names to extract. The tags can include namespace prefixes (e.g., `'pt:Tree'`).
*   `engine(Engine)`: (Optional) The parsing engine to use. The following engines are supported:
    *   `iterparse` (default): Streaming parser based on Python's `lxml.etree.iterparse`. This engine provides constant memory usage and is suitable for very large files.
    *   `xmllint`: CLI fallback that shells out to `xmllint --xpath` and rehydrates namespace declarations in the extracted fragments. Useful when `lxml` is unavailable but the `libxml2-utils` package is installed.
    *   `xmlstarlet`: CLI fallback that shells out via `xmlstarlet sel`. Fast for small inputs but loads the full document in memory.
*   `namespace_fix(Boolean)`: (Optional, xmllint-only) Controls whether namespace declarations are rehydrated onto each extracted fragment. Defaults to `true`. Set to `false` if downstream tooling already understands inherited prefixes and you want to avoid the additional processing.

### Example Configuration

```prolog
:- source(xml, pearltrees_data, [
    xml_file('path/to/your/data.rdf'),
    tags(['pt:Tree', 'pt:Page'])
]).
```

## 3. Output Format

The `xml_source` plugin generates a bash script that streams the full XML elements matching the specified tags. The elements are separated by the null character (`\0`).

### Example Output

Given the following input XML:

```xml
<root>
  <pt:Tree id="1">...</pt:Tree>
  <other>...</other>
  <pt:Page id="2">...</pt:Page>
</root>
```

And the configuration:

```prolog
:- source(xml, my_data, [
    xml_file('my.xml'),
    tags(['pt:Tree', 'pt:Page'])
]).
```

The generated script `my_data_stream` would produce the following output:

```
<pt:Tree id="1">...</pt:Tree>\0<pt:Page id="2">...</pt:Page>\0
```

(Where `\0` represents the null character).

## 4. Generated Bash Interface

The compiled `xml_source` will provide the following bash functions:

*   `<predicate_name>()`: Streams the extracted XML elements, separated by the null character.
*   `<predicate_name>_stream()`: An alias for `<predicate_name>()`.

### Example Usage

```bash
# Source the generated script
source my_data.sh

# Stream all extracted elements
my_data_stream

# Process the stream using other tools
my_data_stream | xargs -0 -n 1 echo "Found element:"
```

## 5. Arity

The `xml_source` plugin will only support arity 1, where the single argument will be the full XML element as a string.

## 6. Error Handling

The plugin will perform the following checks during compilation:

*   Verify that `xml_file` and `tags` are provided.
*   Check if the `xml_file` exists and issue a warning if it doesn't.

The generated script will rely on Python 3 and the `lxml` library. If `python3` is not in the path or if the `lxml` library is not installed, the script will fail at runtime. This dependency will be documented.
