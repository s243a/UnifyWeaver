# XML Source Plugin Implementation Strategy

## 1. Overview

This document describes the implementation strategy for the `xml_source` plugin. The plugin will be implemented in Prolog and will generate a bash script that executes a small Python script to extract data from XML files in a streaming fashion.

## 2. Plugin Structure

A new file, `src/unifyweaver/sources/xml_source.pl`, will be created. This file will contain the Prolog code for the plugin. The structure of the file will be similar to other source plugins, such as `python_source.pl`.

### Key Predicates

*   `source_info/1`: Provides metadata about the plugin.
*   `validate_config/1`: Validates the configuration options (`xml_file`, `tags`, `engine`).
*   `compile_source/4`: The main entry point for compiling the source. It will extract the configuration and call the bash code generator.
*   `generate_xml_bash/5`: A helper predicate to generate the bash code using the template system.
*   `template_system:template/2`: A multifile predicate that defines the bash template for the `xml_source`.

## 3. XML Parsing Tool: Python with `lxml`

We will use a generated Python script with the `lxml` library for parsing XML. This approach ensures true streaming and constant memory usage, which is crucial for large files.

### `lxml.etree.iterparse`

The generated Python script will use `lxml.etree.iterparse` to iterate over the XML file. This function parses the file incrementally, allowing us to process elements one by one and discard them to save memory.

### Python Script Sketch

The generated bash script will embed a Python script similar to the following:

```python
#!/usr/bin/env python3
import sys
from lxml import etree

file = "{{file}}"
tags = set("{{tags}}".split(','))
null = b'\0'

# Capture namespace map from the root
context = etree.iterparse(file, events=('start', 'end'))
event, root = next(context)
nsmap = root.nsmap or {}

def expand(tag):
    if ':' in tag:
        pfx, local = tag.split(':', 1)
        uri = nsmap.get(pfx)
        return f'{{{uri}}}{local}' if uri else tag
    return tag

want = {expand(t) for t in tags}

for event, elem in context:
    if event == 'end' and elem.tag in want:
        sys.stdout.buffer.write(etree.tostring(elem))
        sys.stdout.buffer.write(null)
        # Memory release pattern
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        root.clear()
```

## 4. Bash Code Generation

We will create a new bash template named `xml_source` in `xml_source.pl`. This template will contain the bash code that executes the embedded Python script.

### Template Placeholders

The template will have the following placeholders:

*   `{{pred}}`: The name of the predicate.
*   `{{python_script}}`: The generated Python script.

### Generated Script

The generated script will look like this:

```bash
#!/bin/bash
# {{pred}} - XML source

{{pred}}() {
    python3 -c '{{python_script}}'
}

{{pred}}_stream() {
    {{pred}}
}

# ... (auto-execute boilerplate)
```

## 5. Firewall Configuration

The use of `python3` and the `lxml` library will require an update to the firewall rules. We will need to add `python3` to the list of allowed services and `lxml` to the list of allowed Python modules.

This will be done by adding the following to the default firewall configuration in `src/unifyweaver/core/firewall_v2.pl`:

```prolog
service(bash, executable(python3)),
python_module(lxml)
```

We will add this to the `firewall_default/1` predicate.

## 6. Testing Strategy

To ensure the `xml_source` plugin works correctly, we will create a new test suite.

### Test Files

1.  **Sample RDF File:** A sample `test.rdf` file will be created in the `test_data` directory. This file will contain a mix of tags, including the ones we want to extract.
2.  **Test Prolog File:** A new test file, `tests/core/test_xml_source.pl`, will be created. This file will contain the following tests:
    *   A test that compiles an `xml_source` and verifies the generated bash script and embedded Python script.
    *   A test that runs the generated script and checks if it correctly extracts the specified tags.
    *   A test that checks for proper handling of multiple tags.
    *   A test that ensures the output is correctly separated by the null character.
    *   A test with a large XML file to verify the streaming and memory performance (optional).

## 7. Implementation Steps

1.  Update the planning documents (this is already done).
2.  Create the `xml_source.pl` file with the structure outlined above.
3.  Implement the bash template that embeds the Python script.
4.  Update the firewall rules in `firewall_v2.pl`.
5.  Create the test files (`test.rdf` and `test_xml_source.pl`).
6.  Implement the tests.
7.  Run the tests to verify the implementation.