---
file_type: UnifyWeaver Example Library
---
# Playbook Examples: XML Processing

This file contains executable records for UnifyWeaver playbooks related to XML data processing.

## Record: unifyweaver.execution.xml_data_source

This record demonstrates how to use UnifyWeaver's Python source to parse XML data and sum product prices.

> [!example-record]
> id: unifyweaver.execution.xml_data_source
> name: unifyweaver.execution.xml_data_source
> description: Demonstrates Python source with XML processing to sum product prices

```bash
#!/bin/bash
#
# XML Data Source Example - Demonstrates Python source with XML processing
# This script is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
PROLOG_SCRIPT="$TMP_FOLDER/xml_example.pl"
OUTPUT_SCRIPT="$TMP_FOLDER/sum_xml_prices.sh"

# --- Step 1: Create Prolog Script ---
echo "Creating Prolog script..."
mkdir -p "$TMP_FOLDER"
cat > "$PROLOG_SCRIPT" <<'PROLOG_EOF'
:- use_module('src/unifyweaver/sources').
:- use_module('src/unifyweaver/sources/python_source').
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Define a Python source that sums XML prices
:- source(python, sum_xml_prices, [
    python_inline("
import xml.etree.ElementTree as ET
import sys

def sum_prices():
    # Embedded XML data for demonstration
    xml_data = '''
    <products>
        <product>
            <name>Laptop</name>
            <price>1200</price>
        </product>
        <product>
            <name>Keyboard</name>
            <price>75</price>
        </product>
        <product>
            <name>Mouse</name>
            <price>25</price>
        </product>
    </products>
    '''
    root = ET.fromstring(xml_data)
    total_price = 0
    for product in root.findall('product'):
        price = int(product.find('price').text)
        total_price += price
    print(f'Total price: {total_price}')

sum_prices()
")
]).

run_example :-
    % Compile the Python source to bash
    compile_dynamic_source(sum_xml_prices/2, [], BashCode),
    % Write to file
    open('tmp/sum_xml_prices.sh', write, Stream),
    write(Stream, BashCode),
    close(Stream),
    format('~nCompiled Python source to tmp/sum_xml_prices.sh~n', []),
    format('~nExecuting the generated script...~n', []).
PROLOG_EOF

# --- Step 2: Execute Prolog to Generate Bash Script ---
echo "Compiling Python source to bash..."
swipl -g "consult('$PROLOG_SCRIPT'), run_example, halt"

# --- Step 3: Execute Generated Script ---
echo ""
echo "Running generated bash script..."
if [[ -f "$OUTPUT_SCRIPT" ]]; then
    bash "$OUTPUT_SCRIPT"
else
    echo "Error: Expected output script not found: $OUTPUT_SCRIPT"
    exit 1
fi
```

---

## Approach Comparison: Three Ways to Process XML

UnifyWeaver provides three complementary approaches for XML processing, each optimized for different use cases.

### Approach 1: Python Source with XML (Embedded, In-Memory)

**Best for:** Small embedded XML data, complex processing

```prolog
:- source(python, process_products, [
    python_inline("
import xml.etree.ElementTree as ET

xml_data = '''
<products>
    <product id=\"101\"><name>Laptop</name><price>1200</price></product>
    <product id=\"102\"><name>Mouse</name><price>25</price></product>
</products>
'''

root = ET.fromstring(xml_data)
for product in root.findall('product'):
    prod_id = product.get('id')
    name = product.find('name').text
    price = product.find('price').text
    print(f'product({prod_id}, \\'{name}\\', {price}).')
    ")
]).
```

**Characteristics:**
- Uses existing `python` source type
- XML embedded in code
- In-memory processing
- Full Python/ElementTree API available

---

### Approach 2: XML Source (Dedicated, Auto-Streaming)

**Best for:** External XML files, automatic optimization

```prolog
:- source(xml, extract_trees, [
    file('pearltrees_export.rdf'),
    tag('pt:Tree'),
    element_type(tree),
    output('trees.pl')
]).

% Auto-detects file size:
% <10MB → in-memory
% >10MB → streaming
% Can override with mode(streaming) or mode(parallel)
```

**With explicit mode:**
```prolog
:- source(xml, extract_parallel, [
    file('large_catalog.xml'),
    tag('product'),
    mode(parallel),
    workers(8),
    output('products.pl')
]).
```

**Characteristics:**
- Dedicated `xml` source type
- Auto-detects optimal mode
- Configurable thresholds
- Simple, declarative API

---

### Approach 3: Bash Pipeline (Generic, Composable)

**Best for:** Custom pipelines, explicit control

```prolog
:- source(bash_pipeline, extract_and_filter, [
    stages([
        stage(awk, 'select_xml_elements.awk', [
            tag='pt:.*Pearl'
        ]),
        stage(python, 'filter_by_parent_tree.py', [
            tree_id='2492215'
        ]),
        stage(python, 'xml_to_prolog_facts.py', [
            element_type='pearl'
        ])
    ]),
    input_file('pearltrees.rdf'),
    output_file('pearls_filtered.pl')
]).
```

**Characteristics:**
- Explicit pipeline stages
- Maximum flexibility
- Composable filters/transforms
- Works for any bash pipeline

---

## Configuration Options

### XML Source Configuration

```prolog
:- source(xml, my_xml_processor, [
    % INPUT (required, choose one)
    file('path/to/file.xml'),           % External file
    xml_inline('<data>...</data>'),     % Embedded XML
    
    % SELECTION (required)
    tag('element_pattern'),             % Tag to extract (regex)
    
    % MODE (optional, default: auto)
    mode(auto),                         % Auto-detect based on size
    mode(in_memory),                    % Force in-memory
    mode(streaming),                    % Force streaming
    mode(parallel),                     % Force parallel
    
    % PARALLEL OPTIONS
    workers(N),                         % Number of workers (default: 4)
    streaming_threshold(Bytes),         % Threshold for streaming (default: 10MB)
    
    % PROCESSING
    element_type(tree),                 % How to transform elements
    
    % OUTPUT
    output('output.pl')                 % Output file (default: stdout)
]).
```

### Bash Pipeline Configuration

```prolog
:- source(bash_pipeline, my_pipeline, [
    stages([
        stage(awk, 'script.awk', [Args]),
        stage(python, 'script.py', [Args]),
        stage(bash, 'script.sh', [Args]),
        stage(custom, 'any | bash | command', [])
    ]),
    input_file('input'),
    output_file('output')
]).
```

---

## Performance Comparison

| Approach | XML Size | Memory | Speed | Use Case |
|----------|----------|--------|-------|----------|
| **Python (inline)** | <1MB | ~3x file | Fast | Embedded data |
| **XML (in-memory)** | <10MB | ~3x file | Fast | Small files |
| **XML (streaming)** | >10MB | ~20KB | Medium | Large files |
| **XML (parallel)** | >100MB | ~20KB×N | Fast | Very large |
| **Bash Pipeline** | Any | Configurable | Varies | Custom needs |

---

## Examples: All Three Approaches on Same Task

**Task:** Extract products from catalog XML

### Using Python Source (Approach 1)

```prolog
:- source(python, get_products, [
    python_file('process_catalog.py')
]).

% process_catalog.py:
% import xml.etree.ElementTree as ET
% for event, elem in ET.iterparse('catalog.xml', events=('end',)):
%     if 'product' in elem.tag:
%         print(f"product(...).")
%         elem.clear()
```

### Using XML Source (Approach 2)

```prolog
:- source(xml, get_products, [
    file('catalog.xml'),
    tag('product'),
    element_type(product),
    output('products.pl')
]).
```

### Using Bash Pipeline (Approach 3)

```prolog
:- source(bash_pipeline, get_products, [
    stages([
        stage(awk, 'select_xml_elements.awk', [tag='product']),
        stage(python, 'xml_to_prolog_facts.py', [element_type='product'])
    ]),
    input_file('catalog.xml'),
    output_file('products.pl')
]).
```

**All three produce the same output, choose based on your needs!**

