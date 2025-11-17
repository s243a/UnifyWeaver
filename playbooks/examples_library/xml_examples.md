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
