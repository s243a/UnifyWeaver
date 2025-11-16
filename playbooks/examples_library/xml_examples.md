# Playbook Examples: XML Processing

This file contains executable records for UnifyWeaver playbooks related to XML data processing.

## Record: unifyweaver.execution.xml_data_source

This record demonstrates how to define a UnifyWeaver data source that reads and parses an XML file using a Python worker.

```prolog
:- begin_unifyweaver_record(unifyweaver.execution.xml_data_source).

:- use_module(library(unifyweaver/data_source)).
:- use_module(library(unifyweaver/backend/bash_fork)).

:- data_source_driver(python).
:- data_source_work_fn(sum_prices).

:- data_source_schema([
    string(name),
    number(price)
]).

:- data_source_prolog_fn(xml_prolog_data_source).

data_source_definition(xml_data_source, _{
    driver: python,
    work_fn: sum_prices,
    schema: [string(name), number(price)],
    prolog_fn: xml_prolog_data_source
}).

sum_prices(Data) :-
    python_script(Data,
"
import xml.etree.ElementTree as ET
import sys

def sum_prices():
    # In a real scenario, this would read from a file path provided.
    # For this example, we embed the XML data.
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
"
).

:- end_unifyweaver_record(unifyweaver.execution.xml_data_source).
```
