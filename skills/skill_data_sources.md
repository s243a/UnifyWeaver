# Skill: Data Sources (Sub-Master)

Data source handling for JSON files, JSONL streams, and structured record extraction from documentation.

## When to Use

- User needs to read JSON or JSONL data in Prolog
- User asks about column projection or JSONPath
- User wants to extract structured records from Markdown
- User needs to declare dynamic data sources
- User asks about null handling or nested records

## Skill Hierarchy

```
skill_data_tools.md (parent)
└── skill_data_sources.md (this file)
    ├── skill_json_sources.md - JSON/JSONL data source handling
    └── skill_extract_records.md - Markdown record extraction
```

## Quick Start

### JSON Source with Column Projection

```prolog
:- source(json, order_totals, [
    json_file('data/orders.json'),
    columns(['order.customer.name', 'items[0].total'])
]).

% Query the source
?- order_totals(CustomerName, Total).
```

### JSONL Stream with Null Handling

```prolog
:- source(json, events, [
    json_file('data/events.jsonl'),
    record_format(jsonl),
    columns(['event.type', 'event.timestamp']),
    null_policy(skip)  % or default('N/A'), fail
]).
```

### Schema-Based Typed Records

```prolog
:- source(json, products, [
    json_file('data/products.json'),
    schema([
        field(id, 'id', string),
        field(name, 'name', string),
        field(price, 'price', double)
    ]),
    record_type('ProductRecord')
]).

% Returns typed ProductRecord objects
?- products(Record).
```

### Extract Records from Markdown

```bash
# Extract as JSON
perl scripts/utils/extract_records.pl \
  -f json \
  -q "pattern" \
  path/to/file.md

# Extract just content
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.xml_data_source" \
  playbooks/examples_library/xml_examples.md > tmp/script.sh
```

## JSON Source Features

### Column Projection Modes

| Mode | Description | Example |
|------|-------------|---------|
| Dot notation | Simple path | `'order.customer.name'` |
| Array index | Access element | `'items[0].total'` |
| JSONPath | Full selector | `jsonpath('$.orders[*].total')` |

### JSONPath Selectors

```prolog
:- source(json, deep_data, [
    json_file('data/nested.json'),
    columns([
        jsonpath('$.users[*].profile.email'),
        jsonpath('$..address.city')  % recursive descent
    ])
]).
```

Supported features:
- Root selector: `$`
- Dot notation: `.property`
- Bracket notation: `['property']`
- Array indices: `[0]`, `[*]`
- Recursive descent: `..property`

### Return Object Mode

Get full JSON objects instead of projected columns:

```prolog
:- source(json, raw_products, [
    json_file('data/products.json'),
    arity(1),
    return_object(true),
    type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json')
]).
```

### Nested Schema Records

```prolog
:- source(json, orders, [
    json_file('data/orders.json'),
    schema([
        field(order, 'order', record('OrderRecord', [
            field(id, 'id', string),
            field(customer, 'customer.name', string)
        ])),
        field(first_item, 'items[0]', record('LineItemRecord', [
            field(product, 'product', string),
            field(total, 'total', double)
        ]))
    ]),
    record_type('OrderSummaryRecord')
]).
```

### Null Handling Policies

| Policy | Behavior |
|--------|----------|
| `null_policy(fail)` | Throw on null projection |
| `null_policy(skip)` | Skip rows with null values |
| `null_policy(default(Value))` | Substitute default value |

```prolog
:- source(json, partial_data, [
    json_file('data/sparse.jsonl'),
    record_format(jsonl),
    columns(['required_field', 'optional_field']),
    null_policy(default('N/A'))
]).
```

## Record Extraction Features

### Output Formats

| Format | Flag | Description |
|--------|------|-------------|
| Full | `-f full` | Original Markdown block |
| Content | `-f content` | Just the code block content |
| JSON | `-f json` | Structured JSON output |

### Filtering

```bash
# Filter by record name pattern
perl scripts/utils/extract_records.pl \
  -q "unifyweaver\\.api\\..*" \
  docs/examples.md

# Filter by file type (frontmatter)
perl scripts/utils/extract_records.pl \
  --file-filter "file_type=UnifyWeaver Example Library" \
  docs/
```

### Record Types

Records can contain different code types - execute appropriately:

| Code Fence | Execute With |
|------------|--------------|
| ` ```bash ` | `bash script.sh` |
| ` ```prolog ` | `swipl -f init.pl -g "consult('code.pl')"` |
| ` ```python ` | `python3 script.py` |

```bash
# Extract bash script
perl scripts/utils/extract_records.pl -f content -q "setup" file.md > tmp/setup.sh
bash tmp/setup.sh

# Extract Prolog code
perl scripts/utils/extract_records.pl -f content -q "rules" file.md > tmp/rules.pl
swipl -f init.pl -g "consult('tmp/rules.pl'), run, halt"
```

## Validation

### JSON Source Validation

| Condition | Error |
|-----------|-------|
| Missing `columns/1` | `domain_error(json_columns, _)` |
| Column count ≠ arity | Arity mismatch error |
| Empty column name | `domain_error(json_column_entry, _)` |
| `return_object(true)` without `type_hint/1` | Type hint required |
| `return_object(true)` with arity ≠ 1 | Arity must be 1 |

### Record Format Validation

```prolog
% JSONL format - each line is independent
:- source(json, events, [
    json_file('data/events.jsonl'),
    record_format(jsonl),  % Required for .jsonl files
    columns(['event', 'timestamp'])
]).
```

## Common Patterns

### Processing Large JSON Files

```prolog
% Stream JSONL for memory efficiency
:- source(json, large_dataset, [
    json_file('data/large.jsonl'),
    record_format(jsonl),
    columns(['id', 'value']),
    null_policy(skip)
]).

process_all :-
    forall(large_dataset(Id, Value), process_record(Id, Value)).
```

### Joining JSON Sources

```prolog
:- source(json, users, [
    json_file('data/users.json'),
    columns(['id', 'name'])
]).

:- source(json, orders, [
    json_file('data/orders.json'),
    columns(['user_id', 'total'])
]).

user_totals(Name, Total) :-
    users(UserId, Name),
    orders(UserId, Total).
```

### Dynamic Source Declaration

```prolog
% Declare source at runtime
declare_json_source(Name, File, Columns) :-
    assert(source(json, Name, [
        json_file(File),
        columns(Columns)
    ])).
```

## Child Skills

- `skill_json_sources.md` - JSON/JSONL dynamic sources with full options
- `skill_extract_records.md` - Markdown record extraction tooling

## Related

**Parent Skill:**
- `skill_data_tools.md` - Data tools master

**Sibling Sub-Masters:**
- `skill_query_patterns.md` - Query and aggregation
- `skill_ml_tools.md` - Machine learning tools

**Documentation:**
- `docs/playbooks/parsing/README.md` - Parser tooling overview

**Code:**
- `src/unifyweaver/sources/json_source.pl` - JSON source implementation
- `scripts/utils/extract_records.pl` - Perl record extractor
