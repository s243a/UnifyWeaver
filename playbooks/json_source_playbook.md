# Playbook: JSON Data Source (jq)

## Audience
This playbook demonstrates JSON processing using jq through UnifyWeaver's json_source plugin.

## Overview
The `json_source` plugin uses jq for powerful JSON filtering and transformation. Unlike json_litedb (which uses LiteDB), this is lightweight and jq-based.

## When to Use

✅ **Use json_source when:**
- Processing JSON with jq filters
- Need lightweight JSON handling
- Working with JSON APIs/files
- Want jq's powerful query language

❌ **Use json_litedb when:**
- Need a JSON document database
- Require complex queries and indexing

## Agent Inputs

1. **Executable Records** – `playbooks/examples_library/json_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/json_source.pl`

## Execution Guidance

### Example 1: Basic jq Filtering

```bash
cd /path/to/UnifyWeaver

perl scripts/extract_records.pl playbooks/examples_library/json_source_examples.md \
    json_basic > tmp/json_basic.sh
chmod +x tmp/json_basic.sh
bash tmp/json_basic.sh
```

**Expected Output:**
```
Compiling JSON source: get_names/1
Generated: tmp/get_names.sh
Testing get_names/1:
Alice
Bob
Charlie
```

### Example 2: jq with Filtering

```bash
perl scripts/extract_records.pl playbooks/examples_library/json_source_examples.md \
    json_filter > tmp/json_filter.sh
chmod +x tmp/json_filter.sh
bash tmp/json_filter.sh
```

**Expected Output:**
```
Compiling JSON source: high_scorers/1
Generated: tmp/high_scorers.sh
Testing high_scorers/1:
Bob:92
Diana:95
```

### Example 3: jq Array Processing

```bash
perl scripts/extract_records.pl playbooks/examples_library/json_source_examples.md \
    json_array > tmp/json_array.sh
chmod +x tmp/json_array.sh
bash tmp/json_array.sh
```

**Expected Output:**
```
Compiling JSON source: extract_fields/1
Generated: tmp/extract_fields.sh
Testing extract_fields/1:
1:Alice:85
2:Bob:92
3:Charlie:78
```

## Configuration Options

- `jq_filter(Filter)` - jq filter expression (required)
- `json_file(File)` - JSON file to process
- `json_stdin(true)` - Read JSON from stdin
- `output_format(Format)` - Output format: `tsv`, `json`, `raw`, `csv`

## jq Filter Examples

```prolog
% Extract array elements
jq_filter('.items[]')

% Filter by condition
jq_filter('.users[] | select(.score > 90)')

% Transform structure
jq_filter('.[] | {name: .name, total: (.quantity * .price)}')

% Multiple fields
jq_filter('[.[] | .id, .name, .score] | @tsv')
```

## See Also

- `playbooks/json_litedb_playbook.md` - LiteDB JSON database
- `playbooks/yaml_source_playbook.md` - YAML processing
- `playbooks/http_source_playbook.md` - HTTP + JSON

## Summary

**Key Concepts:**
- ✅ Lightweight JSON processing with jq
- ✅ Powerful jq query language
- ✅ Multiple output formats
- ✅ File and stdin support
- ✅ Perfect for API data processing
