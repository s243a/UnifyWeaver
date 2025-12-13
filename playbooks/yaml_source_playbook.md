# Playbook: YAML Data Source

## Audience
This playbook demonstrates how to process YAML files as data sources using UnifyWeaver's yaml_source plugin (via Python PyYAML).

## Overview
The `yaml_source` plugin lets you declare Prolog predicates that read and filter YAML data. It uses Python's PyYAML library under the hood.

## When to Use This Playbook

✅ **Use yaml_source when:**
- Processing configuration files (Kubernetes, Docker Compose, etc.)
- Reading structured YAML data
- Filtering YAML with Python expressions
- Converting YAML to Prolog facts

## Agent Inputs

Reference the following artifacts:
1. **Executable Records** – `playbooks/examples_library/yaml_source_examples.md`
2. **Source Module** – `src/unifyweaver/sources/yaml_source.pl`

## Execution Guidance

### Example 1: Basic YAML File Reading

```bash
cd /path/to/UnifyWeaver

perl scripts/extract_records.pl playbooks/examples_library/yaml_source_examples.md \
    yaml_basic > tmp/yaml_basic.sh
chmod +x tmp/yaml_basic.sh
bash tmp/yaml_basic.sh
```

**Expected Output:**
```
Compiling YAML source: read_config/1
Generated: tmp/read_config.sh
Testing read_config/1:
appname:MyApp
version:1.0.0
port:8080
```

### Example 2: YAML with Filtering

```bash
perl scripts/extract_records.pl playbooks/examples_library/yaml_source_examples.md \
    yaml_filter > tmp/yaml_filter.sh
chmod +x tmp/yaml_filter.sh
bash tmp/yaml_filter.sh
```

**Expected Output:**
```
Compiling YAML source: get_users/1
Generated: tmp/get_users.sh
Testing get_users/1:
alice:admin
bob:editor
charlie:viewer
```

### Example 3: YAML Array Processing

```bash
perl scripts/extract_records.pl playbooks/examples_library/yaml_source_examples.md \
    yaml_array > tmp/yaml_array.sh
chmod +x tmp/yaml_array.sh
bash tmp/yaml_array.sh
```

**Expected Output:**
```
Compiling YAML source: list_services/1
Generated: tmp/list_services.sh
Testing list_services/1:
web
api
database
cache
```

## Configuration Options

- `yaml_file(File)` - YAML file to read
- `yaml_stdin(true)` - Read YAML from stdin
- `yaml_filter(Expr)` - Python expression to filter data (default: `"data"`)
- `python_interpreter(Path)` - Python interpreter path (default: `python3`)

## See Also

- `playbooks/json_litedb_playbook.md` - JSON processing
- `playbooks/xml_data_source_playbook.md` - XML processing
- `playbooks/csv_data_source_playbook.md` - CSV processing

## Summary

**Key Concepts:**
- ✅ Process YAML files from Prolog
- ✅ Filter YAML with Python expressions
- ✅ Supports files and stdin
- ✅ Handles complex nested structures
- ✅ Kubernetes/Docker Compose compatible
