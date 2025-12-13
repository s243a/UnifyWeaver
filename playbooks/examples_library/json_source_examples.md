# JSON Source Examples

This file contains executable records for the json_source playbook.

## Example 1: Basic jq Filter

Query: `json_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== JSON Source Demo: Basic jq ==="

mkdir -p tmp
cat > tmp/users.json << 'EOF'
[
  {"id": 1, "name": "Alice", "score": 85},
  {"id": 2, "name": "Bob", "score": 92},
  {"id": 3, "name": "Charlie", "score": 78}
]
EOF

cat > tmp/json_basic.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(get_names/1, json, [
    json_file('tmp/users.json'),
    jq_filter('.[].name')
]).

main :-
    compile_dynamic_source(get_names/1, [], 'tmp/get_names.sh'),
    format("Generated: tmp/get_names.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/json_basic.pl
echo "Testing get_names/1:"
bash tmp/get_names.sh get_names
echo "Success: JSON basic filtering works"
```

## Example 2: jq Conditional Filter

Query: `json_filter`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== JSON Source Demo: jq Filtering ==="

mkdir -p tmp
cat > tmp/scores.json << 'EOF'
[
  {"name": "Alice", "score": 85},
  {"name": "Bob", "score": 92},
  {"name": "Charlie", "score": 78},
  {"name": "Diana", "score": 95}
]
EOF

cat > tmp/json_filter.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(high_scorers/1, json, [
    json_file('tmp/scores.json'),
    jq_filter('.[] | select(.score > 90) | .name + ":" + (.score | tostring)')
]).

main :-
    compile_dynamic_source(high_scorers/1, [], 'tmp/high_scorers.sh'),
    format("Generated: tmp/high_scorers.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/json_filter.pl
echo "Testing high_scorers/1:"
bash tmp/high_scorers.sh high_scorers
echo "Success: JSON filtering works"
```

## Example 3: jq Array to TSV

Query: `json_array`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== JSON Source Demo: Array to TSV ==="

mkdir -p tmp
cat > tmp/data.json << 'EOF'
[
  {"id": 1, "name": "Alice", "score": 85},
  {"id": 2, "name": "Bob", "score": 92},
  {"id": 3, "name": "Charlie", "score": 78}
]
EOF

cat > tmp/json_array.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(extract_fields/1, json, [
    json_file('tmp/data.json'),
    jq_filter('.[] | [.id, .name, .score] | join(":")')
]).

main :-
    compile_dynamic_source(extract_fields/1, [], 'tmp/extract_fields.sh'),
    format("Generated: tmp/extract_fields.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/json_array.pl
echo "Testing extract_fields/1:"
bash tmp/extract_fields.sh extract_fields
echo "Success: JSON array processing works"
```
