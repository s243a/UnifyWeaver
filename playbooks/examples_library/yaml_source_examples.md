# YAML Source Examples

This file contains executable records for the yaml_source playbook.

## Example 1: Basic YAML Reading

Query: `yaml_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== YAML Source Demo: Basic Reading ==="

mkdir -p tmp
cat > tmp/config.yml << 'EOF'
appname: MyApp
version: 1.0.0
port: 8080
debug: true
EOF

cat > tmp/yaml_basic.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(read_config/1, yaml, [
    yaml_file('tmp/config.yml'),
    yaml_filter('[(k + ":" + str(v)) for k, v in data.items()]')
]).

main :-
    compile_dynamic_source(read_config/1, [], 'tmp/read_config.sh'),
    format("Generated: tmp/read_config.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/yaml_basic.pl
echo "Testing read_config/1:"
bash tmp/read_config.sh read_config
echo "Success: YAML basic reading works"
```

## Example 2: YAML Filtering

Query: `yaml_filter`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== YAML Source Demo: Filtering ==="

mkdir -p tmp
cat > tmp/users.yml << 'EOF'
users:
  - name: alice
    role: admin
  - name: bob
    role: editor
  - name: charlie
    role: viewer
EOF

cat > tmp/yaml_filter.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(get_users/1, yaml, [
    yaml_file('tmp/users.yml'),
    yaml_filter('[u["name"] + ":" + u["role"] for u in data["users"]]')
]).

main :-
    compile_dynamic_source(get_users/1, [], 'tmp/get_users.sh'),
    format("Generated: tmp/get_users.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/yaml_filter.pl
echo "Testing get_users/1:"
bash tmp/get_users.sh get_users
echo "Success: YAML filtering works"
```

## Example 3: YAML Array

Query: `yaml_array`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== YAML Source Demo: Arrays ==="

mkdir -p tmp
cat > tmp/services.yml << 'EOF'
services:
  - web
  - api
  - database
  - cache
EOF

cat > tmp/yaml_array.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(list_services/1, yaml, [
    yaml_file('tmp/services.yml'),
    yaml_filter('data["services"]')
]).

main :-
    compile_dynamic_source(list_services/1, [], 'tmp/list_services.sh'),
    format("Generated: tmp/list_services.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/yaml_array.pl
echo "Testing list_services/1:"
bash tmp/list_services.sh list_services
echo "Success: YAML array processing works"
```
