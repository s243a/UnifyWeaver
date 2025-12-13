# Bash Pipeline Examples

This file contains executable records for the bash_pipeline_source playbook.

## Example 1: Grep + AWK

Query: `pipeline_basic`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Bash Pipeline Demo: Grep + AWK ==="

mkdir -p tmp
cat > tmp/logs.txt << 'EOF'
INFO:Application started
ERROR:Connection failed
WARN:Slow response
ERROR:Timeout occurred
INFO:Request processed
EOF

cat > tmp/pipeline_basic.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(find_errors/1, bash_pipeline, [
    stages([
        stage(grep, 'grep', ['ERROR']),
        stage(awk, 'awk', ['-F:', '{print $1":"$2}'])
    ]),
    input_file('tmp/logs.txt')
]).

main :-
    compile_dynamic_source(find_errors/1, [], 'tmp/find_errors.sh'),
    format("Generated: tmp/find_errors.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/pipeline_basic.pl
echo "Testing find_errors/1:"
bash tmp/find_errors.sh find_errors
echo "Success: Grep + AWK pipeline works"
```

## Example 2: Multi-Stage Processing

Query: `pipeline_complex`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Bash Pipeline Demo: Multi-Stage ==="

mkdir -p tmp
cat > tmp/sales.txt << 'EOF'
Widget:100
Gadget:50
Widget:200
Gadget:75
Widget:150
Gadget:125
EOF

cat > tmp/pipeline_complex.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(top_sellers/1, bash_pipeline, [
    stages([
        stage(awk, 'awk', ['-F:', 'BEGIN{OFS=":"} {sum[$1]+=$2} END{for(p in sum) print p,sum[p]}']),
        stage(sort, 'sort', ['-t:', '-k2', '-nr'])
    ]),
    input_file('tmp/sales.txt')
]).

main :-
    compile_dynamic_source(top_sellers/1, [], 'tmp/top_sellers.sh'),
    format("Generated: tmp/top_sellers.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/pipeline_complex.pl
echo "Testing top_sellers/1:"
bash tmp/top_sellers.sh top_sellers
echo "Success: Multi-stage pipeline works"
```

## Example 3: Sort + Uniq

Query: `pipeline_aggregate`

```bash
#!/bin/bash
set -euo pipefail
cd /root/UnifyWeaver

echo "=== Bash Pipeline Demo: Sort + Uniq ==="

mkdir -p tmp
cat > tmp/users.txt << 'EOF'
alice
bob
charlie
alice
bob
alice
EOF

cat > tmp/pipeline_aggregate.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

:- dynamic_source(unique_users/1, bash_pipeline, [
    stages([
        stage(sort, 'sort', []),
        stage(uniq, 'uniq', [])
    ]),
    input_file('tmp/users.txt')
]).

main :-
    compile_dynamic_source(unique_users/1, [], 'tmp/unique_users.sh'),
    format("Generated: tmp/unique_users.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

swipl tmp/pipeline_aggregate.pl
echo "Testing unique_users/1:"
bash tmp/unique_users.sh unique_users
echo "Success: Sort + Uniq pipeline works"
```
