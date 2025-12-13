# AWK Source Examples

This file contains executable records for the awk_source playbook.

## Example 1: Basic AWK Command

Query: `awk_command_basic`

```bash
#!/bin/bash
# AWK Source Demo - Basic Command Binding
set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Source Demo: Basic Command ==="

# Create test data
mkdir -p tmp
cat > tmp/test_lines.txt << 'EOF'
Line 1
Line 2
Line 3
Line 4
Line 5
EOF

# Create Prolog script using awk_source
cat > tmp/awk_basic.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Declare AWK command as dynamic source
:- dynamic_source(count_lines/1, awk, [
    awk_command('END { print NR }'),
    input_file('tmp/test_lines.txt')
]).

% Compile to bash
main :-
    compile_dynamic_source(count_lines/1, [], 'tmp/count_lines.sh'),
    format("Generated: tmp/count_lines.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

# Compile
swipl tmp/awk_basic.pl

# Test
echo "Testing count_lines/1:"
bash tmp/count_lines.sh count_lines
echo ""
echo "Success: AWK command binding works"
```

## Example 2: AWK File Binding

Query: `awk_file_basic`

```bash
#!/bin/bash
# AWK Source Demo - File Binding
set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Source Demo: AWK File ==="

# Create test data
mkdir -p tmp
cat > tmp/users.txt << 'EOF'
1	Alice	admin
2	Bob	user
3	Charlie	guest
EOF

# Create AWK script
cat > tmp/extract_field.awk << 'AWK'
{ print $2 }
AWK

# Create Prolog script using awk_file
cat > tmp/awk_file.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Declare AWK file as dynamic source
:- dynamic_source(extract_field/2, awk, [
    awk_file('tmp/extract_field.awk'),
    input_file('tmp/users.txt')
]).

% Compile to bash
main :-
    compile_dynamic_source(extract_field/2, [], 'tmp/extract_field.sh'),
    format("Generated: tmp/extract_field.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

# Compile
swipl tmp/awk_file.pl

# Test
echo "Testing extract_field/2 with field 2:"
bash tmp/extract_field.sh extract_field 2
echo ""
echo "Success: AWK file binding works"
```

## Example 3: AWK with Field Separator

Query: `awk_separator`

```bash
#!/bin/bash
# AWK Source Demo - Custom Separator
set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Source Demo: Field Separator ==="

# Create CSV test data
mkdir -p tmp
cat > tmp/data.csv << 'EOF'
id,name,score
1,Alice,85
2,Bob,92
3,Charlie,78
EOF

# Create Prolog script with field separator
cat > tmp/awk_sep.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Declare AWK source with comma separator
:- dynamic_source(parse_csv/1, awk, [
    awk_command('NR > 1 { printf "id:%s,name:%s,score:%s\n", $1, $2, $3 }'),
    input_file('tmp/data.csv'),
    field_separator(',')
]).

% Compile to bash
main :-
    compile_dynamic_source(parse_csv/1, [], 'tmp/parse_csv.sh'),
    format("Generated: tmp/parse_csv.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

# Compile
swipl tmp/awk_sep.pl

# Test
echo "Testing parse_csv/1:"
bash tmp/parse_csv.sh parse_csv
echo ""
echo "Success: AWK field separator works"
```

## Example 4: AWK Aggregation

Query: `awk_aggregate`

```bash
#!/bin/bash
# AWK Source Demo - Aggregation
set -euo pipefail
cd /root/UnifyWeaver

echo "=== AWK Source Demo: Aggregation ==="

# Create test data
mkdir -p tmp
cat > tmp/sales.tsv << 'EOF'
region	amount
North	100
South	150
North	200
East	75
South	125
EOF

# Create Prolog script for aggregation
cat > tmp/awk_agg.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').

% Declare AWK aggregation
:- dynamic_source(total_by_region/1, awk, [
    awk_command('
        NR > 1 { sum[$1] += $2 }
        END { for (r in sum) print r ":" sum[r] }
    '),
    input_file('tmp/sales.tsv')
]).

% Compile to bash
main :-
    compile_dynamic_source(total_by_region/1, [], 'tmp/total_by_region.sh'),
    format("Generated: tmp/total_by_region.sh~n"),
    halt(0).

:- initialization(main, main).
PROLOG

# Compile
swipl tmp/awk_agg.pl

# Test
echo "Testing total_by_region/1:"
bash tmp/total_by_region.sh total_by_region | sort
echo ""
echo "Success: AWK aggregation works"
```
