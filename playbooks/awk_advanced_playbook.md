# Playbook: AWK Advanced Patterns

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to compile Prolog predicates into optimized AWK scripts.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "awk_advanced" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use awk advanced"


## Workflow Overview
Use UnifyWeaver's awk_target module to:
1. Compile Prolog facts to AWK associative array lookups
2. Transform aggregation operations (sum, count, max, min, avg) to AWK
3. Convert tail-recursive predicates to AWK while loops
4. Generate constraint-based filters as AWK conditionals

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - `playbooks/examples_library/awk_advanced_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step 1: Navigate to project root
```bash
cd /root/UnifyWeaver
```

### Step 2: Extract the aggregation demo
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.awk_aggregation" \
  playbooks/examples_library/awk_advanced_examples.md \
  > tmp/run_awk_aggregation.sh
```

### Step 3: Make it executable and run
```bash
chmod +x tmp/run_awk_aggregation.sh
bash tmp/run_awk_aggregation.sh
```

**Expected Output**:
```
=== AWK Advanced Demo: Aggregation Patterns ===

Created test data: tmp/awk_demo/sales.tsv

Running Prolog to generate AWK aggregation scripts...

=== Generating AWK Aggregation Scripts ===

1. Generating SUM aggregation...
=== Compiling sales/4 to AWK ===
  Aggregation type: sum
[AwkTarget] Generated executable AWK script: tmp/awk_demo/sum_sales.awk

...

SUM of quantities: 465
COUNT of records: 6
MAX quantity: 120
AVG quantity: 77.5

Success: AWK aggregation demo complete
```

### Step 4: Test tail recursion compilation (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.awk_tail_recursion" \
  playbooks/examples_library/awk_advanced_examples.md \
  > tmp/run_awk_tail_rec.sh
chmod +x tmp/run_awk_tail_rec.sh
bash tmp/run_awk_tail_rec.sh
```

### Step 5: Test constraint compilation (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.awk_constraints" \
  playbooks/examples_library/awk_advanced_examples.md \
  > tmp/run_awk_constraints.sh
chmod +x tmp/run_awk_constraints.sh
bash tmp/run_awk_constraints.sh
```

### Step 6: View module info (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.awk_module_info" \
  playbooks/examples_library/awk_advanced_examples.md \
  > tmp/run_awk_info.sh
chmod +x tmp/run_awk_info.sh
bash tmp/run_awk_info.sh
```

## What This Playbook Demonstrates

1. **awk_target module** (`src/unifyweaver/targets/awk_target.pl`):
   - `compile_predicate_to_awk/3` - Main compilation function
   - `write_awk_script/2` - Write and chmod AWK script

2. **Compilation patterns**:
   - **Facts**: Prolog facts -> AWK associative array lookups
   - **Aggregation**: sum/count/max/min/avg -> AWK BEGIN/END blocks
   - **Tail Recursion**: Recursive predicates -> AWK while loops
   - **Constraints**: Comparison operators -> AWK conditionals
   - **Regex Matching**: match/4 -> AWK regex patterns

3. **Configuration options**:
   - `record_format(jsonl|tsv|csv)` - Input format (default: tsv)
   - `field_separator(Char)` - Field separator (default: tab)
   - `unique(true|false)` - Deduplicate results
   - `aggregation(sum|count|max|min|avg)` - Aggregation operation

## Example: Tail Recursion to While Loop

### Prolog (factorial with accumulator):
```prolog
factorial(0, Acc, Acc).
factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, Result).
```

### Compiled AWK (while loop):
```awk
BEGIN { FS = "\t" }
{
    n = $1; acc = $2
    while (n > 0) {
        acc1 = (acc * n)
        n1 = (n - 1)
        acc = acc1
        n = n1
    }
    print acc
}
```

### Usage:
```bash
echo -e "5\t1" | awk -f factorial.awk
# Output: 120
```

## AWK Regex Support

Supported regex types for AWK target:
- `auto` - Auto-detect (default)
- `ere` - Extended Regular Expressions
- `bre` - Basic Regular Expressions
- `awk` - AWK-native regex

NOT supported (will error):
- `pcre` - Perl-Compatible Regex
- `python` - Python regex
- `dotnet` - .NET regex

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** ensure awk/gawk is installed before testing
- **DO** use tab-separated input by default (or set `field_separator`)
- **DO** define predicates as `user:predicate/N` for the compiler to find them

## Expected Outcome
- Generated AWK scripts from Prolog predicates
- Efficient streaming data processing
- Tail recursion transformed to iterative loops

## Citations
[1] playbooks/examples_library/awk_advanced_examples.md
[2] src/unifyweaver/targets/awk_target.pl
[3] skills/skill_unifyweaver_environment.md

