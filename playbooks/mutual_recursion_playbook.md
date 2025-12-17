# Playbook: Mutual Recursion (Even/Odd)

## Audience
This playbook is a high-level guide for coding agents. Agents orchestrate UnifyWeaver to demonstrate mutual recursion compilation.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "mutual_recursion" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use mutual recursion"


## Workflow Overview
Use UnifyWeaver to compile mutually recursive predicates:
1. Generate Prolog code for is_even/1 and is_odd/1 (mutually recursive).
2. Compile both predicates to bash using compiler_driver.
3. Execute and verify both generated scripts.

## Agent Inputs
Reference the following artifacts:
1. **Executable Record** – `unifyweaver.execution.mutual_recursion` in `playbooks/examples_library/recursion_examples.md`.
2. **Environment Setup** – `skills/skill_unifyweaver_environment.md`.
3. **Extraction Skill** – `skills/skill_extract_records.md`.
4. **Reference** – `docs/development/testing/playbooks/mutual_recursion_playbook__reference.md`.

## Execution Guidance

**IMPORTANT**: The record contains a **BASH SCRIPT**. Extract and run with `bash`.

### Step-by-Step Instructions

**Step 1-3**: Extract and prepare
```bash
cd /path/to/UnifyWeaver
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.mutual_recursion" \
  playbooks/examples_library/recursion_examples.md \
  > tmp/run_mutual_recursion.sh
chmod +x tmp/run_mutual_recursion.sh
```

**Step 4**: Run
```bash
bash tmp/run_mutual_recursion.sh
```

**Expected Output**:
```
Generating mutually recursive Prolog code...
✓ Generated Prolog code: tmp/even_odd.pl

Compiling mutually recursive predicates...
...
Generated scripts: [tmp/is_even.sh,tmp/is_odd.sh]

Testing generated scripts...
Testing is_even(4):
  ✓ 4 is even
Testing is_odd(3):
  ✓ 3 is odd

Success: Mutual recursion compiled and executed
```

## Expected Outcome
- Both is_even.sh and is_odd.sh generated
- is_even(4) succeeds, is_odd(3) succeeds
- Demonstrates mutual recursion compilation
- Exit code 0

## Citations
[1] playbooks/examples_library/recursion_examples.md
[2] skills/skill_unifyweaver_environment.md
[3] docs/development/testing/playbooks/mutual_recursion_playbook__reference.md
