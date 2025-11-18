# Playbook: Tree Recursion (Tree Sum)

## Audience
This playbook is a high-level guide for coding agents. Agents orchestrate UnifyWeaver to demonstrate tree recursion compilation.

## Workflow Overview
Use UnifyWeaver to compile tree recursive predicates:
1. Generate Prolog code for tree_sum/2 (tree recursion).
2. Compile the predicate to bash using compiler_driver.
3. Execute and verify with tree data structures.

## Agent Inputs
Reference the following artifacts:
1. **Executable Record** – `unifyweaver.execution.tree_recursion` in `playbooks/examples_library/recursion_examples.md`.
2. **Environment Setup** – `skills/skill_unifyweaver_environment.md`.
3. **Extraction Skill** – `skills/skill_extract_records.md`.
4. **Reference** – `docs/development/testing/playbooks/tree_recursion_playbook__reference.md`.

## Execution Guidance

**IMPORTANT**: The record contains a **BASH SCRIPT**. Extract and run with `bash`.

### Step-by-Step Instructions

**Step 1-3**: Extract and prepare
```bash
cd /path/to/UnifyWeaver
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.tree_recursion" \
  playbooks/examples_library/recursion_examples.md \
  > tmp/run_tree_recursion.sh
chmod +x tmp/run_tree_recursion.sh
```

**Step 4**: Run
```bash
bash tmp/run_tree_recursion.sh
```

**Expected Output**:
```
Generating tree recursive Prolog code...
✓ Generated Prolog code: tmp/tree_sum.pl

Compiling tree recursive predicate...
...
Generated scripts: [tmp/tree_sum.sh]

Testing generated script...
Testing tree_sum([10,[5,[],[]],[15,[],[]]])
[10,[5,[],[]],[15,[],[]]]:30

Success: Tree recursion compiled and executed
```

## Expected Outcome
- tree_sum.sh generated successfully
- tree_sum([10,[5,[],[]],[15,[],[]]], 30) produces correct result
- Demonstrates tree recursion compilation
- Exit code 0

## Citations
[1] playbooks/examples_library/recursion_examples.md
[2] skills/skill_unifyweaver_environment.md
[3] docs/development/testing/playbooks/tree_recursion_playbook__reference.md
