# Playbook: Prolog Code Generation and Compilation

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents do not handwrite scripts here—they orchestrate UnifyWeaver to generate Prolog code, compile it to bash, and verify execution.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "prolog_generation" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use prolog generation"


## Workflow Overview
Use UnifyWeaver to demonstrate the complete Prolog-to-bash compilation workflow:
1. Generate Prolog code for a specific task (factorial calculation).
2. Save the Prolog code to a file.
3. Use UnifyWeaver's compiler_driver to transpile Prolog to bash.
4. Execute and verify the generated bash script.

## Agent Inputs
Reference the following artifacts instead of embedding raw commands:
1. **Executable Record** – `unifyweaver.execution.generate_factorial` in `playbooks/examples_library/prolog_generation_examples.md`.
2. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md` explains how to set up the Prolog environment with init.pl.
3. **Parser Catalog** – `docs/playbooks/parsing/README.md` lists the available extractors (Perl, Python, `parsc`) and usage order.
4. **Extraction Skill** – `skills/skill_extract_records.md` documents CLI flags and environment notes.
5. **Reviewer Reference** – `docs/development/testing/playbooks/prolog_generation_playbook__reference.md` for validation details.

## Execution Guidance

**IMPORTANT**: The record in [1] contains a **BASH SCRIPT**, not Prolog code. You must extract it and run it with `bash`, not with `swipl`.

### Step-by-Step Instructions

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.generate_factorial" \
  playbooks/examples_library/prolog_generation_examples.md \
  > tmp/run_factorial_example.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_factorial_example.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_factorial_example.sh
```

**Expected Output**:
```
Generating Prolog code for factorial...
✓ Generated Prolog code: tmp/factorial.pl

Compiling Prolog to bash using UnifyWeaver...
[UnifyWeaver] Environment initialized
  ...
Generated scripts: [tmp/factorial.sh]

Testing generated factorial script...
Running factorial(5):
5:120

Success: Factorial compiled and executed correctly
```

### What the Script Does

The bash script you extracted will:
1. Generate Prolog code for factorial calculation
2. Save it to `tmp/factorial.pl`
3. Use `init.pl` to set up the UnifyWeaver environment
4. Compile the Prolog code to bash using compiler_driver
5. Execute the generated bash script with test input
6. Verify factorial(5) = 120

### Common Mistakes to Avoid

❌ **DO NOT** try to consult the extracted file as Prolog:
```bash
# WRONG - This will fail!
swipl -g "consult('tmp/run_factorial_example.sh'), ..."
```

✅ **DO** run it as a bash script:
```bash
# CORRECT
bash tmp/run_factorial_example.sh
```

## Expected Outcome
- Successful runs show "Success: Factorial compiled and executed correctly"
- factorial(5) produces output `5:120`
- Exit code 0
- Generated artifacts in `tmp/` directory

## Citations
[1] playbooks/examples_library/prolog_generation_examples.md (`unifyweaver.execution.generate_factorial`)
[2] skills/skill_unifyweaver_environment.md
[3] docs/playbooks/parsing/README.md
[4] skills/skill_extract_records.md
[5] docs/development/testing/playbooks/prolog_generation_playbook__reference.md
