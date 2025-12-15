# Playbook: C# Query Compilation and Execution

## Audience
This playbook is a high-level guide for coding agents to orchestrate UnifyWeaver for C# code generation, compilation, and execution using the `csharp_query` target.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "csharp_query" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use csharp query"


## Workflow Overview
This playbook demonstrates the complete Prolog-to-C# compilation workflow for arithmetic predicates:
1. Define a UnifyWeaver program with facts and arithmetic rules using `is/2`.
2. Compile the program to C# using the `csharp_target` compiler.
3. Verify the generated C# code.

## Agent Inputs
Reference the following artifacts:
1. **Bash Executable Record** – `unifyweaver.execution.csharp_sum_pair` in `playbooks/examples_library/csharp_examples.md`.
2. **PowerShell Executable Record** – `unifyweaver.execution.csharp_sum_pair_ps` in `playbooks/examples_library/csharp_examples.md`.
3. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md`.
4. **Extraction Skill** – `skills/skill_extract_records.md`.

## Execution Guidance

### Step-by-Step Instructions

An agent should choose the appropriate script based on the execution environment.

#### For Linux/macOS (bash) environment:

**Step 1: Navigate to project root**
```bash
cd /path/to/UnifyWeaver
```

**Step 2: Extract the bash script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.csharp_sum_pair" \
  playbooks/examples_library/csharp_examples.md \
  > tmp/run_csharp_sum_pair.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_csharp_sum_pair.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_csharp_sum_pair.sh
```

#### For Windows (PowerShell) environment:

**Step 1: Navigate to project root**
```powershell
cd C:\path\to\UnifyWeaver
```

**Step 2: Extract the PowerShell script**
```powershell
perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.csharp_sum_pair_ps" playbooks/examples_library/csharp_examples.md | Out-File -FilePath tmp/run_csharp_sum_pair.ps1
```

**Step 3: Run the PowerShell script**
```powershell
./tmp/run_csharp_sum_pair.ps1
```

**Expected Output**:
```
Compiling Prolog to C#...
C# code generated successfully.
[Generated C# code will be displayed]
Success: C# program compiled successfully.
```

### What the Script Does
The script will:
1. Create a Prolog script in `tmp/sum_pair_csharp.pl` containing facts and a rule using `is/2` arithmetic.
2. Compile the Prolog script to C# using `compile_predicate_to_csharp/3`, creating `tmp/csharp_sum_project/sum_pair.cs`.
3. Display the generated C# code.

## Expected Outcome
- Successful execution will print "Success: C# program compiled successfully."
- The generated C# code will be displayed.
- Exit code 0.

## Citations
[1] playbooks/examples_library/csharp_examples.md (`unifyweaver.execution.csharp_sum_pair`, `unifyweaver.execution.csharp_sum_pair_ps`)
[2] skills/skill_unifyweaver_environment.md
[3] skills/skill_extract_records.md
