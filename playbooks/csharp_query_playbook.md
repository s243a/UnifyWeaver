# Playbook: C# Query Compilation and Execution

## Audience
This playbook is a high-level guide for coding agents to orchestrate UnifyWeaver for C# code generation, compilation, and execution using the `csharp_query` target.

## Workflow Overview
This playbook demonstrates the complete Prolog-to-C# compilation workflow for a recursive predicate:
1. Define a UnifyWeaver program with a recursive predicate (e.g., Fibonacci).
2. Compile the program to C# using the `csharp_query` target.
3. Execute the compiled C# program.
4. Verify the output.

## Agent Inputs
Reference the following artifacts:
1. **Bash Executable Record** – `unifyweaver.execution.csharp_fibonacci` in `playbooks/examples_library/csharp_examples.md`.
2. **PowerShell Executable Record** – `unifyweaver.execution.csharp_fibonacci_ps` in `playbooks/examples_library/csharp_examples.md`.
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
  -q "unifyweaver.execution.csharp_fibonacci" \
  playbooks/examples_library/csharp_examples.md \
  > tmp/run_csharp_fibonacci.sh
```

**Step 3: Make it executable**
```bash
chmod +x tmp/run_csharp_fibonacci.sh
```

**Step 4: Run the bash script**
```bash
bash tmp/run_csharp_fibonacci.sh
```

#### For Windows (PowerShell) environment:

**Step 1: Navigate to project root**
```powershell
cd C:\path\to\UnifyWeaver
```

**Step 2: Extract the PowerShell script**
```powershell
perl scripts/utils/extract_records.pl -f content -q "unifyweaver.execution.csharp_fibonacci_ps" playbooks/examples_library/csharp_examples.md | Out-File -FilePath tmp/run_csharp_fibonacci.ps1
```

**Step 3: Run the PowerShell script**
```powershell
./tmp/run_csharp_fibonacci.ps1
```

**Expected Output**:
```
Compiling Prolog to C#...
Executing C# program...
8: 21
Success: C# program compiled and executed successfully.
```

### What the Script Does
The script will:
1. Create a Prolog script in `tmp/fib_csharp.pl` containing a recursive Fibonacci definition.
2. Compile the Prolog script to C# using the `csharp_query` target, creating a C# project in `tmp/csharp_fib_project`.
3. Run the C# project, which will calculate and print the 8th Fibonacci number.

## Expected Outcome
- Successful execution will print "Success: C# program compiled and executed successfully."
- The output will contain the line "8: 21".
- Exit code 0.

## Citations
[1] playbooks/examples_library/csharp_examples.md (`unifyweaver.execution.csharp_fibonacci`, `unifyweaver.execution.csharp_fibonacci_ps`)
[2] skills/skill_unifyweaver_environment.md
[3] skills/skill_extract_records.md
