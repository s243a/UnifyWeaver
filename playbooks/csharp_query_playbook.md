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
This playbook demonstrates the Prolog-to-C# compilation workflow for arithmetic predicates in query mode:
1. Define a UnifyWeaver program with facts and arithmetic rules using `is/2`.
2. Compile the predicate to C# using the `csharp_target` query compiler.
3. Inspect the generated C# code.

`is/2` (derived columns) is supported by the query runtime when RHS variables are already bound.

For recursive arithmetic (e.g., Fibonacci), query mode requires **parameterized query mode** (`user:mode/1` inputs) and currently has additional constraints; see:
- `docs/development/proposals/PARAMETERIZED_QUERIES_STATUS.md`
- `docs/development/analysis/IS_PREDICATE_COMPATIBILITY_ANALYSIS.md`
- `playbooks/csharp_generator_playbook.md` (generator mode fallback)

## Multiple mode entrypoints (query mode)
If you declare multiple concrete `user:mode/1` facts for the same predicate (e.g., `mode(p(+, -)).` and `mode(p(-, +)).`), query-mode codegen emits multiple C# entrypoints in the generated module:
- `BuildAllOutput()` for all-output mode (if declared or defaulted)
- `BuildIn0()`, `BuildIn1()`, ... for input-position variants
- `Build()` aliases the most general variant
- `BuildForInputs(...)` selects a variant by input positions (e.g., `BuildForInputs(0)`).

On the Prolog side, you can also build/select the corresponding plan variants:
```prolog
?- csharp_target:build_query_plans(p/2, [target(csharp_query)], Plans).
?- csharp_target:build_query_plan_for_inputs(p/2, [target(csharp_query)], [0], Plan).
```

Example usage (C#):
```csharp
var (provider, plan) = YourPredicateQueryModule.BuildForInputs(0); // arg0 is an input
var executor = new QueryExecutor(provider);
var rows = executor.Execute(plan, new[] { new object[] { "alice" } });
```

## Agent Inputs
Reference the following artifacts:
1. **Bash Executable Record** – `unifyweaver.execution.csharp_sum_pair` in `playbooks/examples_library/csharp_examples.md`.
2. **PowerShell Executable Record** – `unifyweaver.execution.csharp_sum_pair_ps` in `playbooks/examples_library/csharp_examples.md`.
3. **Bash Executable Record** – `unifyweaver.execution.csharp_fib_param_query` in `playbooks/examples_library/csharp_examples.md`.
4. **PowerShell Executable Record** – `unifyweaver.execution.csharp_fib_param_query_ps` in `playbooks/examples_library/csharp_examples.md`.
5. **Environment Setup Skill** – `skills/skill_unifyweaver_environment.md`.
6. **Extraction Skill** – `skills/skill_extract_records.md`.

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
mkdir -p tmp
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
New-Item -ItemType Directory -Force -Path tmp | Out-Null
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
2. Compile `sum_pair/3` to C# using the `csharp_query` target, writing `tmp/csharp_sum_project/sum_pair.cs`.
3. Display the generated C# code.

## Expected Outcome
- Successful execution will print "C# code generated successfully."
- The generated C# code will be displayed.
- Exit code 0.

## Citations
[1] playbooks/examples_library/csharp_examples.md (`unifyweaver.execution.csharp_sum_pair`, `unifyweaver.execution.csharp_sum_pair_ps`)
[2] skills/skill_unifyweaver_environment.md
[3] skills/skill_extract_records.md
