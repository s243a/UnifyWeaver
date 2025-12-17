# Playbook: Cross-Target Glue

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to generate cross-target pipeline scripts that combine AWK, Python, Bash, Go, and Rust processors.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "cross_target_glue" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
python3 scripts/skills/lookup_example.py "how to use cross target glue"


## Workflow Overview
Use UnifyWeaver's glue modules to:
1. Generate pipeline scripts that combine multiple target languages
2. Use shell_glue for AWK/Python/Bash integration
3. Use native_glue for Go/Rust binary orchestration
4. Chain processors via Unix pipes for streaming data

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - Phase-specific records in `playbooks/examples_library/cross_target_glue_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

This playbook has multiple phases. Start with Phase 1 (shell integration) which requires only AWK, Python, and Bash. Later phases require Go, Rust, or .NET.

### Phase 1: Shell Integration (AWK + Python + Bash)

**Step 1: Navigate to project root**
```bash
cd /root/UnifyWeaver
```

**Step 2: Extract the Phase 1 script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.cross_target_glue_shell" \
  playbooks/examples_library/cross_target_glue_examples.md \
  > tmp/run_glue_shell.sh
```

**Step 3: Make it executable and run**
```bash
chmod +x tmp/run_glue_shell.sh
bash tmp/run_glue_shell.sh
```

**Expected Output**:
```
=== Cross-Target Glue Demo: Shell Integration ===
Created sample data: tmp/glue_demo/sales_data.tsv

Running Prolog to generate glue scripts...
Generated: tmp/glue_demo/filter.awk
Generated: tmp/glue_demo/aggregate.py
Generated: tmp/glue_demo/format.sh
Generated: tmp/glue_demo/run_pipeline.sh

All scripts generated successfully!

=== Generated Scripts ===
...
Success: Cross-target glue shell integration demo complete
```

### Phase 2: Go Binary Integration (Optional)

Requires: Go installed (`go version` works)

**Step 1: Extract Phase 2 script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.cross_target_glue_go" \
  playbooks/examples_library/cross_target_glue_examples.md \
  > tmp/run_glue_go.sh
```

**Step 2: Run**
```bash
chmod +x tmp/run_glue_go.sh
bash tmp/run_glue_go.sh
```

**Expected Output** (if Go is installed):
```
=== Cross-Target Glue Demo: Go Binary Integration ===
go version go1.21.x ...
Created sample data: tmp/glue_demo/records.jsonl

Building Go processor...
Built: tmp/glue_demo/go_grader

Running Prolog to generate Go pipeline...
Generated: tmp/glue_demo/run_go_pipeline.sh

=== Running Go Pipeline ===
{"id":1,"name":"Alice","score":85,"grade":"B"}
{"id":2,"name":"Bob","score":92,"grade":"A"}
...
Success: Cross-target glue Go integration demo complete
```

### Phase 3: .NET Integration (Optional)

Requires: .NET SDK (`dotnet --version` works)

**Important for Termux proot-distro**: Set memory limit first:
```bash
export DOTNET_GCHeapHardLimit=1C0000000
```

**Step 1: Extract Phase 3 script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.cross_target_glue_dotnet" \
  playbooks/examples_library/cross_target_glue_examples.md \
  > tmp/run_glue_dotnet.sh
```

**Step 2: Run**
```bash
chmod +x tmp/run_glue_dotnet.sh
bash tmp/run_glue_dotnet.sh
```

### Phase 4: Rust Binary Integration (Optional)

Requires: Rust installed (`rustc --version` works)

**Step 1: Extract Phase 4 script**
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.cross_target_glue_rust" \
  playbooks/examples_library/cross_target_glue_examples.md \
  > tmp/run_glue_rust.sh
```

**Step 2: Run**
```bash
chmod +x tmp/run_glue_rust.sh
bash tmp/run_glue_rust.sh
```

**Expected Output** (if Rust is installed):
```
=== Cross-Target Glue Demo: Rust Binary Integration ===
rustc 1.xx.x ...
Created sample data: tmp/glue_demo/metrics.tsv

Building Rust processor...
Built: tmp/glue_demo/rust_aggregator

=== Running Rust Pipeline ===
metric	count	min	max	avg
cpu_usage	3	38.90	52.10	45.40
memory_mb	3	2048.00	2150.00	2099.33

Success: Cross-target glue Rust integration demo complete
```

## What This Playbook Demonstrates

1. **shell_glue module** (`src/unifyweaver/glue/shell_glue.pl`):
   - `generate_awk_script/4` - AWK script with input parsing
   - `generate_python_script/4` - Python script with format handling
   - `generate_bash_script/4` - Bash script with pipe support
   - `generate_pipeline/3` - Multi-step pipeline orchestrator

2. **native_glue module** (`src/unifyweaver/glue/native_glue.pl`):
   - `register_binary/4` - Register compiled Go/Rust binaries
   - `compiled_binary/3` - Query registered binaries
   - `generate_go_pipe_main/3` - Go pipe wrapper generation
   - `generate_rust_pipe_main/3` - Rust pipe wrapper generation

3. **Pipeline patterns**:
   - TSV/CSV/JSON format conversion
   - Streaming data through multiple processors
   - Mixing interpreted (AWK/Python) and compiled (Go/Rust) targets

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** check if Go/Rust/dotnet are installed before running optional phases
- **DO** set `DOTNET_GCHeapHardLimit` in Termux proot-distro environments

## Expected Outcome
- Phase 1 always succeeds (only needs AWK/Python/Bash)
- Phases 2-4 succeed if respective runtimes are installed
- Generated scripts demonstrate cross-target data flow

## Citations
[1] playbooks/examples_library/cross_target_glue_examples.md
[2] src/unifyweaver/glue/shell_glue.pl
[3] src/unifyweaver/glue/native_glue.pl
[4] skills/skill_unifyweaver_environment.md
