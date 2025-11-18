# Prolog Generation Playbook — Reviewer Reference

## Overview
This document is a reviewer's guide and checklist for validating the agent-facing Prolog generation playbook:
[playbooks/prolog_generation_playbook.md](../../../../playbooks/prolog_generation_playbook.md).

- **The executable playbook designed for agents and LLMs resides in the playbooks folder.**
- This guide provides context, test conventions, validation steps, and expected behaviors when the playbook is run by an agent.

## Agent Execution Example

An AI coding agent (e.g., Gemini, Claude) can be prompted with:
```
Pretend you have fresh context and run the playbook at playbooks/prolog_generation_playbook.md
```

## Purpose

This document validates UnifyWeaver's ability to compile Prolog code to bash using the recursive compiler. The aim is to demonstrate the complete workflow: generate Prolog → compile to bash → execute and verify.

## Inputs & Artifacts
- Playbook file: `playbooks/prolog_generation_playbook.md`
- Example record: `playbooks/examples_library/prolog_generation_examples.md`
- Generated Prolog: `tmp/factorial.pl`
- Generated bash: `tmp/factorial.sh`
- Temporary directory for artifacts: `tmp/`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Perl installed for record extraction.
3. `init.pl` exists in project root.
4. Run all commands from the repository root.

## Execution Steps

1. **Navigate to Project Root**
   ```bash
   cd /path/to/UnifyWeaver
   ```

2. **Extract the Record**
   ```bash
   perl scripts/utils/extract_records.pl \
     -f content \
     -q "unifyweaver.execution.generate_factorial" \
     playbooks/examples_library/prolog_generation_examples.md \
     > tmp/run_factorial_example.sh
   ```

3. **Run the Bash Script**
   ```bash
   chmod +x tmp/run_factorial_example.sh
   bash tmp/run_factorial_example.sh
   ```

## Verification

**Expected output:**
```
Generating Prolog code for factorial...
✓ Generated Prolog code: tmp/factorial.pl

Compiling Prolog to bash using UnifyWeaver...
[UnifyWeaver] Environment initialized
  Project root: /path/to/UnifyWeaver
  Source directory: /path/to/UnifyWeaver/src
  UnifyWeaver modules: /path/to/UnifyWeaver/src/unifyweaver
--- Firewall validation passed for factorial/2. Proceeding with compilation. ---
=== Analyzing factorial/2 ===
=== Advanced Recursive Compilation: factorial/2 ===
  Compiling linear recursion: factorial/2

Generated scripts: [tmp/factorial.sh]

Testing generated factorial script...
Running factorial(5):
5:120

Success: Factorial compiled and executed correctly
```

**Success criteria:**
- Script exits with status code 0
- factorial(5) produces `5:120` (correct result)
- Prolog file created at `tmp/factorial.pl`
- Bash script created at `tmp/factorial.sh`
- Compilation uses advanced recursive compiler

## Troubleshooting

| Symptom                                   | Likely Cause              | Fix                                                                  |
| ------------------------------------------ | ------------------------- | --------------------------------------------------------------------- |
| "init.pl not found"                        | Missing init.pl           | Ensure init.pl exists in project root                                 |
| "Unknown procedure: compile/3"             | Module not loaded         | Check that compiler_driver module loads correctly                     |
| SWI-Prolog errors loading modules          | Bad paths or module names | Verify running from project root                                      |
| Artifacts missing in tmp/                  | Directory doesn't exist   | Create tmp/ directory in project root                                 |
| Wrong factorial result                     | Compilation error         | Check tmp/factorial.sh for correct bash implementation                |
| "Advanced recursive compilation" not shown | Wrong compiler used       | Verify recursive_compiler is being invoked                            |

## Related Material

- Agent-facing playbook: [playbooks/prolog_generation_playbook.md](../../../../playbooks/prolog_generation_playbook.md)
- Environment setup skill: `skills/skill_unifyweaver_environment.md`
- Example record: `playbooks/examples_library/prolog_generation_examples.md`
- Compiler driver: `src/unifyweaver/core/compiler_driver.pl`
- Recursive compiler: `src/unifyweaver/core/recursive_compiler.pl`
