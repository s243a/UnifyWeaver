# Mutual Recursion Playbook — Reviewer Reference

## Overview
Reviewer's guide for [playbooks/mutual_recursion_playbook.md](../../../../playbooks/mutual_recursion_playbook.md).

## Agent Execution Example
```
Pretend you have fresh context and run the playbook at playbooks/mutual_recursion_playbook.md
```

## Purpose
Validates UnifyWeaver's ability to compile mutually recursive predicates (is_even/1, is_odd/1) to bash.

## Inputs & Artifacts
- Playbook: `playbooks/mutual_recursion_playbook.md`
- Example: `playbooks/examples_library/recursion_examples.md`
- Generated Prolog: `tmp/even_odd.pl`
- Generated bash: `tmp/is_even.sh`, `tmp/is_odd.sh`

## Prerequisites
1. SWI-Prolog installed
2. Perl for extraction
3. init.pl in project root
4. Run from repository root

## Execution Steps

```bash
cd /path/to/UnifyWeaver
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.mutual_recursion" \
  playbooks/examples_library/recursion_examples.md \
  > tmp/run_mutual_recursion.sh
chmod +x tmp/run_mutual_recursion.sh
bash tmp/run_mutual_recursion.sh
```

## Verification

**Expected output:**
```
Generating mutually recursive Prolog code...
✓ Generated Prolog code: tmp/even_odd.pl

Compiling mutually recursive predicates...
[UnifyWeaver] Environment initialized
  ...
=== Analyzing is_even/1 ===
=== Advanced Recursive Compilation: is_even/1 ===
  Detected mutually recursive predicate group: [is_even/1,is_odd/1]
  ...
Generated scripts: [tmp/is_even.sh,tmp/is_odd.sh]

Testing generated scripts...
Testing is_even(4):
  ✓ 4 is even
Testing is_odd(3):
  ✓ 3 is odd

Success: Mutual recursion compiled and executed
```

**Success criteria:**
- Both is_even.sh and is_odd.sh created
- is_even(4) succeeds
- is_odd(3) succeeds
- Compiler detects mutual recursion
- Exit code 0

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Only one script generated | Mutual recursion not detected | Check Prolog file has both predicates |
| Wrong results | Logic error | Verify generated bash scripts |
| Module load errors | Wrong directory | Run from project root |

## Related Material
- Playbook: [playbooks/mutual_recursion_playbook.md](../../../../playbooks/mutual_recursion_playbook.md)
- Examples: `playbooks/examples_library/recursion_examples.md`
- Compiler: `src/unifyweaver/core/recursive_compiler.pl`
