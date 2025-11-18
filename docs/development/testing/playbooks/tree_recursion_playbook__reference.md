# Tree Recursion Playbook — Reviewer Reference

## Overview
Reviewer's guide for [playbooks/tree_recursion_playbook.md](../../../../playbooks/tree_recursion_playbook.md).

## Agent Execution Example
```
Pretend you have fresh context and run the playbook at playbooks/tree_recursion_playbook.md
```

## Purpose
Validates UnifyWeaver's ability to compile tree recursive predicates (tree_sum/2) to bash.

## Inputs & Artifacts
- Playbook: `playbooks/tree_recursion_playbook.md`
- Example: `playbooks/examples_library/recursion_examples.md`
- Generated Prolog: `tmp/tree_sum.pl`
- Generated bash: `tmp/tree_sum.sh`

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
  -q "unifyweaver.execution.tree_recursion" \
  playbooks/examples_library/recursion_examples.md \
  > tmp/run_tree_recursion.sh
chmod +x tmp/run_tree_recursion.sh
bash tmp/run_tree_recursion.sh
```

## Verification

**Expected output:**
```
Generating tree recursive Prolog code...
✓ Generated Prolog code: tmp/tree_sum.pl

Compiling tree recursive predicate...
[UnifyWeaver] Environment initialized
  ...
=== Analyzing tree_sum/2 ===
=== Advanced Recursive Compilation: tree_sum/2 ===
  Compiling tree recursion: tree_sum/2
  ...
Generated scripts: [tmp/tree_sum.sh]

Testing generated script...
Testing tree_sum([10,[5,[],[]],[15,[],[]]])
[10,[5,[],[]],[15,[],[]]]:30

Success: Tree recursion compiled and executed
```

**Success criteria:**
- tree_sum.sh created
- tree_sum([10,[5,[],[]],[15,[],[]]], 30) produces `30`
- Compiler handles tree structure
- Exit code 0

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Script not generated | Compilation failed | Check Prolog syntax |
| Wrong sum | Logic error | Verify tree structure parsing |
| Module load errors | Wrong directory | Run from project root |
| Parse errors | Tree format wrong | Check tree representation format |

## Related Material
- Playbook: [playbooks/tree_recursion_playbook.md](../../../../playbooks/tree_recursion_playbook.md)
- Examples: `playbooks/examples_library/recursion_examples.md`
- Compiler: `src/unifyweaver/core/recursive_compiler.pl`
