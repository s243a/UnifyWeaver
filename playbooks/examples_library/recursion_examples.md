---
file_type: UnifyWeaver Example Library
---
# Playbook Examples: Advanced Recursion Patterns

This file contains executable records for UnifyWeaver playbooks demonstrating advanced recursion patterns (mutual recursion and tree recursion).

## Record: unifyweaver.execution.mutual_recursion

This record demonstrates mutual recursion with is_even/is_odd predicates.

> [!example-record]
> id: unifyweaver.execution.mutual_recursion
> name: unifyweaver.execution.mutual_recursion
> description: Demonstrates mutual recursion compilation with even/odd predicates

```bash
#!/bin/bash
#
# Mutual Recursion Example - Demonstrates compiling mutually recursive predicates
# This script is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
PROLOG_FILE="$TMP_FOLDER/even_odd.pl"
OUTPUT_DIR="$TMP_FOLDER"

# --- Step 1: Generate Prolog Code ---
echo "Generating mutually recursive Prolog code..."
mkdir -p "$TMP_FOLDER"
cat > "$PROLOG_FILE" <<'PROLOG_EOF'
% Mutually recursive even/odd predicates
is_even(0).
is_even(N) :-
    N > 0,
    N1 is N - 1,
    is_odd(N1).

is_odd(N) :-
    N > 0,
    N1 is N - 1,
    is_even(N1).
PROLOG_EOF
echo "✓ Generated Prolog code: $PROLOG_FILE"

# --- Step 2: Compile to Bash ---
echo ""
echo "Compiling mutually recursive predicates..."
swipl -f init.pl -g "
    consult('$PROLOG_FILE'),
    use_module(library(unifyweaver/core/compiler_driver)),
    compile(is_even/1, [output_dir('$OUTPUT_DIR')], Scripts),
    format('~nGenerated scripts: ~w~n', [Scripts]),
    halt"

# --- Step 3: Test ---
echo ""
echo "Testing generated scripts..."
if [[ -f "$OUTPUT_DIR/is_even.sh" && -f "$OUTPUT_DIR/is_odd.sh" ]]; then
    echo "Testing is_even(4):"
    bash "$OUTPUT_DIR/is_even.sh" is_even 4 && echo "  ✓ 4 is even"
    echo "Testing is_odd(3):"
    bash "$OUTPUT_DIR/is_odd.sh" is_odd 3 && echo "  ✓ 3 is odd"
    echo ""
    echo "Success: Mutual recursion compiled and executed"
else
    echo "Error: Expected scripts not found"
    exit 1
fi
```

## Record: unifyweaver.execution.tree_recursion

This record demonstrates tree recursion with tree_sum predicate.

> [!example-record]
> id: unifyweaver.execution.tree_recursion
> name: unifyweaver.execution.tree_recursion
> description: Demonstrates tree recursion compilation with tree_sum predicate

```bash
#!/bin/bash
#
# Tree Recursion Example - Demonstrates compiling tree recursive predicates
# This script is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
PROLOG_FILE="$TMP_FOLDER/tree_sum.pl"
OUTPUT_DIR="$TMP_FOLDER"

# --- Step 1: Generate Prolog Code ---
echo "Generating tree recursive Prolog code..."
mkdir -p "$TMP_FOLDER"
cat > "$PROLOG_FILE" <<'PROLOG_EOF'
% Tree recursion: sum all values in a binary tree
% Tree representation: [Value, LeftSubtree, RightSubtree]
tree_sum([], 0).
tree_sum([Value, Left, Right], Sum) :-
    tree_sum(Left, LeftSum),
    tree_sum(Right, RightSum),
    Sum is Value + LeftSum + RightSum.
PROLOG_EOF
echo "✓ Generated Prolog code: $PROLOG_FILE"

# --- Step 2: Compile to Bash ---
echo ""
echo "Compiling tree recursive predicate..."
swipl -f init.pl -g "
    consult('$PROLOG_FILE'),
    use_module(library(unifyweaver/core/compiler_driver)),
    compile(tree_sum/2, [output_dir('$OUTPUT_DIR')], Scripts),
    format('~nGenerated scripts: ~w~n', [Scripts]),
    halt"

# --- Step 3: Test ---
echo ""
echo "Testing generated script..."
if [[ -f "$OUTPUT_DIR/tree_sum.sh" ]]; then
    echo "Testing tree_sum([10,[5,[],[]],[15,[],[]]]):"
    bash "$OUTPUT_DIR/tree_sum.sh" '[10,[5,[],[]],[15,[],[]]]'
    echo ""
    echo "Success: Tree recursion compiled and executed"
else
    echo "Error: Expected script not found: $OUTPUT_DIR/tree_sum.sh"
    exit 1
fi
```
