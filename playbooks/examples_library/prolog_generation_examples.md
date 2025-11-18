---
file_type: UnifyWeaver Example Library
---
# Playbook Examples: Prolog Code Generation

This file contains executable records for UnifyWeaver playbooks demonstrating Prolog code generation and compilation.

## Record: unifyweaver.execution.generate_factorial

This record demonstrates generating Prolog code for factorial calculation and compiling it to bash.

> [!example-record]
> id: unifyweaver.execution.generate_factorial
> name: unifyweaver.execution.generate_factorial
> description: Generates factorial Prolog code and compiles it using recursive compiler

```bash
#!/bin/bash
#
# Prolog Generation Example - Demonstrates generating and compiling Prolog code
# This script is designed to be extracted and run directly.

# --- Config ---
TMP_FOLDER="${TMP_FOLDER:-tmp}"
PROLOG_FILE="$TMP_FOLDER/factorial.pl"
OUTPUT_DIR="$TMP_FOLDER"

# --- Step 1: Generate Prolog Code ---
echo "Generating Prolog code for factorial..."
mkdir -p "$TMP_FOLDER"
cat > "$PROLOG_FILE" <<'PROLOG_EOF'
% factorial(N, F) - Calculates factorial of N
% Base case: 0! = 1
factorial(0, 1).

% Recursive case: N! = N * (N-1)!
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
PROLOG_EOF
echo "✓ Generated Prolog code: $PROLOG_FILE"

# --- Step 2: Compile Prolog to Bash ---
echo ""
echo "Compiling Prolog to bash using UnifyWeaver..."
swipl -f init.pl -g "
    consult('$PROLOG_FILE'),
    use_module(library(unifyweaver/core/compiler_driver)),
    compile(factorial/2, [output_dir('$OUTPUT_DIR')], Scripts),
    format('~nGenerated scripts: ~w~n', [Scripts]),
    halt"

# --- Step 3: Test Generated Script ---
echo ""
echo "Testing generated factorial script..."
if [[ -f "$OUTPUT_DIR/factorial.sh" ]]; then
    echo "Running factorial(5):"
    bash "$OUTPUT_DIR/factorial.sh" 5
    echo ""
    echo "Success: Factorial compiled and executed correctly"
else
    echo "Error: Expected script not found: $OUTPUT_DIR/factorial.sh"
    exit 1
fi
```
