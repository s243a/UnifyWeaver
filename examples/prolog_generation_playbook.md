# Playbook: Generate and Transpile Factorial Prolog

## Goal

Generate a Prolog program for calculating the factorial of a number, transpile it using UnifyWeaver's `compiler_driver.pl`, and verify its execution.

## Context

This playbook demonstrates the agent's ability to generate declarative Prolog code based on a natural language specification, and then integrate with UnifyWeaver's transpilation capabilities.

## Strategy

1.  **Generate Prolog:** Based on the goal, generate a Prolog program for the factorial function.
2.  **Save Prolog:** Save the generated Prolog to a temporary file.
3.  **Transpile Prolog:** Use `compiler_driver.pl` to transpile the Prolog file into a Bash script.
4.  **Execute and Verify:** Run the generated Bash script with a test input and verify the output.

## Tools

*   **UnifyWeaver Compiler Driver (Prolog Module):** `src/unifyweaver/core/compiler_driver.pl`
    *   **Purpose:** A Prolog module that transpiles Prolog predicates and their dependencies into various target languages (e.g., Bash).
    *   **Usage:** Can be queried from a SWI-Prolog interpreter, e.g., `compile(Predicate/Arity, Options, GeneratedScripts)`.

*   **Perl Example Extraction Tool:** (Mentioned in Workflow)
    *   **Purpose:** A Perl tool used to extract specific examples or sections from example libraries to teach the agent how to generate Prolog.

## Expected Output (Prolog Generation)

> [!output]
> language: prolog
> purpose: Generated Prolog code for the factorial function.
> format: declarative_logic
>
> ```prolog
> % factorial(N, F) :- F is N!
> factorial(0, 1).
> factorial(N, F) :-
>     N > 0,
>     N1 is N - 1,
>     factorial(N1, F1),
>     F is N * F1.
> ```

## Execution Steps (for Agent)

1.  **Generate the Prolog code** as shown in the "Expected Output (Prolog Generation)" section.
2.  **Save this Prolog code** to a temporary file, e.g., `/tmp/factorial.pl`.
3.  **Transpile the Prolog to Bash using SWI-Prolog:**
    ```bash
    swipl -l /data/data/com.termux/files/home/UnifyWeaver/tmp/factorial.pl -s src/unifyweaver/core/compiler_driver.pl -g "compile(factorial/2, [], GeneratedScripts), writeln(GeneratedScripts), halt."
    ```
    *   This command loads the `compiler_driver.pl` module.
    *   It then calls the `compile/3` predicate with `factorial/2` as the predicate to compile, an empty list for options, and `GeneratedScripts` to capture the output file paths.
    *   The `writeln(GeneratedScripts)` will print the paths of the generated scripts.
    *   `halt.` exits SWI-Prolog.
    *   The generated script will likely be in `education/output/advanced/factorial.sh` by default.
    *   **Note on Compiler Limitation:** Currently, the `compiler_driver.pl` generates Bash scripts that define functions but lack a top-level execution block to process command-line arguments and print results. This requires a temporary manual modification for direct execution.
4.  **Manually add main execution block to the Bash script (Temporary Workaround):**
    *   After transpilation, open `education/output/advanced/factorial.sh` and append the following lines to the end of the file:
    ```bash
    # Main execution block for direct script invocation
    if [[ -n "$1" ]]; then
        input_n="$1"
        # Assuming factorial function returns "N:Result"
        result_line=$(factorial "$input_n")
        echo "$result_line" | cut -d':' -f2 # Extract just the result part
    else
        echo "Usage: $0 <number>"
        exit 1
    fi
    ```
    *   This step is a temporary workaround. The `compiler_driver.pl` needs to be improved to generate this block automatically (see `docs/development/ai-skills/compiler_driver_improvements.md`).
5.  **Make the Bash script executable:**
    ```bash
    chmod +x education/output/advanced/factorial.sh
    ```
6.  **Test the Bash script:**
    ```bash
    education/output/advanced/factorial.sh 5
    ```
    **Expected Result:** The script should output `120`. The agent should verify this output.

## Verification

The agent should compare the actual output of `/tmp/factorial.sh 5` with the expected result (120). If they match, the generation and transpilation were successful.
