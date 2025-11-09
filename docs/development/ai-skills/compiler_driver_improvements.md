# Compiler Driver Improvements for LLM Agent Executability

## Problem: Incomplete Bash Script Generation

The current `compiler_driver.pl` (and its underlying `recursive_compiler.pl` and `stream_compiler.pl` modules) generates Bash scripts that define functions corresponding to the compiled Prolog predicates. However, these generated scripts **lack a top-level execution block** that:
1.  Parses command-line arguments.
2.  Invokes the compiled predicate's corresponding Bash function.
3.  Prints the result to standard output.

**Example:** When compiling `factorial/2`, the generated `factorial.sh` defines a `factorial` Bash function but does not call it with command-line arguments.

## Impact on LLM Agents

This incompleteness significantly hinders an LLM agent's ability to seamlessly execute and test generated code. An agent following a playbook would expect a directly executable script that produces output upon invocation. The current behavior leads to:
*   **Confusion:** The agent might not understand why a seemingly correct script produces no output.
*   **Manual Intervention:** Requires the agent (or human) to manually add a main execution block for testing or direct use.
*   **Increased Complexity:** Adds an unstated, non-trivial step to the agent's execution pipeline, making playbooks harder to follow and less robust.

## Proposed Improvement: Automatic Main Execution Block Generation

The `compiler_driver.pl` should be enhanced to automatically include a default main execution block in the generated Bash scripts. This block would ensure the scripts are immediately executable and produce expected output when run from the command line.

### Recommended Features for the Main Block:

1.  **Argument Parsing:** A simple mechanism to parse command-line arguments and pass them to the compiled predicate's Bash function.
2.  **Function Invocation:** Call the primary Bash function corresponding to the compiled Prolog predicate.
3.  **Result Output:** Print the function's return value (or relevant output) to standard output.
4.  **Error Handling (Optional but Recommended):** Basic error handling for incorrect arguments or execution failures.

### Example Implementation for `factorial.sh`:

For a predicate like `factorial/2`, the generated Bash script should include a section similar to this at the end:

```bash
# Main execution block for direct script invocation
if [[ -n "$1" ]]; then
    input_n="$1"
    # Assuming factorial function returns "N:Result" or just "Result"
    # The current factorial function echoes "N:Result"
    result_line=$(factorial "$input_n")
    echo "$result_line" | cut -d':' -f2 # Extract just the result part
else
    echo "Usage: $0 <number>"
    exit 1
fi
```

### Future Considerations:

*   **Configuration Option:** Add an option to the `compile/3` predicate in `compiler_driver.pl` (e.g., `main_block(true/false)`) to control the generation of this main execution block, allowing for library-only script generation when desired.
*   **Output Format Standardization:** Standardize the output format of compiled Bash functions to make parsing results easier for the main block.
