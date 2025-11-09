# Workflow TODOs for LLM Agent Development

This document outlines potential next steps for enhancing the LLM agent's capabilities within the UnifyWeaver ecosystem, building upon the established workflow philosophy, playbook format, and improved testing infrastructure.

## 1. Refine `test_runner_inference.pl` for Helper Functions
*   **Description:** While primary function tests are now clean, helper functions still receive generic tests that often fail, leading to noise in the test output.
*   **Goal:** Improve `test_runner_inference.pl` to either generate more intelligent/specific tests for helper functions or to explicitly mark/skip generic tests for them, making the test output more reliable.
*   **Approach:**
    *   Analyze the structure and naming conventions of generated helper functions.
    *   Develop a strategy to distinguish between primary and helper functions more robustly.
    *   Implement changes in `test_runner_inference.pl` to apply this strategy (e.g., by generating minimal sanity checks for helpers, or by completely omitting tests for functions not intended for direct external verification).

## 2. Implement a New Playbook for a Different Recursion Pattern
*   **Description:** We have successfully demonstrated linear recursion with factorial. UnifyWeaver is designed to handle various recursion patterns.
*   **Goal:** Further validate the system's capabilities (Prolog generation, transpilation, improved testing) and uncover new challenges by implementing a playbook for a different recursion pattern.
*   **Approach:**
    *   Choose a new recursion pattern (e.g., mutual recursion, tail recursion, tree recursion, fold/unfold patterns).
    *   Generate a Prolog example for this pattern.
    *   Create a new playbook (similar to `prolog_generation_playbook.md`) that guides the agent through generating, transpiling, and testing this new pattern.

## 3. Enhance the `compiler_driver.pl` with a "Main" Execution Block
*   **Description:** The `compiler_driver.pl` currently generates Bash scripts that define functions but lack a top-level execution block, making them non-executable as standalone utilities without manual sourcing or a test runner.
*   **Goal:** Modify `compiler_driver.pl` to include a default main execution block in the generated Bash scripts, allowing them to be run directly from the command line with arguments.
*   **Approach:**
    *   Analyze the requirements for a generic Bash main block that can parse arguments and call the primary compiled function.
    *   Implement the logic within `compiler_driver.pl` to inject this main block into the generated scripts.

## 4. Develop a Playbook for "Workflow Environment Setup"
*   **Description:** While `workflow_environment.md` exists, an executable playbook for setting up an agent's environment would streamline onboarding and ensure consistency.
*   **Goal:** Create a self-contained playbook that an agent can execute to verify and set up its operational environment for UnifyWeaver development.
*   **Approach:**
    *   Define the necessary environment checks (e.g., SWI-Prolog installation, required Bash utilities, environment variables).
    *   Outline steps for an agent to perform these checks and potentially fix issues.
    *   Structure the playbook for automated execution and verification.

## 5. Integrate Example Extraction Tool
*   **Description:** The `ch1_introduction.md` mentions a Perl tool for extracting examples from the example libraries (`compilation_examples.md`, `testing_examples.md`). This tool has not yet been integrated into any workflow.
*   **Goal:** Leverage the existing example libraries more effectively by integrating the example extraction tool into a practical workflow.
*   **Approach:**
    *   Locate or develop the Perl tool for example extraction.
    *   Create a playbook step or a new playbook that demonstrates how to use this tool to programmatically extract and utilize examples from the documentation.
