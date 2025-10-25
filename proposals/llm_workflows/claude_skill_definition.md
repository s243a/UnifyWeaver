# Claude Skill: UnifyWeaver Tool Executor

## Purpose
The purpose of this skill is to execute complex, multi-step tasks by interpreting a "Natural Language Program" and using a set of compiled, local tools. This allows Claude to perform actions that go beyond its built-in capabilities, such as file manipulation, running external processes, and interacting with a local development environment.

## Input
This skill takes one primary input: the path to a "compiled tool script" (e.g., `compiled_tool.sh`).

This script is a self-contained bash script that includes two key parts:
1.  A **"Natural Language Program"**: Embedded in a comment block, this program is written in a specialized Markdown format. It describes the overall goal, the plan of action, API documentation for the tools, and a formal logic representation.
2.  A set of **"Compiled Tools"**: The actual tool code, included as callable bash functions.

## How to Use
1.  You will be given a path to a `compiled_tool.sh` file.
2.  Your first step is to **read the "Natural Language Program" section** of this file to understand the goal and the plan.
3.  The program is written in a special format. You can infer the structure from the Markdown headers and the labeled code blocks (e.g., `todo`, `pseudocode_prolog`, `bash`). The full language specification is defined in the UnifyWeaver project, but you should have enough information from the program itself.
4.  After understanding the plan, you should **execute it by calling the bash functions** that are available in the same script.
5.  You can and should interact with the user to ask for clarification or provide updates on your progress.

## Example Invocation
"Claude, please execute the task defined in `/path/to/compiled_tool.sh`."

## Language Specification
The language specification for the "Natural Language Program" is currently under development within the UnifyWeaver project. The core principles are:
-   Markdown for overall structure.
-   YAML frontmatter for metadata.
-   Labeled code blocks for different representations of the plan.
-   A focus on instructing the LLM, not providing full code context.
