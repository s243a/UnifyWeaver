# Handoff for Architectural Review

**To:** Reviewing Agent / Developer
**From:** Gemini CLI
**Date:** 2025-10-25
**Subject:** Review of Playbook Compiler Architecture & Proposed Next Steps

## 1. Context

We are in the process of building a Prolog-based compiler named `playbook_compiler.pl`. The purpose of this compiler is to transform a high-level "Playbook" (written in Markdown) into an executable bash script. This compiled script will enable an AI agent to make strategic, resource-aware decisions at runtime.

This work is based on the "Economic Agent" philosophy, detailed in `proposals/llm_workflows/philosophy.md`.

## 2. What We Just Did: Implemented a Template-Driven Architecture

In our latest session, we established the code generation architecture for the compiler. The key decisions and actions were:

1.  **Adopted a Template-Based Approach:** Instead of having the Prolog compiler generate a monolithic bash script as a single string, we decided to use a more composable, template-driven method. This aligns with the existing design patterns in the UnifyWeaver project.

2.  **Created Composable Templates:** We created a new directory, `templates/playbook/`, and populated it with several small, single-purpose bash templates:
    *   `playbook_header.tmpl.sh`: The header for the compiled script.
    *   `tool_function.tmpl.sh`: A wrapper to turn any tool script into a callable bash function.
    *   `strategy_decision_tree.tmpl.sh`: The core logic for the `case` statement that will allow the agent to select a strategy.

3.  **Updated the Prolog Compiler:** We modified the `generate_decision_script/2` predicate within `src/unifyweaver/compilers/playbook_compiler.pl`. The new implementation now uses the `render_named_template/3` predicate to assemble the final script from the various template files. This provides a clean separation between the compiler's logic and the generated code's structure.

**Why we did this:** This architecture is more maintainable, scalable, and reusable. It allows us to change the structure of the output script by editing simple template files, without having to modify the more complex Prolog compiler logic.

## 3. Proposed Next Step: Implement the Playbook Parser

**What I propose we do next:** Focus on implementing the `parse_playbook/2` predicate in `playbook_compiler.pl`.

**Why this is the logical next step:**

Currently, our compiler's parsing and profiling logic are just placeholders. The code generation step, while structurally sound, is being fed static, hard-coded data. To make the compiler functional, it needs to be able to ingest real data from a playbook file.

The `parse_playbook/2` predicate is the gateway for this data. It is responsible for reading a file like `context_gatherer_playbook.md` and transforming it into a structured Prolog term that the rest of the compiler can use.

My specific suggestion is to start with a focused and manageable part of this task: **write a predicate that can read the playbook file and reliably extract the content of the Prolog code blocks (` ```prolog ... ``` `)**.

This is the most critical data in the playbook, as it contains the formal, declarative rules for strategy selection and error handling. Once we can extract this logic, we can then use Prolog's powerful parsing capabilities (like DCGs) to turn that raw text into structured facts and rules that the agent can reason about.

## 4. Request for Review

I am handing this off for a second opinion on the proposed next step.

**The key question is:** Is focusing on implementing the `parse_playbook/2` predicate, starting with the extraction of the Prolog code blocks, the correct and most logical next step in the development of our compiler? Or is there a different architectural approach or a higher-priority task we should consider first?
