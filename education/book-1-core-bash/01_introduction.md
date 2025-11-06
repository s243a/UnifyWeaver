<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 1: Introduction to UnifyWeaver

Welcome to the study of UnifyWeaver! This document series will guide you through the theory, architecture, and practical application of this unique and powerful tool.

## What is UnifyWeaver?

At its heart, **UnifyWeaver is a Prolog-to-Bash compiler.**

That statement, while accurate, requires some unpacking. UnifyWeaver uses the logic programming language Prolog as a high-level, declarative language to define data relationships, rules, and queries. It then compiles these definitions into highly efficient, stream-based Bash scripts, where possible and otherwise uses specialized recursion patterns.

Think of it as a bridge between two very different worlds:
*   **The Declarative World of Prolog:** Where you describe *what* you want to achieve and the logical rules that govern your data.
*   **The Imperative World of Bash:** Where you write scripts to execute a sequence of commands to get things done in a Unix-like environment.

UnifyWeaver's specialty is compiling complex data relationships, especially recursive ones (like finding all ancestors in a family tree or all reachable nodes in a network), into Bash code that is both correct and performant.

## The Problem It Solves

In any large software project, complexity is a major challenge. This complexity often manifests as:
*   **Repetitive boilerplate:** Writing the same script logic in slightly different ways across a project.
*   **Inconsistent data handling:** Different scripts processing the same data in subtly different ways, leading to bugs.
*   **Hidden dependencies:** It's hard to see how different pieces of data and logic relate to each other.
*   **Maintenance nightmares:** A change in one place requires hunting down and changing dozens of other scripts.

UnifyWeaver addresses this by providing a **single source of truth**. The Prolog code becomes the master blueprint for your data logic. The Bash scripts are the generated, disposable artifacts. If you need to change the logic, you change it in one place—the Prolog source—and regenerate the scripts.

## Core Concepts

To understand UnifyWeaver, we must first grasp a few key ideas.

### Declarative Programming
Instead of writing step-by-step instructions (imperative programming), you describe the desired result and the rules that apply.

*   **Imperative:** "First, get a list of employees. Then, loop through the list. For each employee, check if their salary is over $50,000. If it is, add their name to a new list. Finally, return the new list."
*   **Declarative:** "Give me the names of all employees whose salary is over $50,000."

Prolog is a declarative language, which makes it excellent for defining rules and relationships without getting bogged down in implementation details.

### Logic Programming & Prolog
Prolog is the most well-known logic programming language. It is based on a few simple but powerful ideas:
*   **Facts:** Basic assertions about the world. `parent(alice, bob).`
*   **Rules:** Definitions of new relationships based on existing ones. `ancestor(X, Y) :- parent(X, Y).`
*   **Queries:** Questions asked to the system. `?- ancestor(alice, X).`

The Prolog engine uses these facts and rules to automatically deduce the answers to your queries.

### Code Generation
Code generation is the process of using a program to create other programs. UnifyWeaver is a code generator that takes Prolog as input and produces Bash as output. This is the mechanism it uses to translate the declarative logic into executable scripts.

## An Analogy: The Smart Loom

The name "UnifyWeaver" is a good metaphor for what it does. Imagine your project is a complex tapestry. You have different threads: data files, configuration, business logic, and infrastructure code.

UnifyWeaver acts as a **smart loom**.
*   The **Prolog code** is the pattern you design for the tapestry. It defines how all the threads should interconnect.
*   The **compiler** is the loom itself, automatically weaving the threads together according to your pattern.
*   The resulting **Bash scripts** are the finished sections of the tapestry, perfectly woven and ready to be used.

If you want to change the design, you don't re-weave it by hand. You adjust the pattern (the Prolog code), and the loom re-weaves it for you, ensuring the entire tapestry remains consistent.

## What to Expect in this Course

Over the next chapters, we will explore:
1.  **Prolog Fundamentals:** The core concepts of Prolog needed to use UnifyWeaver effectively.
2.  **UnifyWeaver Architecture:** How the compiler is structured, from the template system to the recursion analyzer.
3.  **Writing Your First Predicates:** Hands-on examples of defining facts and rules.
4.  **Compiling and Running:** How to use the compiler and execute the generated Bash scripts.
5.  **Advanced Topics:** A deep dive into how UnifyWeaver handles complex recursion, constraints, and optimizations.

Let's begin our journey by diving into the fundamentals of Prolog.

---

## Navigation

[📖 Back to Book 1](README.md) | [Next: Chapter 2: Prolog Fundamentals →](02_prolog_fundamentals.md)

