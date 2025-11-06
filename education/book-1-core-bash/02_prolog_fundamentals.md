<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 2: Prolog Fundamentals for UnifyWeaver

To effectively use UnifyWeaver, you don't need to be a Prolog guru, but you do need a solid grasp of its fundamental concepts. This chapter introduces the core building blocks of the Prolog language, with a focus on how they are used in the context of UnifyWeaver.

## The Building Blocks: Terms

In Prolog, everything is built from **terms**. There are a few basic types of terms.

### 1. Atoms
An **atom** is a general-purpose name for an object or a relationship. It is a sequence of characters that starts with a lowercase letter (unless it is enclosed in single quotes). Atoms are constants, like symbols in other languages.

**Examples:**
*   `alice`
*   `bob`
*   `parent`
*   `'Starts with a capital'`
*   `'an atom with spaces'`

### 2. Numbers
Prolog supports integers and floating-point numbers, though for UnifyWeaver, you will primarily deal with integers if at all. The tool's focus is on relationships, not heavy computation.

**Examples:**
*   `42`
*   `-100`
*   `3.14`

### 3. Variables
A **variable** is a placeholder for a term that is not yet known. In Prolog, variables begin with an uppercase letter or an underscore `_`.

**Examples:**
*   `X`
*   `Parent`
*   `FileName`
*   `_` (The anonymous variable, used when you don't care about the value)

### 4. Complex Terms (Structures)
A **complex term**, or **structure**, is the primary way data is structured in Prolog. It consists of a **functor** (which looks like an atom) and a number of **arguments** enclosed in parentheses.

**Format:** `functor(argument1, argument2, ...)`

The number of arguments a structure has is called its **arity**. The combination of a functor and its arity is often written as `functor/arity`.

**Examples:**
*   `parent(alice, bob)` - A structure with functor `parent` and arity 2.
*   `file('main.c', 1024, '2025-10-05')` - A structure with functor `file` and arity 3.
*   `edge(node1, node2)` - `edge/2`

## Facts: Stating What Is True

A **fact** is a complex term that is asserted to be true. Facts are the foundation of a Prolog program; they are the raw data the program works with.

In a UnifyWeaver context, facts are often used to represent data from your project, such as dependencies, file metadata, or entity relationships.

**Example: Defining a file dependency graph**
```prolog
% file_dependency(dependent_file, dependency).
file_dependency('main.o', 'main.c').
file_dependency('main.o', 'utils.h').
file_dependency('utils.o', 'utils.c').
file_dependency('utils.o', 'utils.h').
```
These facts state that `main.o` depends on `main.c` and `utils.h`, and so on.

## Rules: Defining New Knowledge

A **rule** is a way to define new knowledge based on existing facts and other rules. A rule has two parts: a **head** and a **body**, separated by the `:-` operator, which can be read as "if".

**Format:** `head :- body.`

The **head** is a complex term that defines the new relationship. The **body** is a comma-separated list of one or more goals that must be true for the head to be true.

**Example: Defining a transitive dependency**
Let's define a rule to find out if a file needs to be recompiled. A file needs to be recompiled if any of its dependencies have been updated. Let's expand on this. A file `F` has a transitive dependency on `D` if `F` depends on `D` directly, or if `F` depends on some other file `I` which itself has a transitive dependency on `D`.

```prolog
% A file F has a transitive dependency on D if...

% Base Case: F depends directly on D.
transitive_dependency(F, D) :- file_dependency(F, D).

% Recursive Step: F depends on I, and I has a transitive dependency on D.
transitive_dependency(F, D) :- 
    file_dependency(F, I), 
    transitive_dependency(I, D).
```
This is a classic transitive closure, a pattern that UnifyWeaver is specifically designed to optimize.

## Queries: Asking Questions

Once you have facts and rules, you can ask **queries** to retrieve information. A query is a goal (or a list of goals) that you ask the Prolog system to solve. In SWI-Prolog, queries are entered at the `?-` prompt.

Prolog answers a query by trying to find a way to make the query true, based on its database of facts and rules. If it succeeds, it will show you the variable bindings that make it true.

**Example Queries:**

1.  **What files does `main.o` depend on directly?**
    ```prolog
    ?- file_dependency('main.o', Dependency).
    ```
    Prolog would find all matching facts and give you the results:
    ```
    Dependency = 'main.c' ;
    Dependency = 'utils.h' .
    ```
    (The semicolon `;` is typed by the user to ask for more results.)

2.  **Does `main.o` have a transitive dependency on `utils.c`?**
    ```prolog
    ?- transitive_dependency('main.o', 'utils.c').
    ```
    Prolog would use the `transitive_dependency/2` rule and reply:
    ```
    true.
    ```

## Unification: The Engine of Prolog

The core mechanism that makes Prolog work is **unification**. Unification is a process of matching two terms. If they can be matched, any variables involved are **instantiated** (given a value) to make the terms identical.

*   An atom only unifies with itself: `alice` unifies with `alice`.
*   A variable can unify with any term. `X` unifies with `alice`, and `X` is now instantiated to `alice`.
*   A complex term unifies with another complex term if they have the same functor and arity, and if all their corresponding arguments can be unified.

**Example:**
When you make the query `?- file_dependency('main.o', Dependency).`, Prolog tries to unify it with the facts in its database.
1.  It tries to unify `file_dependency('main.o', Dependency)` with `file_dependency('main.o', 'main.c')`.
    *   The functors (`file_dependency`) match.
    *   The arities (2) match.
    *   The first arguments (`'main.o'` and `'main.o'`) match.
    *   The second arguments (`Dependency` and `'main.c'`) can be unified by instantiating `Dependency` to `'main.c'`.
    *   Success! Prolog reports this as a solution.

Unification is the process that drives the search for answers in a Prolog program.

## Next Steps

This chapter has covered the absolute essentials: facts, rules, queries, and the terms they are built from. With this foundation, you are ready to understand the architecture of UnifyWeaver and how it translates these Prolog constructs into executable Bash scripts.

In the next chapter, we will look at the high-level architecture of the UnifyWeaver compiler itself.

---

## Navigation

**←** [Previous: Chapter 1: Introduction](01_introduction.md) | [📖 Back to Book 1](README.md) | [Next: Chapter 3: UnifyWeaver Architecture →](03_unifyweaver_architecture.md)

