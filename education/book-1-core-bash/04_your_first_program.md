<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 4: Your First UnifyWeaver Program

This chapter is a hands-on tutorial that will guide you through the process of defining, compiling, and running a simple program with UnifyWeaver. We will use the concepts from the previous chapters to create a practical example.

We will be working with a simple family tree, defining parents and then creating a rule to find any ancestor of a person. We have already created the Prolog file for this tutorial: `education/family_tree.pl`.

## Step 1: Review the Prolog Code

First, let's examine the contents of `education/family_tree.pl`. It contains a set of facts and rules.

### The Facts
It starts by defining a series of `parent/2` facts. These represent the direct parent-child relationships.

```prolog
% parent(Parent, Child).
parent(abraham, ishmael).
parent(abraham, isaac).
parent(sarah, isaac).
parent(isaac, esau).
parent(isaac, jacob).
% ... and so on
```

### The Rules
Next, it defines two rules:

1.  `grandparent/2`: A simple, non-recursive rule that defines a grandparent as a parent of a parent.

    ```prolog
    grandparent(GP, GC) :-
        parent(GP, P),
        parent(P, GC).
    ```

2.  `ancestor/2`: A recursive rule that defines an ancestor using the classic transitive closure pattern. This is the kind of rule UnifyWeaver is built to optimize.

    ```prolog
    % Base case
    ancestor(A, D) :- parent(A, D).
    % Recursive step
    ancestor(A, D) :- 
        parent(A, P),
        ancestor(P, D).
    ```

## Step 2: Start SWI-Prolog

Now, open a terminal in the root directory of the UnifyWeaver project and start the SWI-Prolog interactive console. You can usually do this by typing `swipl`.

```bash
# Make sure swipl prolog is installed, and in your execution path.
#
# Open a terminal (e.g. cmd (on windows)), in the parent directory of this project.
#
# In the parent directory of this project type (and press enter):
$ swipl
Welcome to SWI-Prolog ...
...
?- 
```
The `?-` is the Prolog prompt, waiting for your query.

## Step 2.5: Initialize the Environment

Before we can use the UnifyWeaver compiler, we need to initialize the environment. We have provided an `init.pl` file in the `education` directory that sets up the necessary library paths.

At the Prolog prompt, enter the following query:
```prolog
?- ['education/init'].
```
You will see a message confirming that the environment has been initialized.

## Step 3: Load the UnifyWeaver Compiler

Now that the environment is initialized, we can load the UnifyWeaver compiler.
```prolog
?- use_module(unifyweaver(core/recursive_compiler)).
true.
```

## Step 4: Load Your Program

Next, load the `family_tree.pl` file into the Prolog environment. This will make its facts and rules available to the compiler.

```prolog
?- ['education/family_tree'].
true.
```
(Note: You can omit the `.pl` extension.)

## Step 4.5: Compile the Facts

Before we compile our `ancestor/2` rule, we need to compile the `parent/2` facts it relies on. UnifyWeaver is smart enough to know that `ancestor/2` depends on `parent/2`, but it's good practice to compile the base facts into their own script. This allows us to reuse the `parent` data for other rules as well.

We will use the `stream_compiler` for this, as `parent/2` is a simple set of facts (non-recursive).

1.  **Load the stream compiler:**
    ```prolog
    ?- use_module(unifyweaver(core/stream_compiler)).
    true.
    ```

2.  **Compile the facts:**
    ```prolog
    ?- stream_compiler:compile_facts(parent, 2, [], BashCode).
    ```
    This will generate the Bash code for the `parent/2` facts and store it in the `BashCode` variable.

## Step 5: Compile the `ancestor/2` Rule

Now for the main event. We will ask UnifyWeaver to compile our `ancestor/2` rule. We will use the `compile_recursive/3` predicate.

```prolog
?- compile_recursive(ancestor/2, [], BashCode).
```

## Step 6: Save the Generated Code

Now we will save both the `parent` and `ancestor` scripts to the `education/output` directory.

First, let's create the output directory if it doesn't exist:

```prolog
?- make_directory_path('education/output').
true.
```

Now save the scripts:

1.  **Save the `parent` script:**
    ```prolog
    ?- stream_compiler:compile_facts(parent, 2, [], BashCode),
       open('education/output/parent.sh', write, Stream),
       write(Stream, BashCode),
       close(Stream).
    true.
    ```

2.  **Save the `ancestor` script:**
    ```prolog
    ?- compile_recursive(ancestor/2, [], BashCode),
       open('education/output/ancestor.sh', write, Stream),
       write(Stream, BashCode),
       close(Stream).
    true.
    ```
You should now have two new files in `education/output/`: `parent.sh` and `ancestor.sh`.

## Step 7: Run the Bash Script

Finally, let's exit Prolog (`halt.`) and run our new scripts.

The `ancestor.sh` script depends on the `parent.sh` script, so we need to source them both.

```bash
$ source education/output/parent.sh
$ source education/output/ancestor.sh

# Find all known descendants of abraham
$ ancestor abraham
abraham:ishmael
abraham:isaac
abraham:esau
abraham:jacob
abraham:reuben
...

# Check if isaac is an ancestor of judah
$ ancestor isaac judah && echo "Yes" || echo "No"
Yes

# Check if sarah is an ancestor of esau
$ ancestor sarah esau && echo "Yes" || echo "No"
Yes

# Check if ishmael is an ancestor of jacob
$ ancestor ishmael jacob && echo "Yes" || echo "No"
No
```

## Step 8: Declarative Test Generation

In the previous steps, we manually tested our generated scripts. However, UnifyWeaver provides a more advanced and declarative way to generate a test runner script automatically.

This is done using an "inference-based" approach, where a Prolog script inspects the generated shell scripts and infers the appropriate test cases based on their signatures.

### 1. Prerequisites (Setup Environment)

Since we exited Prolog in Step 7, we need to set up our environment again. As before:

```bash
# Start SWI-Prolog
$ swipl
```

```prolog
% Initialize the environment
?- ['education/init'].
[UnifyWeaver] Educational environment initialized.
...
true.

% Load the recursive compiler
?- use_module(unifyweaver(core/recursive_compiler)).
true.

% Load the stream compiler
?- use_module(unifyweaver(core/stream_compiler)).
true.

% Load the family tree definitions
?- ['education/family_tree'].
true.
```

Now we're ready to proceed with the test generation.

### 2. Compile to the Output Directory

The test inference system will generate a test runner for all scripts in a directory. Let's compile our `parent` and `ancestor` predicates to the `education/output` directory.

```prolog
?- stream_compiler:compile_facts(parent, 2, [], BashCode),
   open('education/output/advanced/parent.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).
true.

?- compile_recursive(ancestor/2, [], BashCode),
   open('education/output/advanced/ancestor.sh', write, Stream),
   write(Stream, BashCode),
   close(Stream).
true.
```

### 3. Generate the Test Runner

Now, we will use the `test_runner_inference.pl` module to generate the test runner. This module is not part of the `unifyweaver` library alias, so we need to load it directly.

```prolog
?- use_module('src/unifyweaver/core/advanced/test_runner_inference').
true.

?- generate_test_runner_inferred('education/output/advanced/test_runner.sh', [output_dir('education/output')]).
true.
```

This will:
1.  Scan the `education/output` directory for `.sh` files.
2.  Extract the function signatures from `parent.sh` and `ancestor.sh`.
3.  Infer a set of test cases for these functions.
4.  Generate a new test runner script at `education/output/test_runner.sh`.

### 4. Run the Inferred Test Runner

Finally, let's exit Prolog (`halt.`) and run the generated test runner from within the output directory.

```bash
$ cd education/output/advanced
$ bash test_runner.sh
```

You can also run it directly from the project root:

```bash
$ cd education/output/advanced && bash test_runner.sh && cd ../..
```

You will see the output of the test runner, executing the inferred test cases against your compiled scripts. This provides a powerful, declarative, and automated way to ensure your compiled logic is working correctly.

## Congratulations!

You have successfully created a Prolog program, compiled it with UnifyWeaver, and executed the resulting high-performance Bash script. You have seen the entire workflow from a declarative logic program to an imperative, optimized script.

## Next Steps

In the next chapter, we will delve deeper into the compilation process, exploring the different options and how UnifyWeaver handles non-recursive predicates with the `stream_compiler`.

---

## Navigation

**←** [Previous: Chapter 3: UnifyWeaver Architecture](03_unifyweaver_architecture.md) | [📖 Back to Book 1](README.md) | [Next: Chapter 5: Stream Compilation →](05_stream_compilation.md)

