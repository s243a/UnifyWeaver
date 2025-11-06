<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 10: Seamless Compilation with the Compiler Driver

In the previous chapters, we have seen how to compile Prolog predicates into Bash scripts. However, the process involved manually compiling each predicate and its dependencies. While the inference-based test runner helped in automating the testing, the compilation process itself was not as seamless as it could be.

This chapter introduces the **Compiler Driver**, a new module that provides a truly recursive, "one-click" compilation experience.

## The Limitation of the Previous Approach

Previously, to compile the `ancestor/2` rule, we had to:
1.  Manually compile the `parent/2` facts into `parent.sh`.
2.  Manually compile the `ancestor/2` rule into `ancestor.sh`.

This works for simple examples, but in a large project with complex dependencies, it becomes tedious and error-prone.

## Introducing the Compiler Driver

The new `compiler_driver.pl` module solves this problem. It provides a single entry point, `compile/3`, that automatically finds and compiles all dependencies of a given predicate.

### How it Works

When you call `compile/3` on a predicate:
1.  **Dependency Analysis:** It first scans the predicate's code to find all other predicates it depends on. For `ancestor/2`, it will find the dependency on `parent/2`.
2.  **Recursive Compilation:** It then recursively calls the compiler for each dependency it finds.
3.  **Tracks Compiled Files:** It keeps track of what has already been compiled to avoid redundant work and infinite loops.
4.  **Compiles the Target:** Once all dependencies are compiled, it compiles the predicate you originally requested.
5.  **Returns a Full List:** Finally, it returns a complete list of all the shell scripts that were generated, both for the target predicate and all its dependencies.

## Hands-On: Using the Compiler Driver

Let's see how this simplifies our `ancestor/2` example.

### 1. Load the Necessary Modules

Start SWI-Prolog and load the `init.pl` file as before. Then, load the new `compiler_driver` module.

```prolog
?- ['education/init'].
true.

?- use_module(unifyweaver(core/compiler_driver)).
true.
```

### 2. Compile `ancestor/2`

Now, we can compile the `ancestor/2` predicate with a single call. We will also load our `family_tree.pl` file to make our predicates available to the compiler.

```prolog
?- ['education/family_tree'].
true.

?- compile(ancestor/2, [output_dir('education/output/recursive')], GeneratedScripts).
```

Notice the new `output_dir` option, which tells the compiler where to save the generated files. After the command finishes, the `GeneratedScripts` variable will contain a list of all the files that were created:

```
GeneratedScripts = ['education/output/recursive/parent.sh', 'education/output/recursive/ancestor.sh'].
```

As you can see, the compiler automatically found the `parent/2` dependency and compiled it as well.

### 3. A Simplified Test Runner

This new recursive compilation makes creating a test runner trivial. The `GeneratedScripts` variable gives us the exact list of files we need to `source`.

Here is a simple Prolog predicate that can generate a test runner for any compiled predicate:

```prolog
generate_test_runner(Predicate, TestRunnerPath) :-
    compile(Predicate, [output_dir('education/output/recursive')], GeneratedScripts),
    open(TestRunnerPath, write, Stream),
    write(Stream, '#!/bin/bash\n'),
    forall(member(Script, GeneratedScripts),
           format(Stream, 'source ~w\n', [Script])),
    format(Stream, '\n# --- Add your tests here ---\n'),
    format(Stream, 'ancestor abraham jacob && echo "PASS" || echo "FAIL"\n'),
    close(Stream).
```

You could run this like so:

```prolog
?- generate_test_runner(ancestor/2, 'education/output/recursive/test_ancestor.sh').
true.
```

Then, you can run the generated test script from your terminal:

```bash
$ bash education/output/recursive/test_ancestor.sh
```

## Conclusion

The compiler driver represents a significant step forward in the usability of UnifyWeaver. By automatically handling dependencies, it makes the compilation process more robust, less error-prone, and much more seamless, truly living up to the goal of one-click compilation from a declarative source.

---

## Navigation

**←** [Previous: Chapter 11: Test Runner Inference](11_test_runner_inference) | [📖 Back to Book 1](README) | [Next: Chapter 13: Partitioning and Parallel Execution →](13_partitioning_and_parallel_execution)

