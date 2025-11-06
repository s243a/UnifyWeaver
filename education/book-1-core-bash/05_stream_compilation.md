<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 5: Stream Compilation: Handling Non-Recursive Predicates

In the previous chapter, we focused on compiling a recursive predicate with the `recursive_compiler`. However, many useful predicates are not recursive. These are handled by a different component of UnifyWeaver: the `stream_compiler`.

This chapter explores how UnifyWeaver efficiently transforms simple, non-recursive rules into streaming Bash pipelines.

## The Role of the `stream_compiler`

The `stream_compiler.pl` module is responsible for compiling any predicate that does not call itself (either directly or indirectly). Its philosophy is to treat Prolog's logic as a blueprint for a data processing pipeline, where data flows from one command to the next, much like a Unix pipeline.

This approach is highly efficient in terms of memory, as it avoids loading large datasets into memory all at once. Instead, data is processed one line at a time.

## Compiling a Simple Rule

Let's consider the `grandparent/2` rule from our `family_tree.pl` example:

```prolog
grandparent(GP, GC) :-
    parent(GP, P),
    parent(P, GC).
```

Logically, this means "GP is a grandparent of GC if GP is a parent of P, and P is a parent of GC." The variable `P` acts as the bridge, linking the two `parent` relationships.

The `stream_compiler` translates this into a Bash pipeline that mirrors the logic:

1.  **Get all parents:** The first `parent(GP, P)` goal generates a stream of all known parent-child pairs.
2.  **Join with parents again:** This stream is then piped to a second stage. For each pair `GP:P` from the input stream, this stage finds all children `GC` where `P` is the parent. It then outputs the final `GP:GC` pair.

This is a classic **stream join**. Here's the **actual generated code** from UnifyWeaver:

```bash
parent_join() {
    while IFS= read -r input; do
        IFS=":" read -r a b <<< "$input"
        for key in "${!parent_data[@]}"; do
            IFS=":" read -r c d <<< "$key"
            [[ "$b" == "$c" ]] && echo "$a:$d"
        done
    done
}

grandparent() {
    parent_stream | parent_join | sort -u
}
```

**How it works:**
1. `parent_stream` outputs all parent:child pairs
2. Each line flows through `parent_join`
3. For each line (a:b), we check all parent data entries (c:d)
4. If b==c (the child of first pair matches parent of second), output a:d
5. `sort -u` removes duplicates

**Note on variable scope:** The variables `a`, `b`, `c`, `d` are local to the `while` loop. The pipe is safe here because we're not accumulating state - just transforming each line. See Chapter 7 for details on when variable scope matters.

## How to Compile a Non-Recursive Predicate

To compile a non-recursive predicate, you can use the `compile_stream/3` predicate, which works just like `compile_recursive/3`.

Let's try this from the SWI-Prolog prompt (assuming you have loaded the compiler and `family_tree.pl`):

```prolog
?- compile_stream(grandparent/2, [], BashCode),
   open('education/grandparent.sh', write, S), write(S, BashCode), close(S).
true.
```

This will create a new file, `education/grandparent.sh`, containing the Bash implementation for finding grandparents.

You can now run this script just as you did before:

```bash
$ source education/grandparent.sh

# Find all grandparents and their grandchildren
$ grandparent_all
grandparent:esau
grandparent:jacob
...

# Check if abraham is a grandparent of jacob
$ grandparent_check abraham jacob && echo "Yes" || echo "No"
Yes
```

## Handling Multiple Rules

What happens when a predicate has more than one rule? For example, let's define a `child/2` predicate.

```prolog
% A person is a child if they have a mother or a father.
child(C) :- parent(M, C), female(M).
child(C) :- parent(F, C), male(F).
```

In Prolog, these two rules represent an "OR" condition. A goal `child(X)` will succeed if *either* rule succeeds.

The `stream_compiler` handles this by:
1.  Generating a pipeline for the first rule.
2.  Generating a separate pipeline for the second rule.
3.  Concatenating the output of both pipelines.
4.  Using `sort -u` to remove any duplicate results.

The conceptual Bash code would look like this:

```bash
child_stream() {
    {
        # Code for the first rule (mother)
        generate_children_from_mothers;

        # Code for the second rule (father)
        generate_children_from_fathers;

    } | sort -u
}
```

The use of `sort -u` is a simple and robust way to implement the logical OR and ensure that each distinct result appears only once, just as it would in Prolog.

## The `compile_predicate` Wrapper

Instead of manually choosing between `compile_recursive/3` and `compile_stream/3`, you can use the main wrapper predicate `compile_predicate/3`. This predicate automatically analyzes the predicate and calls the correct underlying compiler.

```prolog
?- compile_predicate(grandparent/2, [], BashCode). % Will use stream_compiler

?- compile_predicate(ancestor/2, [], BashCode).    % Will use recursive_compiler
```

This is the recommended way to compile predicates unless you have a specific reason to invoke one of the compilers directly.

## Next Steps

We have now covered both recursive and non-recursive compilation, the two main pillars of UnifyWeaver. You should now have a good understanding of how your Prolog rules, whether simple or complex, are transformed into executable scripts.

In the next chapter, we will explore some more advanced topics, including the constraint system that lets you control aspects of the compilation, such as ordering and uniqueness.

---

## Navigation

**←** [Previous: Chapter 4: Your First Program](04_your_first_program.md) | [📖 Back to Book 1](README.md) | [Next: Chapter 6: Advanced Constraints →](06_advanced_constraints.md)

