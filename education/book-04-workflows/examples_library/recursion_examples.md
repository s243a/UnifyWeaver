# Prolog Recursion Examples

This file contains examples of common recursive predicates in Prolog, suitable for compilation with UnifyWeaver.

---

### Ancestor Relationship

> [!example-record]
> id: 20251026-150000-001
> name: prolog.recursion.ancestor

This example demonstrates a standard transitive closure, defining an ancestor as a parent or the ancestor of a parent. UnifyWeaver's `recursive_compiler` is specifically designed to optimize this pattern into an efficient breadth-first search in the target bash script.

**Facts (The Base Data):**
```prolog
parent(alice, bob).
parent(bob, charlie).
parent(charlie, diana).
```

**Rules (The Logic):**
```prolog
ancestor(X, Y) :- 
    parent(X, Y).

ancestor(X, Z) :- 
    parent(X, Y), 
    ancestor(Y, Z).
```
