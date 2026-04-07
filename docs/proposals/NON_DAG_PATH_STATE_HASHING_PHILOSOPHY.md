# Non-DAG Path-State Hashing: Philosophy

## Problem

We now have a clearer separation between two classes of recursive graph
workloads:

1. **DAG workloads**
   - well-founded by structure
   - no revisitation ambiguity
   - often admit dynamic programming or plain closure/count algorithms

2. **Non-DAG workloads**
   - cycles are present
   - simple-path semantics matter
   - per-path visited state is part of the meaning, not just an
     implementation detail

The recent dependency reach benchmark reinforced this split. The DAG case
should be optimized as a DAG. The non-DAG case should not be forced into
DAG-style scalar summaries that lose correctness.

## Core Position

For non-DAG simple-path workloads, we should treat **path identity** or
**visited-state identity** as a first-class execution concern.

However, we should not continue paying the full cost of exact path-state
comparison using large heap-heavy structures such as:

- `HashSet<object?>` copied per branch
- frontier scans over exact visited sets without indexing

That is correct, but too expensive.

The right direction is:

- keep exact semantics
- make path-state lookup cheaper
- use hashing as an index, not as a semantic shortcut

## What Hashing Is For

Hashing should be used to make state management cheaper in three ways:

1. **Incremental path fingerprints**
   - every path extension updates a deterministic fingerprint cheaply
2. **Fast candidate lookup**
   - explored-state tables can be keyed by compact fingerprints instead
     of probing richer structures first
3. **Subset / equivalence prefiltering**
   - fingerprints, sizes, and compact summaries can reject most
     impossible matches before exact verification

The important boundary is:

- a hash is a performance device
- not the full proof of state equality

If exactness matters, hash equality must still fall through to exact
verification on collision or possible match.

## Why This Is Better Than Pure Frontier Scans

The current exact non-DAG approaches we explored earlier were dominated
by the cost of comparing rich path states repeatedly.

Hashed path-state indexing changes the shape of that work:

- instead of scanning many frontier states blindly
- first jump to a much smaller candidate bucket
- then verify only those states that might actually match or dominate

This preserves semantics while making the common case cheaper.

## Why This Is Better Than Approximate Hash-Only Semantics

Hash-only equality would be tempting, but it is the wrong abstraction for
the compiler/runtime:

- collisions would create semantic risk
- correctness would become probabilistic
- debugging would become much harder

So the design should stay exact:

- deterministic fingerprint
- exact collision verification

That keeps the optimization honest.

## Relationship to DAG Work

This proposal is specifically for the **non-DAG** side.

It should not be used to paper over DAG workloads that ought to be
compiled differently. The right split is:

- DAG: exploit acyclicity directly
- non-DAG: use path-state indexing for exact simple-path semantics

That division gives us a cleaner compiler story and better benchmarks.

## Recommended Framing

The non-DAG runtime direction should be described as:

- **hashed path-state indexing with exact verification**

not:

- “probabilistic path equality”
- “approximate deduplication”
- “hash-based semantics”

The semantics remain exact. Only the indexing changes.
