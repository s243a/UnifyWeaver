<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM functor parse — zero-alloc (D115)

**Date:** 2026-09-19. **Ledger:** D115. **Author:** Opus (coordinator).
**What:** rewrite `parseFunctorArity` / `parseFunctorName` (Go WAM target) to use
`strings.LastIndexByte` + a substring slice instead of `strings.Split` (+`Join`),
eliminating the biggest allocation-object source on the term path. Byte-identical.
**Allocation objects: 83.5M → 61.9M per B3 (−26%).** The first of the two
allocation levers the post-D114 alloc profile identified.

## Why (post-D114 allocation profile)

After D114 removed the O(heap) scan, the Go term path is GC-bound (~52% of CPU
in GC). An allocation profile of B3 (5000-pkg `resolve_layered`, `pprof` allocs)
ranked the sources two ways:

- **By object count** (drives GC mark work): `strings.genSplit` **27.9%
  (23.3M objects)** — second only to `Step` itself. Its caller is
  `parseFunctorName`/`parseFunctorArity`, and `isConsFunctor` calls
  `parseFunctorName` on **every cons check during list traversal**, so the
  `"name/arity"` `strings.Split` ran constantly.
- **By bytes**: the YSave/barrier stack copy dominates (75%) — a separate lever
  (D116, next).

D115 targets the object-count driver.

## The change

`strings.Split(f, "/")` allocates a `[]string` (and `parseFunctorName`'s
`strings.Join` allocates a second string). Both are avoidable:

```go
// parseFunctorArity: arity is the segment after the LAST '/'
if i := strings.LastIndexByte(f, '/'); i >= 0 {
    arity, _ := strconv.Atoi(f[i+1:])   // f[i+1:] is a slice, no alloc
    return arity
}
return 0

// parseFunctorName: name is everything before the LAST '/'
if i := strings.LastIndexByte(f, '/'); i >= 0 {
    return f[:i]                          // == Join(parts[:len-1], "/"), no alloc
}
return f
```

**Byte-identical by construction.** `parseFunctorArity` returned
`Atoi(parts[len-1])` — the segment after the last `/`, which is exactly
`f[i+1:]`. `parseFunctorName` returned `Join(parts[:len-1], "/")` — the string
before the last `/`, which is exactly `f[:i]` (Go strings are immutable, so a
slice of `f` is the same value, not a copy). The `str(...)` unwrap and the
no-slash fallback are unchanged. Every caller (`isConsFunctor`, `decompose`,
dispatch, etc.) sees identical values; only the allocations disappear.

## Verification (vs SWI oracle)

Regenerated the go example from the template and ran the full gates:

| gate | result |
| --- | --- |
| term corpus (`run_corpus_go.sh`) | 51/51 matched SWI |
| term differential (`run_differential_go.sh`, 2600 cases) | 0 divergences, 0 crashes |

(The change is a pure helper rewrite with identical semantics, so a runtime A/B
flag is unnecessary — byte-identity is by construction and confirmed against the
oracle. The go_store lane shares the template and is likewise unaffected in
correctness.)

## A/B — B3 allocation profile (5000-pkg), D114 → D115

| metric | D114 | D115 | Δ |
| --- | ---: | ---: | ---: |
| alloc objects (20 reps) | 83.5 M | **61.9 M** | **−26%** |
| `strings.genSplit` share | 27.9% (23.3 M) | **gone** | −23.3 M objects |
| alloc bytes (20 reps) | 61.3 GB | 60.7 GB | ~flat (genSplit objects are tiny strings) |
| resolve_ms (scale) | 2.322 s | 2.209 s | −4.9% |

The headline is **object count**: −26% fewer allocations means proportionally
less GC mark/scan work (the ~52% GC bucket). Bytes barely move because the split
strings are small; the byte hog is the YSave stack copy (D116). CPU drops a
modest ~5% — expected, since the win is mostly in reduced GC pressure and
memory-allocation traffic rather than mutator instructions.

## Verdict

A clean, byte-identical, zero-risk allocation cut: −26% of B3 allocation objects
by replacing two `strings.Split` helpers with `LastIndexByte` slicing, fixing
the churn at the source for every caller (chiefly `isConsFunctor` on the
list-traversal hot path). Sets up D116 (the YSave/barrier byte hog, 75% of
allocation bytes — the bigger memory win). `resolver.pl`/`resolver_store.pl`
UNMODIFIED.
