<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM heapConsAfterUnbound — O(heap) scan → O(occurrences) var-index (D114)

**Date:** 2026-09-19. **Ledger:** D114. **Author:** Opus (coordinator).
**What:** replace the whole-heap linear scan in `heapConsAfterUnbound` (Go WAM
target) with an `*Unbound → heap-position` index, turning an O(heap)-per-
list-step cost into O(occurrences-of-the-var). Runtime-flag A/B
(`UW_GO_HEAPSCAN=1` keeps the scan). **Term-lane B3 resolve: 14.2s → 2.3s
(6.1×), byte-identical.**

## How this was found (profile first, not the interner)

The plan was to port the Rust interner levers (D112 FxHash, D113 Sym-threading)
to Go. A `pprof` CPU profile of the Go B3 (5000-pkg `resolve_layered`) said
otherwise — the interner does not appear at all:

| hot spot | flat / cum | what |
| --- | --- | --- |
| `(*WamState).heapConsAfterUnbound` | **43.7% / 74.0%** | scans the whole heap per list-traversal step |
| `runtime.ifaceeq` | **28.9%** | the `cell == v` interface compare that scan runs on every heap cell |
| GC (`gcBgMarkWorker`/`scanobject`/`mallocgc`) | ~14% | allocation churn |

Go maps already use a fast non-cryptographic hash, so the D112 FxHash lever has
no Go analog; and Go carries functors as `"name/arity"` strings, so D113 is a
transposed string-churn issue that the profile showed is negligible next to the
scan. The real cost is algorithmic.

## The bug

`heapConsAfterUnbound(v *Unbound)` answered "does `v` sit in a heap cell whose
next cell is a cons?" by scanning the entire heap:

```go
for i := 0; i < vm.HeapLen; i++ {
    if vm.Heap[i] == v && i+1 < vm.HeapLen { /* check Heap[i+1] is a cons */ }
}
```

It reconstructs a list spine from heap adjacency for the cases the
`put_structure` bind-through misses (A-registers, embedded unbound tails). It is
called once per list-traversal step (`valueListHeadTail`/`listHeadTail`), so
walking an N-element list on a large heap is O(N × HeapLen). At 5k scale the
heap holds the whole catalog term → **14.8s per resolve** (vs Rust's ~58ms).

## The fix

`heapVarPos map[*Unbound][]int` on `WamState`, appended in `heapPush` whenever
the pushed value is an `*Unbound`. `heapConsAfterUnbound` consults it and tests
only that var's recorded cells' `+1` neighbours.

**Correctness — byte-identical to the scan, by construction:**
- **Keyed by pointer, never `Idx`.** `lowered.go` mints many distinct `*Unbound`
  sharing one `Idx` (e.g. `Idx:200` at dozens of sites), and the scan's own test
  is pointer identity (`cell == v`). A map keyed by `Idx` would conflate them.
- **All positions, first-occurrence semantics.** A var is re-pushed by
  `set_value`/`unify_value`, so it can occupy multiple heap cells; the scan
  returns the cons after the *first* occurrence that has one. The index stores
  every position; the lookup iterates them (append order == ascending heap index,
  the heap being append-only) and returns the cons after the first live
  occurrence — the same cell.
- **Completeness.** Every `*Unbound` that reaches a top-level heap cell does so
  through `heapPush` (verified: the only mid-heap writes are `Heap[addr] = s/l`
  filling a just-pushed cons cell, never a var), so the index misses nothing.
- **Truncation staleness.** `heapTrimTo` (backtrack) cuts the heap without
  touching the map, and a trimmed slot can later hold a different value. The
  lookup drops any position with `i >= HeapLen || Heap[i] != v` (compacting the
  slice in place), so stale entries can never produce a wrong hit — no
  backtrack-time rewind needed.
- **Lifecycle.** Initialised in `NewWamState`/`NewWamStateFromCtx`, deep-copied
  in `Clone` (the aggregate sub-VM's copied heap holds the same pointers, so its
  positions carry over; independent slices so compaction doesn't cross-mutate).

`heapListAfterUnbound` (different semantics — scans forward for any list-like
cell; not in the profile's hot path) is left as the scan, to keep the change
minimal.

**A/B mechanism.** Go has no build-tag toggle in this target, so the lever uses
a runtime flag: `forceHeapScan` (from `UW_GO_HEAPSCAN=1`) keeps the exact
original scan and skips index maintenance. One binary; env selects. This is both
the A/B baseline and a permanent correctness fallback.

## Verification (both lanes, map vs SWI oracle and map vs scan)

| lane | corpus vs SWI | differential vs SWI | map == scan (byte-identity) |
| --- | --- | --- | --- |
| **go (term)** | 51/51 matched | 2600 cases, 0 divergences / 0 crashes | corpus 51 lines + differential 2600 lines `cmp`-clean |
| **go_store** | 51/51 matched (identical to term corpus) | 503 cases, 0 divergences / 0 crashes | corpus 51 lines + differential 503 lines `cmp`-clean |

## A/B — B3 resolve (5000-pkg), map (default) vs scan (`UW_GO_HEAPSCAN=1`)

| lane | scan (baseline) | map (D114) | speedup |
| --- | ---: | ---: | ---: |
| **go (term)** | 14.219 s | **2.322 s** | **6.1×** |
| go_store | 0.543 s | 0.542 s | ~1.0× (no change) |

**Why the store lane doesn't move:** the store-backed build serves facts from
the D43 indexed seek store instead of materialising the whole catalog as one
heap term, so its heap stays small and the scan was already cheap. The O(heap)
pathology is specific to the term lane (large heap). The fix is correct and
harmless in both; the store lane's 0.54 s already made it the performant path
for large catalogs.

## Verdict

Profiling-first turned a planned ~2% interner micro-port into a **6.1× term-lane
win** by finding the real O(N × heap) hot spot. Byte-identical to the scan on
both lanes (and to the SWI oracle), with the scan retained as an env-selectable
fallback. The term lane is still ~40× off Rust (2.3 s vs 58 ms) — the residual
is GC pressure (~14% in the original profile) and per-step interface work, so a
follow-up Go re-profile is the next step if the term lane needs to go further;
for large catalogs the store lane (0.54 s) is already the recommended path.
`resolver.pl`/`resolver_store.pl` UNMODIFIED.
