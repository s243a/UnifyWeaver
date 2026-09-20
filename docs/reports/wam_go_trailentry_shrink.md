<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM TrailEntry shrink — drop the D117-orphaned register fields (D118)

**Date:** 2026-09-20. **Ledger:** D118. **Author:** Opus (coordinator).
**What:** remove the now-dead `RegIdx int` + `RegOld Value` fields from `TrailEntry`,
which D117 orphaned when it deleted register-alias trailing. Byte-identical by
construction. **B3 (5000-pkg) allocation bytes 651 MB → 580 MB per resolve
(−11.0%), −32.1% vs the pre-D117 baseline (853 MB).**

## Why (post-D117 profile)

A fresh B3 profile after D117 shows the Go term lane is **GC-bound** (~55–60% of
CPU in `scanobject`/`findObject`/`greyobject`/`memclr…`), so every allocation
byte cut is a direct CPU cut. Allocation bytes by site:

| site | % bytes |
| --- | ---: |
| `Step` (instruction dispatch) | 32% |
| `trailBinding` | 24% |
| `heapPush` | 18% |
| `snapshotAllRegs` | 11% |
| YSave sites | ~8% |

`trailBinding` — the trail-entry appends — is the #2 site. D117 removed ALL
register-alias trailing (complete A+X+Y register snapshots at every backtrack
boundary now restore aliasing registers), so:

- nothing constructs a `TrailEntry{RegIdx: ≥0}` any more (the sole creator,
  `trailBinding`, only ever set `RegIdx: -1`), and
- the only reader of `RegIdx`/`RegOld` was a now-unreachable branch in
  `unwindTrailTo`.

The two fields were pure dead weight on every entry: an `int` (8 B) plus a
2-word interface `Value` (16 B) on a ~56-byte struct.

## The change

`TrailEntry` becomes `{Addr int; Old Value; HadOld bool}` (~24 B). The single
creation site drops `RegIdx: -1`; `unwindTrailTo` drops the dead
`if entry.RegIdx >= 0 { … }` branch and always does the `Bindings`-restore.
Only `templates/targets/go_wam/state.go.mustache` changes.

**Byte-identical BY CONSTRUCTION:** the undone value for every binding
(`Bindings[Addr] = Old`) is unchanged; only two never-read fields are removed, so
no computation can observe the difference. No runtime A/B flag needed.

## Verification (both lanes vs SWI oracle)

| lane | corpus vs SWI | differential vs SWI |
| --- | --- | --- |
| go (term) | 51/51 matched | 2600 cases, 0 divergences / 0 crashes |
| go_store | 51/51 (identical to term) | 503 cases, 0 divergences / 0 crashes |

## A/B — B3 (5000-pkg), TotalAlloc bytes/resolve

| metric | D117 | D118 | Δ |
| --- | ---: | ---: | ---: |
| alloc bytes / resolve | 651 MB | **580 MB** | **−11.0%** |

The ~71 MB/rep cut is ≈47% of `trailBinding`'s bytes, matching the ~24 B removed
from a ~56 B entry. Because the lane is GC-bound, the allocation cut also lowers
GC-scan CPU.

## Trajectory

Go term-lane B3 allocation bytes/resolve: 853 MB (D116) → 651 MB (D117, −23.7%)
→ **580 MB (D118, −32.1% cumulative).**

## Verdict

The cleanup D117 opened up: with register-alias trailing gone, half of every
trail entry was dead. Removing it cuts B3 allocation 11% more, byte-identical,
at near-zero risk — a direct GC-pressure win on the GC-bound term lane. Remaining
top allocation sites are `Step` dispatch (32%, per-instruction Value boxing —
the biggest but structural) and `heapPush` (18%). `resolver.pl`/`resolver_store.pl`
UNMODIFIED.
