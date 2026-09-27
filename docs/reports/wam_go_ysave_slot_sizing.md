<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM YSave slot-sizing — the byte hog (D116)

**Date:** 2026-09-19. **Ledger:** D116. **Author:** Opus (coordinator).
**What:** change the Go WAM Y-register call-frame snapshot (`YSaves`) from a
slice of fixed `[100]Value` arrays to slot-sized `[]Value` entries, and let
choice-point snapshots share those write-once entries by reference instead of
deep-copying them. Byte-identical (runtime-flag A/B). **Allocation bytes:
60.7 GB → 15.9 GB per B3 (−74%); resolve 2.209s → 1.444s (−35%).** Lever A of
the post-D114 Go allocation work.

## Why (post-D115 allocation profile, by bytes)

D114 removed the O(heap) scan; D115 removed the functor-split object churn. The
term path is still GC-bound, and by **bytes** the B3 allocation profile was
dominated (75%) by three sites, all one root cause:

| site | % bytes (D115) |
| --- | ---: |
| `copyYSaveStack` | 30% |
| `pushYSave` | 24.6% |
| `restoreBarrier` | 20.7% |

`YSaves` was `[][100]Value`: a `Value` is a 2-word interface, so each entry is
**~1.6 KB**. `pushYSave` stores one per `Call`; `copyYSaveStack` (per
choice-point push) and `restoreBarrier` (per backtrack) each **deep-copy
`depth × 1.6 KB`** — even though programs use ~10 Y-regs (there is already a
`MaxYReg` high-water field).

## The change

Two coupled moves, both with in-file precedent (`snapshotAllRegs`/
`restoreSavedRegs`, which already slot-size the parallel `SavedRegs` path):

1. **Slot-size the entry.** `pushYSave` snapshots `Regs[200:MaxYReg]` (a
   `[]Value` of `MaxYReg-200` slots) instead of the full 100. `YSaves` and
   `ChoicePoint.YSaves` become `[][]Value`.
2. **Share write-once entries.** With entries as slices, `copyYSaveStack` /
   `restoreBarrier` / `Clone` copy slice *headers* over the shared backing
   arrays instead of `depth × 1.6 KB`. Safe because an entry is **write-once**
   (created by `copy` in `pushYSave`, thereafter only read); the outer slice
   still gets a fresh `make()` so a later `pushYSave` append never perturbs a
   choice point's snapshot.

**The correctness-critical part** — byte-identity of the restore: the old
full-100 `popYSave`/`invokeAtPC` restore had a dual effect — it restored the
caller's live Y's **and** overwrote (to nil) any Y-slot the callee dirtied above
the caller's range (those were nil at save time). The sized restore reproduces
both: `copy(Regs[200:], s)` for the `len(s)` saved slots, then
`for i := 200+len(s); i < MaxYReg; i++ { Regs[i] = nil }` to clear the
callee-dirtied tail. This is exactly `restoreSavedRegs`' tail-clear (the same
"Y slots are global" hazard, documented there). Since `MaxYReg` is a monotonic
high-water, slots `>= MaxYReg` were never written (nil), so the sized+tail-clear
restore lands `Regs[200:300]` in byte-for-byte the same state the full-100
restore did.

**A/B mechanism.** `forceFullYSave` (`UW_GO_YSAVE_FULL=1`) makes `pushYSave`
store the full 100 slots — same output (the restore clears the tail regardless),
isolating the slot-sizing win and serving as a fallback. One binary.

## Verification (both lanes: map/sized vs SWI oracle AND sized vs full)

| lane | corpus vs SWI | differential vs SWI | sized == full |
| --- | --- | --- | --- |
| go (term) | 51/51 matched | 2600 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |
| go_store | 51/51 (identical to term) | 503 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |

## A/B — B3 (5000-pkg)

| metric | D115 | D116 | Δ |
| --- | ---: | ---: | ---: |
| alloc **bytes** (20 reps) | 60.7 GB | **15.9 GB** | **−74%** |
| YSave sites (copy+push+restore) | ~75% of bytes | ~6% of bytes | collapsed |
| resolve_ms (scale, sized) | 2.209 s | **1.444 s** | **−35%** |
| resolve_ms (scale, full-flag) | — | 1.772 s | isolates ~18% sizing win; rest is sharing |

The **bytes** headline is the memory/GC-scan axis: three-quarters of the resolve's
allocation traffic is gone, which is why GC-bound CPU drops 35% too. (Allocation
object *count* is ~flat — D115 handled that; D116 shrinks bytes-per-object.)

## Trajectory

Go term-lane 5k resolve: **14.8 s (pre-D114) → 2.322 s (D114) → 2.209 s (D115)
→ 1.444 s (D116)** — a cumulative **~10.3×**. Allocation bytes/B3: ~61 GB → 15.9 GB.

## Verdict

The biggest Go **memory** win: slot-sizing + write-once sharing of the YSave
call-frame snapshots cut B3 allocation bytes by 74% and resolve time by 35%,
byte-identical on both lanes (verified vs SWI and sized-vs-full). Follows the
`restoreSavedRegs` precedent, with the tail-clear as the load-bearing
correctness detail. A latent identical site remains — `EnvFrame.SavedYRegs`
(`[100]Value`, the Allocate/Deallocate path) — outside the B3 top-3 but the same
pattern if ever wanted. The Go term lane is now ~25× off Rust (1.44 s vs 58 ms),
down from ~250×; further work (Value boxing / small-int interning, or the
`SavedYRegs` twin) has a lower ceiling. `resolver.pl`/`resolver_store.pl`
UNMODIFIED.
