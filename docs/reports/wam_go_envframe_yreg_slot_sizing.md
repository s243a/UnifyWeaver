<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM EnvFrame.SavedYRegs slot-sizing — the D116 twin (D119)

**Date:** 2026-09-20. **Ledger:** D119. **Author:** Opus (coordinator).
**What:** slot-size `EnvFrame.SavedYRegs` from a fixed `[100]Value` (~1.6 KB
inline array, allocated on every `allocate`) to a `[]Value` sized to the used
Y-range (MaxYReg-200, typically ~10), restored with a tail-clear at Deallocate.
Byte-identical (runtime-flag A/B, both lanes). **B3 (5000-pkg) allocation bytes
586 MB → 439 MB per resolve (−25.0%); −48.5% cumulative vs the pre-D117 baseline
(853 MB).**

## Why (post-D118 profile)

A fresh B3 profile after D118 shows the Go term lane is GC-bound, and drilling
into `Step`'s 32% flat allocation with `pprof -list` pinned it to a single line:

```
4.78GB   Allocate handler:   env := &EnvFrame{CP:…, B0:…, CutB0:…, PrevE:…}
```

That one `&EnvFrame{}` per `allocate` instruction was **~28% of total B3 bytes**
— the #1 allocation site once D117/D118 cut the trail. The cause is
`EnvFrame.SavedYRegs [100]Value`: a `Value` is a 2-word interface, so the inline
array made every `EnvFrame` ~1.6 KB, whether or not its Y-slots were used. The
lowered path constructs `&EnvFrame{CP,B0}` too and never saves Y-regs, so it
carried the full 1.6 KB dead.

This is the exact `[100]Value` hog D116 slot-sized for the parallel `YSaves`
(choice-point) path — the D116 report explicitly flagged `EnvFrame.SavedYRegs`
as the latent twin "left (not in B3 top-3)". D117 (trail) and D118 (TrailEntry)
promoted it to the top.

## The change

Only the interpreter's Allocate/Deallocate save/restore uses `SavedYRegs` (the
lowered path's deallocate restores `env.CP` only). Following D116 exactly:

1. `EnvFrame.SavedYRegs`: `[100]Value` → `[]Value`.
2. **Allocate** (`wam_go_target.pl`): `ycount := max(0, MaxYReg-200);
   env.SavedYRegs = make([]Value, ycount); copy(env.SavedYRegs,
   Regs[200:200+ycount])` instead of `copy(env.SavedYRegs[:], Regs[200:300])`.
3. **Deallocate**: `n := len(env.SavedYRegs); copy(Regs[200:200+n],
   env.SavedYRegs); for i := 200+n; i < MaxYReg; i++ { Regs[i] = nil }` —
   the D116 tail-clear.
4. Lowered `&EnvFrame{CP,B0}` literals leave `SavedYRegs` nil — no backing array
   at all (their deallocate never touched it).

**Byte-identical BY CONSTRUCTION** (the D116 argument): `MaxYReg` is a monotonic
high-water, so slots ≥ it were never written (nil). The old full-100 restore both
restored the caller's live Y's and nil'd the callee-dirtied higher slots; the
sized save + tail-clear-to-MaxYReg reproduces both, landing `Regs[200:300]`
byte-for-byte. A lowered-allocated frame (nil `SavedYRegs`, `n==0`) clears
`Regs[200:MaxYReg]`, matching the old full-100 copy of an all-nil array.

**A/B mechanism.** `forceFullEnvY` (`UW_GO_ENVY_FULL=1`) makes Allocate save the
full 100 slots; Deallocate's tail-clear makes the output identical, isolating the
slot-sizing win and serving as a fallback. One binary.

## Verification (both lanes: sized vs SWI oracle AND sized vs full)

| lane | corpus vs SWI | differential vs SWI | sized == full |
| --- | --- | --- | --- |
| go (term) | 51/51 matched | 2600 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |
| go_store | 51/51 (identical to term) | 503 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |

## A/B — B3 (5000-pkg), TotalAlloc bytes/resolve

| metric | full-envY (≈D118) | sized (D119) | Δ |
| --- | ---: | ---: | ---: |
| alloc bytes / resolve | 586 MB | **439 MB** | **−25.0%** |

Resolve wall-clock also dropped (~0.67 s sized vs ~0.74 s full); allocation is the
robust metric on this GC-bound lane.

## Trajectory

Go term-lane B3 allocation bytes/resolve: 853 MB (D116) → 651 (D117, −23.7%) →
580 (D118, −32.1%) → **439 MB (D119, −48.5% cumulative)** — nearly halved across
the three-step trail+env campaign.

## Verdict

The D116 twin, delivered: slot-sizing `EnvFrame.SavedYRegs` cut B3 allocation 25%
more, byte-identical on both lanes (verified vs SWI and sized-vs-full), via the
proven tail-clear restore. Remaining top allocation sites are now `heapPush`
(heap growth — fundamental to term building) and `pushChoicePoint`/`trailBinding`
(genuine backtrack state). The parallel `EnvFrame.SavedYRegs`/`YSaves`/
`snapshotAllRegs` family of fixed-width register snapshots is now uniformly
slot-sized. `resolver.pl`/`resolver_store.pl` UNMODIFIED.
