<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 John William Creighton (s243a) -->

# Go WAM conditional trailing via complete register snapshots (D117)

**Date:** 2026-09-19. **Ledger:** D117. **Author:** Opus (coordinator).
**What:** make every backtrack boundary in the Go WAM restore the FULL register
file (A + X + Y) from its snapshot, which lets the per-binding register-alias
trail entries be dropped entirely, and then skip trailing bindings of variables
younger than every live undo boundary (classic conditional trailing). Byte-identical
(runtime-flag A/B on both lanes). **B3 (5000-pkg) allocation bytes 853 MB → 651 MB
per resolve (−23.7%): −13.0% from dropping the reg-alias trail, −12.2% more from
conditional trailing.**

## Why this was the hard lever (the two dead ends first)

The trail is the #1+#2 Go allocation site at B3 (~45% of bytes: `trailBinding`
~33% + `bindUnbound` reg-alias ~13%). Two simpler attempts failed:

1. **Plain conditional trailing alone** (skip trailing a binding when the var is
   younger than every live boundary) is sound but yields **~0%**: measured skip
   rate **2.3%** (257,922 / 11,224,899 calls). WAM binds almost everything while
   the variable is resident in an argument/temporary register, and those bindings
   could not be skipped — see below.
2. **The register-alias coupling.** `bindUnbound` rewrites every register that
   aliases the just-bound var `u` to the bound value (the store lane reads
   register values directly — leaving `Regs[idx]=u` crashes it, confirmed), and
   trails a `RegIdx` entry so backtrack restores `Regs[idx]=u`. The binding entry
   and the reg-alias entries are ONE atomic undo. Skipping the binding for a
   register-resident young var while its reg-alias entry survives (or vice-versa)
   desyncs `Bindings[u.Idx]` from the registers on unwind — the register then
   dereferences to a stale value. This produced 16 wrong scenarios in the store
   lane. So register-resident vars (97.7% of bindings) could not be skipped, and
   the reg-alias trail could not simply be removed.

## The change (complete snapshots → drop reg-alias trail → skip freely)

The reg-alias trail exists ONLY because the register snapshots were incomplete:
`snapshotAllRegs` captured A + Y but skipped X-regs (deemed clause-local), and
`invokeGoalOnce` / `\+/1` restored no / only-A1-A3 registers. So the reg-alias
entries were doing the register restoration those snapshots omitted.

Make the snapshots complete and the reg-alias trail becomes redundant:

1. **Track `MaxXReg`** (new high-water mark, updated in `putReg` for idx∈[100,200)).
   The lowered emitter's register-establishing writes were converted from direct
   `Regs[i]=…` to `putReg` so the X/A/Y high-water marks stay accurate on the
   lowered path too (the interpreter already used `putReg`).
2. **`snapshotAllRegs`/`restoreSavedRegs` capture A + X + Y** (2-marker layout:
   acount, xcount, then the three ranges), with the same tail-clear discipline
   for each range that the Y-range already used.
3. **`invokeGoalOnce` and `\+/1`** take a complete `snapshotAllRegs` and restore
   it on their failure/undo paths; the foreign-results CP in `wam_go_target.pl`
   (`finishStreamResults`) switches from a partial 8+ycount slice (a silent no-op
   restore that leaned on the reg-alias trail) to a complete `snapshotAllRegs`.
4. **`bindUnbound` keeps the `Regs[idx]=val` rewrite but no longer trails it** —
   complete snapshots at every boundary restore aliasing registers to their
   pre-bind variable cell. A generator that rebinds `u` on its next solution
   still works: the boundary snapshot puts `Regs[idx]` back to `Unbound{u}` before
   the rebind.
5. **Conditional trailing:** `trailBinding` skips when the variable is younger
   than every live undo boundary. The floor = max over the youngest choice
   point's `VarFloor` (NextVarId at push, set centrally in `fillBarrier`, and
   explicitly on the foreign CP literal) AND every "bare-mark" boundary frontier
   (a `trailFloors []int` stack pushed at each `mark := TrailLen; …;
   unwindTrailTo(mark)` site: LoClauseSnapshot/T4, applySelectSolution,
   builtinCatch, `\+/1`, sub_atom/5, member/2, memberchk/2, select/3, the lowered
   ITE else branch, finishForeignResults/finishStreamResults, and a
   `pushTrailFloorAt(cp.VarFloor)` guard over the whole of `backtrack()` for the
   generator/foreign resume paths whose CP is truncated mid-resume). Driver-minted
   output cells [10000,11000) are never skipped (bound and read back across
   boundaries by the shim). Register-index-range vars (Idx<1000, lowered code) are
   always trailed since Idx<floor.

**Soundness rests on:** `allocVarId` Idx is monotonic and never reused, so a var
younger than a boundary (Idx ≥ its frontier) is provably discarded when that
boundary unwinds; and complete A+X+Y snapshots restore every live register, so
the dropped reg-alias entries are redundant. `NextVarId` is monotonic, so the
youngest CP carries the highest CP frontier; `trailFloor()` scans the whole
`trailFloors` stack for the max because `backtrack` can push a CP frontier lower
than an already-active bare-mark frontier.

**A/B mechanism.** `forceFullTrail` (`UW_GO_TRAIL_FULL=1`) trails every binding
unconditionally (reg-alias trail already gone), isolating the conditional-trailing
increment and serving as a correctness fallback. One binary.

## Verification (both lanes: cond vs SWI oracle AND cond vs full)

| lane | corpus vs SWI | differential vs SWI | cond == full |
| --- | --- | --- | --- |
| go (term) | 51/51 matched | 2600 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |
| go_store | 51/51 (identical to term) | 503 cases, 0 divergences / 0 crashes | `cmp`-clean (corpus + differential) |

Independently reproduced in the coordinator's tree (not only the implementing
subagent's worktree). The store lane was the decisive gate: it exercises the
foreign-results CP and the store-specific unwind boundaries the term corpus never
hits, and it caught every earlier incomplete-snapshot bug.

## A/B — B3 (5000-pkg), TotalAlloc bytes/resolve

| metric | baseline (D116) | full (reg-alias dropped) | conditional (D117) |
| --- | ---: | ---: | ---: |
| alloc bytes / resolve | ~853 MB | 742 MB (−13.0%) | **651 MB (−23.7%)** |

Resolve wall-clock dropped too (~0.62–0.66 s vs a noisier baseline), but the
allocation figure is the robust, stable metric.

## Trajectory

Go term-lane B3 allocation bytes/resolve: **853 MB (D116) → 651 MB (D117), −23.7%.**
Cumulative Go term-lane resolve: 14.8 s (pre-D114) → 2.322 (D114) → 2.209 (D115)
→ 1.444 (D116) → sub-second (D117). The trail — long the top Go allocation site —
is no longer dominant.

## Verdict

The trail-allocation lever, delivered soundly: complete register snapshots make
the per-binding register-alias trail redundant (−13%), and conditional trailing
then skips young-variable bindings freely (−12% more), for −23.7% B3 allocation
bytes, byte-identical on both lanes (verified vs SWI and cond-vs-full A/B). The
key insight — and the two recorded dead ends (plain skip = 2.3%; the reg-alias
desync) — is that in a register-machine WAM the trail cost cannot be skipped in
isolation; it has to be unblocked by making register restoration snapshot-complete
first. `resolver.pl`/`resolver_store.pl` UNMODIFIED.
