<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025-2026 John William Creighton (@s243a)
-->
# Proposal: Share WAM-WAT indexing and neck-cut optimizations across WAM-family backends

**Status:** Draft proposal for review.
**Target audience:** WAM-family target maintainers, performance owners.

---

## Summary

Over the last ~20 PRs, the WAM-WAT target accumulated a set of
indexing and CP-elision optimizations (type_dispatch_a1,
trail-aware neck_cut_test, arg_call fusion family, tail_call_5 +
clause-end fusions) that delivered ~55% cumulative improvement on
`bench_sum_big` vs the Phase 6 baseline. All of this work currently
lives inside `src/unifyweaver/targets/wam_wat_target.pl`.

The sibling WAM-family backends —
`wam_go_target.pl`, `wam_rust_target.pl`, `wam_llvm_target.pl`,
`wam_ilasm_target.pl`, `wam_elixir_target.pl`,
`wam_jvm_target.pl` (Jamaica/Krakatau) — parse the **same WAM text
intermediate form** but do not see any of these optimizations.
They receive a less-aggressively-fused instruction stream.

This proposal outlines moving the **pipeline-level** optimizations
(the peephole passes that rewrite the instruction stream before
encoding) up into the shared `wam_target.pl` layer, while leaving
the **encoding-level** specializations (the WAT-specific
instructions and their runtime handlers) in each backend.

---

## What's currently WAM-WAT-specific

### Pipeline passes in `wam_wat_target.pl`

Running in this order (from `pass1_parse_predicates`):

1. `peephole_neck_cut` — `try_me_else + allocate + guard + !/0` →
   `neck_cut_test` (CP-elision for cut-deterministic 2-clause
   predicates)
2. `peephole_nested_arith` — one-level nested arithmetic → chained
   fused-arith ops
3. `peephole_fused_arith` — `X is A+B` → `fused_is_add` (and
   friends for −, ×, const variants)
4. `peephole_direct_builtins` — `builtin_call('arg/3', 3)` →
   `arg_direct` etc. (bypass $execute_builtin dispatch)
5. `peephole_arg_to_a1` — `arg_reg_direct + put_value(_, A1)` →
   `arg_to_a1_reg` (arg/3 result directly feeds next call's A1)
6. `peephole_arg_call_k` — `arg_to_a1 + setup + call(Pred, K)` →
   `arg_call_reg_K` (four-way fusion of arg + call setup)
7. `peephole_tail_call_k` — `put_value × 5 + deallocate + execute`
   → `tail_call_5`; plus clause-end fusions
   (`deallocate_proceed`, `deallocate_builtin_proceed`, etc.)
8. `peephole_type_dispatch` — 2/3-clause type-guarded predicates →
   `type_dispatch_a1` (tag-based first-arg indexing)

### New instruction tags that go with them

Tags 36–72 in `wam_wat_target.pl`:

- 36–40: fused arithmetic
- 41–45: direct-dispatch builtins (arg_direct, functor_direct, …)
- 46–61: arg_call family (K=1/2/3, reg/lit, live/dead variants)
- 62–71: tail-call + clause-end family
- 72: type_dispatch_a1

Plus one env-frame-layout change (PR #1531): a 4-byte `trail_mark`
slot at offset +392 used by neck_cut_test's trail-unwind on
guard failure.

---

## The split: pipeline vs encoding

Not everything should move. The split I propose:

### Move to shared layer (`wam_target.pl`)

**Instruction-stream transformations** that operate on WAM term
lists — the peephole passes 1–8 above. These produce abstract
instructions like `type_dispatch_a1(AtomLbl, IntLbl, CmpdLbl,
DefaultLbl)` that each backend then encodes however it wants.

Why: these are purely about recognizing patterns in the Prolog-
derived WAM stream. The patterns exist regardless of which backend
consumes the output.

### Keep in each backend

- **Instruction encoding** — the tag numbers, operand packing, and
  wire format for each instruction. Each backend has its own
  encoding (WAT uses 20-byte data-segment records; Go uses Go
  slices; JVM uses CIL bytecode; etc.).
- **Instruction implementation** — the WAT/Go/Rust/… function that
  handles each new instruction tag at runtime. This is the
  backend's job.
- **Env-frame / CP-frame layout** — each backend has its own
  storage model. The trail-mark slot in WAM-WAT's env frame
  belongs to the WAT backend; other backends may store it
  elsewhere or synthesize it differently.

### Opt-in per backend

Each backend declares which abstract instructions it implements.
The peephole layer checks the target's capability before emitting
a fused instruction, falling back to the unfused original if the
backend doesn't handle the new form yet.

Concretely: a new predicate
`wam_target:backend_supports_instruction(BackendName, InstrName)`.
Peephole passes consult this to decide whether to fire.

---

## Why do this

### Broadens impact

`bench_sum_big` improved ~55% on WAM-WAT. The same predicates
compiled through `wam_go_target` see none of that gain. Sharing
the peepholes would let Go/Rust/LLVM/ILAsm/Elixir/JVM workloads
pick up whatever portion of the win each backend implements.

### Reduces duplication

The peephole patterns are not backend-specific. They detect
`try_me_else(L) + guard+cut → trust_me` and similar shapes in
the WAM text. Each backend re-implementing the same detection
logic would diverge over time and miss edge cases.

### Aligns with the "single logical front-end" philosophy

From `docs/targets/overview.md`:

> Single logical front-end: The compiler analyses clauses once,
> producing a target-neutral description of joins, projections,
> and constraints.

Today WAM-WAT is a second lowering pass on top of the WAM text.
Its optimizations are target-specific by accident, not design.

---

## Implementation strategy

### Phase 0: Establish the split

Refactor `wam_wat_target.pl` to import peephole passes from a new
`wam_peephole.pl` module rather than define them inline. No
behaviour change; just move code. Validate by running the existing
bench suite — output should be byte-identical.

### Phase 1: Capability declarations

Add `wam_target:backend_supports_instruction/2` facts. WAM-WAT
declares support for all current instructions (36–72). Other
backends declare support for whatever they already implement
(almost certainly only 0–30 range).

Peephole passes check `backend_supports_instruction/2` with the
target backend name (threaded through `Options`) before emitting
fused forms.

### Phase 2: Per-backend opt-in

Each of the sibling backends decides which abstract instructions
to implement. Likely tiers:

**Tier A** (easy to implement anywhere):
- `deallocate_proceed` (trivial; saves one dispatch)
- `deallocate_builtin_proceed` (deallocate + builtin + proceed)
- `neck_cut_test` (guard + cut collapse; trail-aware variant
  optional)

**Tier B** (medium effort):
- `tail_call_5` family
- Fused arithmetic (`fused_is_add` etc.)
- Direct-dispatch builtins

**Tier C** (larger changes):
- `type_dispatch_a1` (requires per-backend tag-dispatch runtime)
- `arg_call_reg_3` family (requires builtin_arg fast path)

Start with Tier A, measure, proceed if signals are good.

### Phase 3: Benchmark parity

Port `examples/wam_term_builtins_bench/` to a generic driver that
runs against any WAM backend. Compare pre/post for each backend.

---

## Risks

### Risk 1: Peepholes interact with backend semantics

A pass that's correct for WAM-WAT might be incorrect for a
different backend if their runtime semantics diverge (e.g.,
different CP-count-guard behaviour in `trust_me`). The PR history
for WAM-WAT shows several subtle bugs of this shape
(`remove_label_trust_me` silently succeeding on 3-clause
predicates; `neck_cut_test` needing trail unwind for
`term_depth_args/5`). Each backend needs to audit whether the
same caveats apply.

**Mitigation:** The capability-declaration system lets each
backend opt in per-instruction. A backend that hasn't audited a
particular peephole simply doesn't declare support, and the
pattern stays unfused for that backend.

### Risk 2: Benchmark framework portability

The WAM-WAT bench (13 workloads) only runs through the WAT/WASM
path. Other backends would need their own driver. The workloads
themselves are portable (pure Prolog) but the "run each bench
100K times and time it" harness is WAM-WAT-specific.

**Mitigation:** A generic harness is plausible (each backend
already compiles to an executable; we just need to invoke it in
a loop). Out of scope for this proposal but tractable.

### Risk 3: Hidden dependencies on WAT-specific state

PR #1531 added a `trail_mark` slot to the env frame. If the
shared peephole emits `neck_cut_test` assuming that slot exists,
other backends would need to add equivalent state. This
cross-cuts runtime data structures.

**Mitigation:** Each backend implements its trail-mark storage
however suits it (env slot, global, CP frame, etc.). The shared
peephole just emits the abstract `neck_cut_test` instruction; the
backend's handler decides how to unwind the trail.

---

## Non-goals

- **Not** a full WAM runtime unification project. Each backend
  keeps its own runtime.
- **Not** a performance promise. Sharing the peepholes only helps
  backends that actually implement the fused instructions;
  unimplemented instructions fall through unchanged.
- **Not** affecting the WAM-WAT target's existing functionality.
  Phase 0 is a pure refactor.

---

## Estimated scope

Rough LOC impact (order of magnitude):

| Phase | Files touched | Est. LOC change |
|-------|--------------|-----------------|
| 0 — extract peepholes | `wam_wat_target.pl` → new `wam_peephole.pl` | ~800 moved; 0 new behaviour |
| 1 — capability facts | `wam_target.pl` + 1 line per backend | ~50 |
| 2 — Tier A per backend | 6 backends × ~200 LOC each | ~1200 |
| 3 — benchmark harness | new generic driver | ~300 |

Phase 0–1 is 1–2 PRs. Phase 2 is one PR per backend (~6 PRs) on
an opt-in schedule.

---

## Decision criteria

Execute this proposal **if**:

- A WAM-family backend (Go, Rust, LLVM, ILAsm, Elixir, JVM) has a
  performance concern that existing techniques would address, OR
- A contributor wants to port specific optimizations into
  their backend and doing it once-shared is cleaner than
  once-per-backend.

Defer **if**:

- No WAM-family backend has a near-term performance pressure.
- The WAM-WAT-specific path is sufficient for current use cases.

The proposal's value is **optionality**: having the pattern
established means when a backend does need this, the work is
port-and-declare rather than reinvent.

---

## Alternatives considered

### Alternative A: Per-backend reimplementation

Each backend implements its own peepholes, in its own style. No
shared code.

**Why not:** Duplication. The patterns (multi-clause type guards,
guard+cut, arg+call) are universal. Each backend reimplementing
them would diverge and miss edge cases.

### Alternative B: Move the peepholes into `wam_target.pl` directly, no capability gate

Apply all peepholes regardless of backend. Any backend that
doesn't handle a fused instruction falls back via "unknown
instruction" handling.

**Why not:** Most backends don't have "graceful unknown-instruction
fallback" — they'd error out or silently produce wrong code.
Capability gating is safer.

### Alternative C: Do nothing, keep WAM-WAT-specific

Status quo. Other backends miss the gains but don't take on any
complexity.

**Why not:** The WAM-WAT work took ~20 PRs. Having it benefit
only one backend is a poor return on that investment.

---

## Follow-ups if this proposal is accepted

- **Spec each peephole's pre/post-conditions** — the WAM-WAT code
  has these implicit (e.g., "only fires when guard isn't a
  type-test"). Make them explicit in the shared layer.
- **Document invariants for shared instruction abstractions** —
  e.g., what `type_dispatch_a1`'s runtime contract is so each
  backend implements it consistently.
- **Cross-backend test suite** — run the same Prolog predicates
  through every backend and verify output matches.

---

## References

- `docs/design/WAM_WAT_ARCHITECTURE.md` — detailed design of the
  WAM-WAT target these optimizations were developed against.
- `docs/targets/wam-wat.md` — user-facing overview.
- `docs/targets/overview.md` — the single-logical-front-end
  philosophy referenced above.
- Recent WAM-WAT PRs (#1456 through #1539) — commit history of
  the work being proposed for sharing.
