# WAM-Lowered Haskell: Specification

> **Background.** Before reading this spec, read
> [`WAM_HASKELL_LOWERED_BACKGROUND.md`](WAM_HASKELL_LOWERED_BACKGROUND.md)
> for the concrete shape of the existing Haskell code-generation paths and
> the precise meaning of "hybrid WAM with/without FFI." This document refers
> to path 1 / path 2 / path 3 as defined there. For the rationale, see
> [`WAM_HASKELL_LOWERED_PHILOSOPHY.md`](WAM_HASKELL_LOWERED_PHILOSOPHY.md).

## Scope

This document specifies the **WAM-lowered Haskell** code-generation path
(path 3 in `WAM_HASKELL_LOWERED_PHILOSOPHY.md`). It defines:

1. The selector mechanism and how the generator picks between paths 2
   and 3 per project and per predicate.
2. The shape of emitted Haskell code for each WAM instruction.
3. How lowered code interoperates with the existing interpreter, the
   FFI (`executeForeign`), and indexed-fact dispatch (`callIndexedFact2`).
4. What the generator is required to handle, what it is allowed to
   punt on (fall back to the interpreter), and how the fallback is
   signalled.

Out of scope: the native Haskell lowering path (`haskell_target.pl`),
the Rust/C/LLVM/etc. WAM targets, and anything upstream of the WAM IR.

## 1. Selector mechanism

### 1.1 `emit_mode/1` option

`write_wam_haskell_project/3` takes a new option:

```prolog
emit_mode(interpreter).                % all predicates via path 2 (default)
emit_mode(functions).                  % all predicates via path 3
emit_mode(mixed([Pred1/A1, ...])).     % named preds via path 3, rest via path 2
```

The selector hierarchy, checked in order:

1. **Option on the call** — `emit_mode(X)` in the `Options` list passed
   to `write_wam_haskell_project/3`.
2. **User-asserted fact** — `user:wam_haskell_emit_mode/1` if defined.
3. **Generator default** — `interpreter`.

### 1.2 Per-predicate fallback

Inside `mixed([...])` and `functions` modes, the generator evaluates each
predicate listed (or all predicates, for `functions`) against a
*lowering feasibility* check. The check is implemented as a Prolog
predicate:

```prolog
wam_haskell_lowerable(+PredIndicator, +WamCode, -Reason)
```

It succeeds when every instruction in `WamCode` appears in the
supported-instruction set (§2). On failure, `Reason` is a human-readable
string, the generator logs the predicate to `stderr` as
`[WAM-Haskell-Lowered] skipping Pred/Arity: Reason`, and the predicate
is compiled via path 2 into the same project. One project may therefore
contain a mix of interpreted and lowered predicates even in non-`mixed`
modes, as a result of automatic fallback.

### 1.3 Shared interpreter

If **any** predicate falls back to the interpreter — because the user
requested `interpreter`, because they used `mixed`, or because
`wam_haskell_lowerable/3` failed on a listed predicate — the emitted
project still includes `WamRuntime.hs`, `WamTypes.hs`, and a shared
`Predicates.hs` with the instruction array for the interpreted
predicates. Lowered predicates live in a new module `Lowered.hs` that
imports both.

If **no** predicate is interpreted, `Predicates.hs` is still emitted
(empty instruction array, empty label map) so `Main.hs` and
`WamRuntime.hs` compile unchanged. This is easier than conditionally
removing the imports.

## 2. Emitted code shape

### 2.1 Per-predicate function signature

Each lowered predicate emits a Haskell function of type:

```haskell
lowered_<pred>_<arity> :: WamContext -> WamState -> Maybe WamState
```

The body is a `do`-less pure expression — a chain of pattern-matches and
`let`-bindings mirroring the instruction sequence. It returns
`Just finalState` on success, `Nothing` on unification failure (which
the caller then routes through `backtrack`).

A manifest `Map.Map String (WamContext -> WamState -> Maybe WamState)`
is emitted in `Lowered.hs`:

```haskell
loweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)
loweredPredicates = Map.fromList
  [ ("category_ancestor/4", lowered_category_ancestor_4)
  , ("power_sum_bound/4",   lowered_power_sum_bound_4)
  ]
```

`WamRuntime.hs`'s `step` function gains a new case at the top of the
`Call` dispatch (before `executeForeign`):

```haskell
step !ctx s (Call pred _arity) =
  let sc = s { wsCP = wsPC s + 1 }
  in case Map.lookup pred (wcLoweredPredicates ctx) of
    Just fn -> fn ctx sc
    Nothing -> case executeForeign ctx pred sc of
      ...  -- unchanged
```

`WamContext` gains a new cold field `wcLoweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)`
populated from `Lowered.loweredPredicates` at startup in `Main.hs`.

This gives us the full interop invariant from the philosophy doc: a
lowered predicate can call an interpreted predicate via the normal
interpreter loop, and an interpreted predicate can call a lowered
predicate via `Map.lookup` at the `Call` dispatch point.

### 2.2 Instruction → Haskell mapping

The generator walks the WAM instruction sequence for a predicate once
and emits straight-line Haskell. A numeric "PC label" is emitted as a
Haskell `let ... in` boundary when there is an incoming branch target
(the predicate's entry, or a `TryMeElse` alternate, or similar). At
each label the emitter starts a new continuation.

| WAM instruction | Emitted Haskell (sketch) |
|---|---|
| `GetConstant c r` | `case IM.lookup r (wsRegs s0) of { Just (Atom c') \| c' == c -> ...; Just (Unbound v) -> let s1 = s0 { wsBindings = IM.insert v (Atom c) (wsBindings s0), wsTrail = TrailEntry v (IM.lookup v (wsBindings s0)) : wsTrail s0, wsTrailLen = wsTrailLen s0 + 1 } in ...; _ -> Nothing }` |
| `GetVariable r1 r2` | `let v = derefVar (wsBindings s0) (fromMaybe (Atom "") (IM.lookup r2 (wsRegs s0))); s1 = s0 { wsRegs = IM.insert r1 v (wsRegs s0) } in ...` |
| `PutConstant c r` | `let s1 = s0 { wsRegs = IM.insert r (Atom c) (wsRegs s0) } in ...` |
| `PutStructure f/n r` | Same as interpreter, inlined. Uses the compile-time arity specialization already in place. |
| `CallResolved pc arity` | Lookup target: if it is also lowered, direct Haskell function call; otherwise `let s1 = s0 { wsPC = pc, wsCP = wsPC s0 + 1 } in run ctx s1`. See §2.3. |
| `Call pred arity` | Same cross-path dispatch: check `wcLoweredPredicates`, then `executeForeign`, then `callIndexedFact2`, then label lookup. Matches interpreter fallback order. |
| `TryMeElse label` | Emit `ChoicePoint` construction inline, push onto `wsCPs`. The `cpNextPC` is the interpreter PC of `label` (computed at generation time). If the alternate clause is ALSO in a lowered function, we still record the PC — backtrack goes through the interpreter's `backtrack` helper. See §2.4. |
| `TrustMe` / `RetryMeElse` | Pop/replace top CP. Same as interpreter. |
| `Proceed` | `let ret = wsCP s0 in if ret == 0 then Just (s0 { wsPC = 0 }) else Just (s0 { wsPC = ret, wsCP = 0 })` — and then yield the state to whichever caller. For the entry function we return `Just s0` with `wsPC` set correctly; interop contract is in §2.4. |
| `Allocate` / `Deallocate` | Exactly the same as interpreter, inlined. |
| `BuiltinCall "is/2" _` | Direct call into a shared helper that the interpreter also uses. No duplication. |
| `BuiltinCall "!/0" _` | Set `wsCPsLen`/`wsCPs` truncation inline. Same shape as interpreter. |
| `SwitchOnConstant disp` | Emit a `case` on the current argument against the dispatch map. The map is emitted as a Haskell `case` expression if it's small (≤ 32 entries) or as a `Map.lookup` otherwise. The WAM-Haskell interpreter currently keeps this as a `HashMap` lookup; lowered code may inline for small dispatch tables. |

The emitter never *interprets* an instruction. Every instruction either
emits Haskell that does the same thing inline, or the predicate fails
the lowerability check and is punted to the interpreter.

### 2.3 Intra-predicate vs inter-predicate calls

When a lowered predicate's body contains a call to **itself** or to
**another lowered predicate in the same module**, the emitter produces
a direct Haskell function call. This is the case where GHC can inline
across predicate boundaries and where the largest wins are expected.

When the target is an **interpreted** predicate, the emitter produces:

```haskell
let s1 = s0 { wsPC = <targetPC>, wsCP = wsPC s0 + 1 }
in run ctx s1
```

That is, it sets the PC to the interpreted target and re-enters the
interpreter's `run` loop. On return, control comes back to the lowered
function. This is the interop bridge in one direction.

The other direction — interpreted code calling a lowered predicate —
uses the `wcLoweredPredicates` map lookup described in §2.1.

### 2.4 Backtracking across paths

A `ChoicePoint` snapshot doesn't care which path installed it. It
records `cpNextPC`, `cpRegs`, `cpBindings`, `cpTrail*`, `cpHeap*`,
`cpStack`, `cpCP`, `cpCutBar`, and optionally `cpAggFrame`/`cpBuiltin`.
On backtrack, the interpreter's `backtrack` function restores the
state and jumps to `cpNextPC`.

For this to work across paths:

- A lowered predicate that installs a CP records `cpNextPC` as the
  **interpreter PC** of the alternate clause. Always. Even if the
  alternate is in the same lowered predicate. The reason is uniformity:
  `backtrack` jumps to `wsPC = cpNextPC` and then the dispatch at the
  top of `step` picks up. If `cpNextPC` points inside a predicate whose
  entry is lowered, the dispatch finds it via `wcLoweredPredicates`;
  otherwise the instruction array gets hit.
- A lowered predicate that is *entered* on backtrack from an
  interpreted predicate sees `wsPC` pointing to its alternate-clause
  interpreter PC. The lowered function must therefore accept a
  `wsPC`-indexed entry and jump into the correct clause body. In
  practice this means the function dispatches on `wsPC` at the top:

  ```haskell
  lowered_category_ancestor_4 !ctx !s = case wsPC s of
    5  -> entry_clause_1 ctx s   -- primary entry
    20 -> entry_clause_2 ctx s   -- TryMeElse alternate
    _  -> Nothing  -- should not happen
  ```

  The `case` is bounded by the number of label entries the generator
  produced for this predicate, so it's a small constant.

This is the non-trivial interop cost. It is also the thing that keeps
the backtracking invariant across the path boundary. There is no
shortcut: if we tried to optimize it away by having lowered code hold
its own private CP representation, interpreted code could not backtrack
into it.

### 2.5 Aggregation and cut

- **Aggregation frames** (`AggPush`/`AggPop`/`AggEmit`) are the same
  persistent stack as the interpreter. Lowered code manipulates it
  inline but the representation is shared.
- **Cut** (`!/0`) truncates `wsCPs` and `wsCPsLen` the same way the
  interpreter does, using `wsCutBar` as the barrier. Lowered code sees
  `wsCutBar` set by `Allocate` exactly as the interpreter does.

## 3. Generator structure

### 3.1 New module layout

```
src/unifyweaver/targets/
    wam_haskell_target.pl            -- existing, gains emit_mode/mixed dispatch
    wam_haskell_lowered_emitter.pl   -- new, per-instruction Haskell emission
```

The new module exports:

- `lower_predicate_to_haskell(+PredIndicator, +WamCode, +Options, -HaskellCode)`
- `wam_haskell_lowerable(+PredIndicator, +WamCode, -Reason)`

`write_wam_haskell_project/3` in `wam_haskell_target.pl` gains logic that
partitions the input predicate list into (InterpretedList, LoweredList)
based on the selector hierarchy and the lowerability check, then:

- Calls the existing `compile_predicates_to_haskell/3` on the interpreted
  partition, producing the usual `Predicates.hs`.
- Calls `lower_predicate_to_haskell/4` on each predicate in the lowered
  partition, then concatenates the results into `Lowered.hs`.
- Adjusts `Main.hs` to populate `wcLoweredPredicates` from
  `Lowered.loweredPredicates`.
- Adjusts `WamRuntime.hs` (the template string in the generator) to
  check `wcLoweredPredicates` first in the `Call` dispatch chain.

### 3.2 Lowerability check

The initial implementation of `wam_haskell_lowerable/3` whitelists these
instructions:

```
GetConstant, GetVariable, GetValue, GetStructure,
PutConstant, PutVariable, PutValue, PutStructure,
CallResolved, Call, Proceed,
Allocate, Deallocate,
TryMeElse, RetryMeElse, TrustMe,
BuiltinCall "is/2", BuiltinCall "!/0", BuiltinCall "\\+/1",
BuiltinCall "length/2", BuiltinCall "member/2", BuiltinCall "</2", BuiltinCall ">/2",
SwitchOnConstant
```

Any other instruction causes `wam_haskell_lowerable/3` to fail with a
reason like `"unsupported instruction: PutStructureDyn"`. Future
expansions add entries to this whitelist.

Aggregation instructions (`AggPush`, `AggPop`, `AggEmit`) are *not* in
the initial whitelist. They land in a follow-up that has the
aggregation shape figured out.

## 4. Testing and correctness strategy

### 4.1 Oracle: the interpreter

Every lowered predicate must produce the same `wsBindings`, `wsTrail`,
`wsCPs`, and `wsRegs` state on every solution as the interpreter does
on the same inputs. The test harness runs each predicate twice — once
interpreted, once lowered — and compares the full solution stream via
the same solution-extraction helper (`collectSolutions`).

This is the *only* source of truth. If the two disagree, the lowered
emitter is wrong.

### 4.2 Benchmark parity

- `examples/benchmark/benchmark_effective_distance.py` gains a
  `wam_haskell_lowered` target alongside `wam_haskell`.
- Both targets run on the same 10k dataset.
- Rows must be byte-for-byte identical (same sort order, same number
  of digits).

### 4.3 Incremental landing

Ship the generator behind a feature flag at first. Specifically:

1. Phase A: `emit_mode(interpreter)` remains the default and is the
   only mode tests use. The `wam_haskell_lowered_emitter.pl` module
   exists but is only exercised by a single dedicated test
   (`tests/test_wam_haskell_lowered.pl`) that lowers one simple
   predicate and verifies it agrees with the interpreter.
2. Phase B: `emit_mode(functions)` becomes testable in the benchmark
   harness but not the default. Correctness tests run both modes.
3. Phase C: `mixed(HotPreds)` becomes usable. The default stays
   `interpreter`.
4. Phase D: after the default-flip criteria in the philosophy doc are
   met, `emit_mode/1` default changes to `functions` and the
   benchmark harness' `wam_haskell` target switches over. The
   `interpreter` mode stays available for fallback.

## 5. Out of scope for the initial landing

- **Aggregation in lowered code.** `AggPush`/`AggPop`/`AggEmit` stay
  interpreted-only until there is a dedicated follow-up with the
  aggregation frame semantics worked out.
- **`PutStructureDyn`** (runtime-parsed functors). Tracked as a
  separate task in the existing todo list.
- **Cross-target lowering (Rust, C, etc.).** This spec is about
  Haskell. The lowerability check and instruction-to-Haskell map are
  Haskell-specific.
- **Per-call-site inlining analysis.** The initial version is "whole
  predicate is lowered or whole predicate is interpreted." A future
  version could inline individual calls, but that is a separate
  design question about GHC inlining budget and inter-module
  optimization.
- **Profile-guided selection of `HotPreds`.** The user picks the
  list. An auto-selector based on profile data is a nice-to-have
  for a later phase.

## 6. Success criteria

The initial landing is considered complete when:

1. `emit_mode(functions)` can compile and run the
   `effective_distance` 10k benchmark end-to-end with output
   byte-identical to `emit_mode(interpreter)`.
2. `emit_mode(mixed([category_ancestor/4]))` produces a working
   project where `category_ancestor/4` is lowered and the rest of
   the predicates are interpreted, and the result is byte-identical
   to both other modes.
3. Backtracking and cut round-trip correctly across a lowered ↔
   interpreted boundary, demonstrated by a dedicated test that
   calls both directions.
4. The lowerability check correctly identifies at least one
   predicate that cannot be lowered (e.g., one using aggregation)
   and falls back to the interpreter without user intervention.
5. A benchmark number — even a slower one — exists. We do not ship
   a "we think this is faster" claim without a measurement. If the
   initial landing is slower than the interpreter on the 10k
   benchmark, that is acceptable but must be documented and
   tracked.

Performance parity (or improvement) over the interpreter is explicitly
**not** part of the initial landing. Correctness and the interop
contract are.
