# WAM Haskell Transpilation: Philosophy

## Why Haskell for the WAM

The Rust WAM spike (`feat/prolog-hybrid-wam-spike`) achieved functional correctness
(270/272 exact matches with SWI-Prolog) but a 343x performance gap. The root cause:
**Rust's ownership model conflicts with WAM's backtracking semantics.** Every choice
point required deep-cloning HashMap-based registers, bindings, and Vec-based stacks.

Haskell eliminates this entire class of problems through **persistent data structures**
with structural sharing. `Data.Map` clone is O(1) — both the old and new maps share
unmodified subtrees. This is exactly what WAM choice points need.

## Design Principles

### 1. Native Lowering, Not Interpretation

The Rust spike built a WAM **interpreter** in Rust — an instruction array processed
by a `step()` match loop. This adds dispatch overhead per instruction and prevents
compiler optimizations.

The Haskell target **natively lowers** WAM instructions to Haskell expressions.
Each compiled predicate becomes a Haskell function. `TryMeElse` becomes list
operations on the choice point stack. `Call` becomes a Haskell function call.
GHC optimizes the result as native Haskell code.

### 2. WAM as Fallback, Not Primary

The existing `haskell_target.pl` handles native lowering of Prolog predicates
to Haskell. The WAM target (`wam_haskell_target.pl`) is the **fallback** for
predicates that resist native lowering — those with complex backtracking,
cut, negation-as-failure, or meta-programming.

The pipeline:
1. Try `haskell_target.pl` native lowering (preferred)
2. If that fails → `wam_target.pl` compiles to WAM instructions
3. `wam_haskell_target.pl` lowers WAM instructions to Haskell

### 3. Immutable State, Pure Functions

WAM state transitions are pure functions: `WamState -> Maybe WamState`.
`Nothing` = failure, `Just s'` = success with new state. This makes:
- Backtracking trivial (keep the old reference)
- Testing simple (compare input/output states)
- Parallelism possible (or-parallelism via Haskell threads)

### 4. The WAM as a Test Oracle

The WAM provides a reference implementation. If the native Haskell lowering
and the WAM-compiled Haskell produce the same results, the native lowering
is correct. This is valuable because native lowering may use different
algorithms (e.g., bottom-up evaluation) that are hard to verify directly.

## Lessons from the Rust Spike

### What Worked
- First-argument indexing via `SwitchOnConstant` (177x speedup)
- Negation-as-failure fast path for `member/2` (direct list scan)
- Trail-based binding history with `__binding__` entries
- Cut barrier at `Allocate` time

### What Failed
- Full state cloning in choice points (HashMap/Vec clone per TryMeElse)
- Trail ordering: `unwind_trail` clobbered registers after `cp.regs` restore
- `PutStructure`/`PutList` with `WriteCtx` stack entries — fragile lifecycle
- `\+/1` as meta-builtin with full state save/restore

### Haskell Solutions
- **State cloning:** `Data.Map` structural sharing. O(1) "clone" = shared reference.
- **Trail ordering:** Not needed. `cpBindings` IS the authoritative snapshot. Just swap.
- **Structure construction:** Algebraic data types. `Str "member/2" [arg1, arg2]` — no WriteCtx.
- **Negation-as-failure:** Pattern matching on the goal. Pure function, no state mutation.

## Performance Expectations

| Component | Rust (current) | Haskell (expected) | Reason |
|-----------|---------------|-------------------|--------|
| ChoicePoint create | O(n) clone | O(1) shared ref | Data.Map structural sharing |
| Backtracking | O(n) restore | O(1) ref swap + O(k) trail | No register clone needed |
| Register access | O(1) HashMap | O(log n) Data.Map | Slightly slower per-access |
| Instruction dispatch | Match loop | Pattern match | Similar, GHC optimizes well |
| Overall 300-scale | 116s | <10s target | Dominated by clone savings |

The O(log n) register access is slightly slower than Rust's HashMap O(1), but
the O(1) choice point creation should far outweigh this for backtracking-heavy
workloads like the effective-distance benchmark.
