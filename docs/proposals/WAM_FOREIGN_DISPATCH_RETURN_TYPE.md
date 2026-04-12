# Proposal: Distinguish "No Handler" from "No Solutions" in WAM Foreign Dispatch

## Status

Draft — written to capture the design gap discovered during the
WAM-lowered Haskell work (#1344, #1346).

## Problem

Every WAM target's foreign dispatch function conflates two semantically
distinct failure modes into a single return value:

| Target | Function | Return type | "No handler" | "Handler found no solutions" |
|---|---|---|---|---|
| **Haskell** | `executeForeign` (~line 619) | `Maybe WamState` | `Nothing` | `Nothing` |
| **Rust** | `execute_foreign_predicate` (~line 1256) | `bool` | `false` | `false` |
| **LLVM** | `@wam_execute_foreign_predicate` (~line 1906) | `i1` (bool) | `false` | `false` |
| **WAT** | *N/A — uses `builtin_call` dispatch* | — | — | — |
| **C** | *N/A — uses `wam_execute_builtin()`* | — | — | — |

The three targets that have explicit foreign dispatch (Haskell, Rust,
LLVM) all exhibit the same conflation. WAT and C use generic builtin
dispatch without a separate foreign predicate mechanism.

The caller (the `Call` instruction dispatch chain in `step` / `vm.step`)
cannot tell whether the foreign dispatch returned "failure" because:

**(A)** No handler is registered for this predicate — the caller should
try the next resolution path (lowered functions, indexed facts, label
lookup).

**(B)** A handler IS registered and ran to completion, but found zero
solutions — the predicate genuinely fails and the caller should
backtrack, NOT try the next resolution path.

### Where this matters

The conflation was harmless when the only resolution paths after
`executeForeign` were `callIndexedFact2` and `wcLabels` label lookup.
Neither of those would redundantly search a graph that the FFI already
explored.

It became load-bearing when the **WAM-lowered Haskell** path added
`wcLoweredPredicates` to the dispatch chain. A lowered function for
`category_ancestor/4` was tried after `executeForeign` returned
`Nothing` for "no solutions." The lowered function then explored the
same graph clause-by-clause via WAM dispatch — exponentially slower —
causing non-termination on realistic datasets.

The current fix (#1346) sidesteps the issue by:
1. Giving `executeForeign` priority over `wcLoweredPredicates`.
2. Restricting lowering to single-clause predicates only.

These are correct but limiting. Multi-clause lowering remains blocked
until the dispatch can distinguish (A) from (B).

## Options

### Option 1: Three-valued return type

Change `executeForeign` to return a three-valued result:

**Haskell:**
```haskell
data ForeignResult = ForeignSuccess WamState
                   | ForeignNoSolutions
                   | ForeignNoHandler
```

**Rust:**
```rust
enum ForeignResult {
    Success(bool),       // true = succeeded, state updated
    NoSolutions,         // handler ran, zero results
    NoHandler,           // no handler registered for this predicate
}
```

The `Call` dispatch chain becomes:
```
case executeForeign ctx pred sc of
  ForeignSuccess sr  -> Just sr
  ForeignNoSolutions -> Nothing        -- backtrack, skip remaining chain
  ForeignNoHandler   -> <try next path: lowered → indexed → labels>
```

**Pros:** Clean, explicit, type-safe. Each target's dispatch chain
can make the right decision without heuristics.

**Cons:** Touches every target's foreign dispatch function and every
call site. Non-trivial migration.

### Option 2: Predicate-set check before dispatch

Instead of changing the return type, check whether the predicate HAS
a handler before calling it:

**Haskell:**
```haskell
hasForeignHandler :: WamContext -> String -> Bool
hasForeignHandler ctx pred = pred `elem` wcForeignPreds ctx

step ctx s (Call pred _arity) =
  let sc = s { wsCP = wsPC s + 1 }
  in if hasForeignHandler ctx pred
     then case executeForeign ctx pred sc of  -- authoritative
            Just sr -> Just sr
            Nothing -> Nothing                -- backtrack
     else <try lowered → indexed → labels>
```

Add a `wcForeignPreds :: [String]` field to `WamContext` listing all
predicates that have FFI handlers. Populated at startup from the
`foreignPreds` list in `Main.hs`.

**Pros:** No change to `executeForeign`'s return type. Minimal diff
per target. The check is O(n) on a small list (typically 1–3 preds).

**Cons:** The `wcForeignPreds` list must be kept in sync with the
pattern-match cases in `executeForeign`. If a handler is added to
`executeForeign` but not to `wcForeignPreds`, the check gives wrong
answers silently.

### Option 3: Dispatch priority (current workaround)

Keep `executeForeign` first in the chain and restrict lowering to
single-clause predicates. Multi-clause predicates with FFI handlers
use the FFI; multi-clause predicates without FFI handlers use the
interpreter.

**Pros:** Already implemented. Zero additional code.

**Cons:** Blocks multi-clause lowering entirely. The lowered path
can never handle predicates like `category_ancestor/4` even if it
could generate correct code, because the dispatch can't tell when
to skip the FFI.

## Recommendation

**Option 2** (predicate-set check) for the near term. It's the
smallest change that unblocks multi-clause lowering:

- One new field on `WamContext` (Haskell) / `VmState` (Rust) /
  equivalent (WAT, LLVM).
- One `if` check at the top of the `Call` dispatch.
- The `foreignPreds` list already exists in `Main.hs` — just
  thread it into `WamContext`.

Option 1 (three-valued return) is the long-term clean solution but
should be tackled as a cross-target refactor after the lowered-Haskell
path has proven its value. It's overkill for unblocking one feature.

## Scope

This proposal covers only the dispatch semantics. It does NOT cover:

- Which predicates to lower (that's the kernel detection work).
- How to lower multi-clause predicates (that's the emitter design).
- Whether the Rust target should adopt `wcLoweredPredicates` too
  (separate discussion).

## References

- #1344 — feat: WAM-lowered Haskell Phases 1–4
- #1346 — fix: dispatch priority + single-clause restriction
- `docs/design/WAM_HASKELL_LOWERED_BACKGROUND.md` — taxonomy of
  Haskell code-gen paths
- `docs/design/WAM_HASKELL_LOWERED_SPECIFICATION.md` §2.1 — the
  `wcLoweredPredicates` dispatch mechanism
- `src/unifyweaver/targets/wam_rust_target.pl:1255` —
  `execute_foreign_predicate` (Rust equivalent)
- `src/unifyweaver/targets/wam_haskell_target.pl:608` —
  `executeForeign` (Haskell)
