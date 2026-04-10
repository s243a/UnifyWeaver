# WAM Haskell Performance: Specification

This document describes the technical shape of the optimizations on
`feat/wam-haskell-perf-profile`. It covers (1) the data shapes that are
already in place, (2) the WamState hot/cold split that is the next planned
change, (3) the FFI optimization that follows it, and (4) a sketch of the
"compile WAM to Haskell functions" path that is being deferred to a
separate exercise.

For the *why* of each decision, see `WAM_HASKELL_PERF_PHILOSOPHY.md`. For
the commit-by-commit history, see `WAM_HASKELL_PERF_IMPLEMENTATION_PLAN.md`.

## 1. Current state (after the perf branch optimizations)

### 1.1 Value type

```haskell
data Value = Atom String
           | Integer Int
           | Float Double
           | VList [Value]
           | Str String [Value]
           | Unbound !Int        -- variable ID, not name
           | Ref Int
           deriving (Eq, Ord, Show)
```

`Unbound` carries an `Int` variable ID drawn from `wsVarCounter`. The
binding table (`wsBindings`) is keyed on the same `Int`, so binding
lookup never hashes a string.

### 1.2 Register IDs

```haskell
type RegId = Int
-- A1-A99: 1-99
-- X1-X99: 101-199
-- Y1-Y99: 201-299
```

The compile-time helper `reg_name_to_int/2` (Prolog side, in
`wam_haskell_target.pl`) parses register names like `"A1"`, `"X3"`, `"Y2"`
once when emitting the Haskell instruction list, so the runtime never
re-parses them.

`getReg`/`putReg` use `rid >= 200` to detect Y-bank registers and dispatch
to the environment frame instead of `wsRegs`:

```haskell
{-# INLINE getReg #-}
getReg :: Int -> WamState -> Maybe Value
getReg rid s
  | rid >= 200 = findYReg rid (wsStack s)
  | otherwise  = derefVar (wsBindings s) <$> IM.lookup rid (wsRegs s)
```

### 1.3 Trail and choice point shapes

```haskell
data TrailEntry = TrailEntry !Int !(Maybe Value)
                deriving (Show)

data EnvFrame = EnvFrame !Int !(IM.IntMap Value)
              deriving (Show)

data ChoicePoint = ChoicePoint
  { cpNextPC   :: !Int
  , cpRegs     :: !(IM.IntMap Value)
  , cpStack    :: ![EnvFrame]
  , cpCP       :: !Int
  , cpTrailLen :: !Int
  , cpHeapLen  :: !Int
  , cpBindings :: !(IM.IntMap Value)
  , cpCutBar   :: !Int
  , cpAggFrame :: !(Maybe AggFrame)
  , cpBuiltin  :: !(Maybe BuiltinState)
  } deriving (Show)
```

The `IntMap` fields (`cpRegs`, `cpBindings`) are O(1) to "snapshot" because
the live state and the CP literally share subtrees through structural
sharing. `cpTrailLen` and `cpHeapLen` are integer indices, so trail/heap
restore is O(k) where k is the number of entries to drop, not O(n).

`BuiltinState` lets choice points carry their own retry logic, so a fact-
indexed call can self-pop without going through `TrustMe`:

```haskell
data BuiltinState
  = FactRetry !String ![String] !Int  -- varName, remaining values, returnPC
  | HopsRetry !Int    ![Int]    !Int  -- varId,   remaining Hops,    returnPC
  deriving (Show)
```

(`HopsRetry` uses an `Int` variable ID since 803ed711.)

### 1.4 Current WamState (the thing we are about to split)

```haskell
data WamState = WamState
  { wsPC            :: !Int
  , wsRegs          :: !(IM.IntMap Value)
  , wsStack         :: ![EnvFrame]
  , wsHeap          :: ![Value]
  , wsHeapLen       :: !Int
  , wsTrail         :: ![TrailEntry]
  , wsTrailLen      :: !Int
  , wsCP            :: !Int
  , wsCPs           :: ![ChoicePoint]
  , wsCPsLen        :: !Int
  , wsBindings      :: !(IM.IntMap Value)
  , wsCutBar        :: !Int
  , wsCode          :: !(Array Int Instruction)
  , wsLabels        :: !(Map.Map String Int)
  , wsBuilder       :: !Builder
  , wsVarCounter    :: !Int
  , wsAggAccum      :: ![Value]
  , wsForeignFacts  :: !(Map.Map String (Map.Map String [String]))
  , wsForeignConfig :: !(Map.Map String Int)
  } deriving (Show)
```

17 fields. Every record-update expression `s { wsPC = ... }` allocates a
fresh 17-field constructor, copies all field pointers, and writes the
modified ones. With ~10⁵ steps per benchmark run, that is ~10⁵ × 17 = 1.7M
pointer copies that are doing nothing useful.

### 1.5 Instruction set

```haskell
data Instruction
  = GetConstant Value !RegId
  | GetVariable !RegId !RegId
  | GetValue !RegId !RegId
  | PutConstant Value !RegId
  | PutVariable !RegId !RegId
  | PutValue !RegId !RegId
  | PutStructure String !RegId !Int    -- functor, target, arity (pre-parsed)
  | PutList !RegId
  | SetValue !RegId
  | SetConstant Value
  | Allocate
  | Deallocate
  | Call String !Int                   -- pre-resolution form
  | CallResolved !Int !Int             -- post-resolution: target PC + arity
  | Execute String
  | Proceed
  | BuiltinCall String Int
  | TryMeElse String
  | RetryMeElse String
  | TrustMe
  | SwitchOnConstant (Map.Map Value String)
  | BeginAggregate String !RegId !RegId
  | EndAggregate !RegId
  deriving (Show, Eq)
```

`Call` survives in two cases:
- Foreign predicates (e.g., `category_ancestor/4` → `executeForeign`) need
  the runtime dispatch.
- Indexed facts (`category_parent/2` → `callIndexedFact2`) need the
  runtime dispatch.

Everything else is converted to `CallResolved` once, at startup, by
`resolveCallInstrs`.

## 2. Planned: WamState hot/cold split

### 2.1 Goal

Eliminate the per-step allocation of a 17-field record by separating fields
that change every step from fields that never change after initialization.

### 2.2 New shapes

```haskell
-- | Read-only after initialization. Threaded as a function argument.
data WamContext = WamContext
  { wcCode          :: !(Array Int Instruction)
  , wcLabels        :: !(Map.Map String Int)
  , wcForeignFacts  :: !(Map.Map String (Map.Map String [String]))
  , wcForeignConfig :: !(Map.Map String Int)
  } deriving (Show)

-- | Mutates per step. The thing we actually allocate on each transition.
data WamState = WamState
  { wsPC         :: !Int
  , wsRegs       :: !(IM.IntMap Value)
  , wsStack      :: ![EnvFrame]
  , wsHeap       :: ![Value]
  , wsHeapLen    :: !Int
  , wsTrail      :: ![TrailEntry]
  , wsTrailLen   :: !Int
  , wsCP         :: !Int
  , wsCPs        :: ![ChoicePoint]
  , wsCPsLen     :: !Int
  , wsBindings   :: !(IM.IntMap Value)
  , wsCutBar     :: !Int
  , wsBuilder    :: !Builder
  , wsVarCounter :: !Int
  , wsAggAccum   :: ![Value]
  } deriving (Show)
```

`WamState` drops from 17 fields to 15. The two cold map fields
(`wsForeignFacts`, `wsForeignConfig`) and the two read-only structural
fields (`wsCode`, `wsLabels`) move into `WamContext`.

### 2.3 Step function shape change

Before:
```haskell
step :: WamState -> Maybe WamState
step s = case unsafeFetchInstr (wsPC s) (wsCode s) of
  GetConstant c rid -> ...
  ...
```

After:
```haskell
step :: WamContext -> WamState -> Maybe WamState
step !ctx !s = case unsafeFetchInstr (wsPC s) (wcCode ctx) of
  GetConstant c rid -> ...
  ...
```

The `!ctx !s` bang patterns are required: GHC must not be lazy about
either argument or the worker/wrapper transform won't unbox the state.

### 2.4 Run loop change

Before:
```haskell
runLoop :: WamState -> Maybe WamState
runLoop s
  | wsPC s == 0 = Just s
  | otherwise   = case step s of
      Nothing -> backtrack s >>= runLoop
      Just s' -> runLoop s'
```

After:
```haskell
runLoop :: WamContext -> WamState -> Maybe WamState
runLoop !ctx !s
  | wsPC s == 0 = Just s
  | otherwise   = case step ctx s of
      Nothing -> backtrack s >>= runLoop ctx
      Just s' -> runLoop ctx s'
```

`backtrack` does *not* need the context — backtracking only touches the
mutable state — so its signature is unchanged.

### 2.5 Things that need to be edited (rough scope)

The post-processing pass `apply_hashmap_rewrite` in `wam_haskell_target.pl`
emits roughly:

- `WamRuntime.hs` — `step`, `runLoop`, `backtrack`, helpers
- `WamTypes.hs` — `WamState`, the new `WamContext`, `Instruction`
- `Predicates.hs` — generated per-predicate code (does not touch state shape)
- `Main.hs` — benchmark driver, builds the initial state

Touchpoints inside `WamRuntime.hs`:
- Every `case ... of` branch in `step` that currently uses `wsCode s`,
  `wsLabels s`, `wsForeignFacts s`, or `wsForeignConfig s` must read from
  `ctx` instead.
- `executeForeign` and `callIndexedFact2` need the context.
- `nativeCategoryAncestor` already takes its facts as an argument; the
  call site needs to read from `ctx` instead of `s`.

Touchpoints inside `Main.hs`:
- The initial state setup splits into `mkContext :: ... -> WamContext`
  and `mkState :: ... -> WamState`.
- The query loop calls `runLoop ctx state0`.

Estimated diff: ~150 lines of generator code, ~5 generated files affected.

### 2.6 Expected wins

The benchmark currently runs ~3.0–4.5 s wall time for 11172 paths at
depth=10. Per-step record allocation is plausibly 15–25% of that. A win
of "drop 0.5–1.0 s" is the realistic expectation.

The bigger win is that *with the cold fields out of the way*, the next
profile pass can see the FFI overhead clearly.

### 2.7 What can go wrong

- **Lazy `ctx`**: if the generator forgets `!ctx`, GHC will allocate
  thunks for `ctx` reads inside `step`. The fix is bang patterns, but
  the bug is silent — the program is correct, just no faster.
- **Forgetting to update `Predicates.hs`**: the generated predicates are
  WAM instructions and don't reference the state shape, so this should
  be a no-op for them. Verify with `cabal build` before running benchmarks.
- **Splitting too aggressively**: if any field that *does* change per step
  gets put into the context, that field can't be updated and the
  simulation is wrong. The safe rule is: if a step function ever writes
  to it, it stays in `WamState`.

## 3. Planned: FFI optimization

### 3.1 Current FFI shape

```haskell
nativeCategoryAncestor
  :: Map.Map String [String]
  -> String   -- starting category
  -> String   -- root we're searching toward
  -> Int      -- max depth
  -> Int      -- current depth
  -> [String] -- visited (cycle detection)
  -> [Int]    -- list of Hops counts where we hit the root
nativeCategoryAncestor parents cat root maxDepth depth visited =
  let directParents = fromMaybe [] (Map.lookup cat parents)
      baseHits = [1 | p <- directParents, p == root, p `notElem` visited]
      recHits  = if depth >= maxDepth then [] else
        concatMap (\mid ->
          if mid `elem` visited then []
          else map (+1) $
            nativeCategoryAncestor parents mid root maxDepth (depth+1) (mid : visited)
        ) directParents
  in baseHits ++ recHits
```

The result is consumed by `executeForeign`, which sets up a `HopsRetry`
choice point so the WAM enumerates each Hops value via normal backtracking.

### 3.2 Issues found in this shape

1. **`baseHits ++ recHits` walks both lists.** Not a big deal at small
   sizes but it does cost allocation. A difference list (`[Int] -> [Int]`)
   or `Data.DList` would be cheaper.

2. **`p `notElem` visited` is redundant for `baseHits`.** The visited
   check has already been done by the caller before recursing, so the
   immediate parents being checked here can never be in `visited` *as
   the variable they were called for*. (They can still be ancestors of
   the original starting category, which is the case the check actually
   handles. Need to verify before removing.)

3. **Building `[Int]` and then handing it to `HopsRetry`** means each
   Hops result pays for one full WAM step round-trip back into dispatch.
   For "give me all paths to the root" workloads where the typical
   answer is 5–50 Hops counts, the dispatch cost of converting back is
   non-trivial.

4. **`map (+1)` allocates a fresh list per recursive call.** For deep
   recursion this adds up. Could be inlined into the `concatMap`.

5. **`directParents` is recomputed once per recursive call.** Already
   minimal cost (it's an `IntMap`-style lookup), but could be hoisted
   if profiling justifies it.

### 3.3 Planned changes

Not yet committed. Sketch:

```haskell
-- | Returns a difference list of Hops, avoiding (++) and intermediate lists.
nativeCategoryAncestorDL
  :: Map.Map String [String]
  -> String -> String -> Int -> Int -> [String]
  -> ([Int] -> [Int])
nativeCategoryAncestorDL parents cat root maxDepth depth visited =
  let directParents = fromMaybe [] (Map.lookup cat parents)
      base = if any (== root) directParents then (1 :)
             else id
      rec  = if depth >= maxDepth then id else
               foldr (\mid acc ->
                 if mid `elem` visited then acc
                 else
                   let inner = nativeCategoryAncestorDL parents mid root maxDepth (depth+1) (mid : visited)
                       bumped = map (+1) (inner [])  -- forced once, intentionally
                   in (bumped ++) . acc
               ) id directParents
  in base . rec

-- Top-level wrapper:
nativeCategoryAncestor :: Map.Map String [String]
                      -> String -> String -> Int
                      -> [Int]
nativeCategoryAncestor p c r d = nativeCategoryAncestorDL p c r d 0 [c] []
```

The structural change is small. The semantic change is zero. The win is
that fewer intermediate lists get built.

### 3.4 Bigger FFI question: do we need the WAM round-trip at all?

The current path is:
```
WAM step → executeForeign → nativeCategoryAncestor → [Int]
                       ↓
                  HopsRetry CP → step → ... (WAM resumes)
```

Each `HopsRetry` consumes one element of the `[Int]` and resumes the
WAM. For `aggregate_all(sum(W), category_ancestor(...), Sum)` the WAM is
just summing the Hops values, which we could do entirely inside the
foreign call:

```haskell
nativeCategoryAncestorSum :: Map.Map String [String] -> String -> String -> Int -> Int -> Int
nativeCategoryAncestorSum parents cat root maxDepth negN =
  sum [ (h + 1) ^^ negN | h <- nativeCategoryAncestor parents cat root maxDepth ]
```

This would skip the WAM round-trip entirely for the aggregate case. The
risk is that it commits us to a specific aggregate shape (sum of
power-of-Hops) that the user happens to want for the effective-distance
benchmark. Generalizing it later means redoing the work.

**Decision (deferred):** Do the difference-list cleanup first. If profile
shows the WAM round-trip is still a bottleneck, evaluate the
direct-aggregate approach as a separate change.

## 4. Deferred: compile WAM-to-Haskell-functions

This is a third path, separate from both native lowering and the current
WAM-as-interpreter approach.

### 4.1 Concept

Take a WAM instruction sequence for a single predicate and emit a
Haskell function that does the same thing without any instruction
dispatch:

```
% Input: WAM for category_ancestor/4
0: try_me_else 5
1: get_constant cat1, A1
2: get_constant root1, A2
3: put_constant 0, A3
4: proceed
5: trust_me
6: ...
```

becomes

```haskell
categoryAncestor :: WamContext -> RegId -> RegId -> RegId -> RegId -> [WamState -> WamState]
categoryAncestor ctx a1 a2 a3 a4 = ...
```

where the function body is straight-line Haskell that mirrors the
instruction sequence. No `case` on instructions, no `IntMap` lookups for
register reads (the registers become Haskell `let`-bindings), no
program counter at all for the inside of one predicate.

### 4.2 What it would buy

- Predicate dispatch (the `Call` / `CallResolved` instruction) becomes
  a Haskell function call. GHC inlines it where profitable.
- Per-predicate register reads become local lets, which GHC tracks as
  unboxed values rather than heap allocations.
- Loop fusion across instructions becomes possible because GHC sees
  the whole predicate at once.

### 4.3 What it would cost

- A new code generator path. Different output shape, different bug
  surface, different test surface.
- Backtracking needs to work *across* compiled predicates, which means
  the choice point shape from the current WAM has to flow through the
  compiled functions. Probably the compiled functions still take and
  return `WamState`, just internally they don't dispatch.
- The compile-time analysis of "which fields does this predicate read
  vs write" is not free.

### 4.4 Why it is deferred

The current branch's goal is to make the WAM-as-interpreter path
competitive. The compile-WAM-to-Haskell path is a different shape with a
different design and a different test plan. It deserves its own design
docs and its own branch.

The expected order:
1. Land the perf branch (current)
2. Land the WamState split (next)
3. Land the FFI optimization (after the split)
4. Open the compile-WAM-to-Haskell discussion separately

## 5. Out of scope for this branch

These are explicitly *not* in the spec for this work:

- Switching to `IORef`/`STRef`/mutable arrays (see philosophy doc).
- Adding `unsafePerformIO` to the step function.
- A new instruction set or opcode renumbering.
- Changes to native lowering (`haskell_target.pl`).
- Changes to the Rust WAM target.
- A general parser for runtime functor parsing (`PutStructureDyn` —
  see project task #16).
- Any change to the test or benchmark harness beyond what is needed
  to verify the optimizations don't regress correctness.
