# Haskell WAM Mutable Registers: Specification

**Status**: Design complete, POC validated. Implementation pending.
**Date**: 2026-05-25
**Companion**: `WAM_HASKELL_MUTABLE_REGISTERS_PHILOSOPHY.md`
**POC**: `examples/benchmark/haskell_st_wam_proto/STWam.hs` (7.66x speedup)

## 1. Type changes

### WamState (unchanged externally)

`WamState` keeps `wsRegs :: IM.IntMap Value` for serialization and
initialization. The mutable array is separate, passed alongside the
state inside `runST`.

### New: PureWamState (all fields except regs)

```haskell
-- All WAM state except registers. Threaded as a pure record.
-- Register access goes through the separate STArray.
data PureWamState = PureWamState
  { pwPC        :: {-# UNPACK #-} !Int
  , pwStack     :: ![EnvFrame]
  , pwHeap      :: ![Value]
  , pwHeapLen   :: {-# UNPACK #-} !Int
  , pwTrail     :: ![TrailEntry]
  , pwTrailLen  :: {-# UNPACK #-} !Int
  , pwCP        :: {-# UNPACK #-} !Int
  , pwCPs       :: ![MutableChoicePoint]
  , pwCPsLen    :: {-# UNPACK #-} !Int
  , pwBindings  :: !(IM.IntMap Value)
  , pwCutBar    :: {-# UNPACK #-} !Int
  , pwBuilder   :: !Builder
  , pwVarCounter :: {-# UNPACK #-} !Int
  , pwAggAccum  :: ![Value]
  }
```

### MutableChoicePoint (frozen register snapshot)

```haskell
data MutableChoicePoint = MutableChoicePoint
  { mcpNextPC    :: {-# UNPACK #-} !Int
  , mcpRegs      :: !(Array Int Value)    -- FROZEN snapshot
  , mcpStack     :: ![EnvFrame]
  , mcpCP        :: {-# UNPACK #-} !Int
  , mcpTrailLen  :: {-# UNPACK #-} !Int
  , mcpHeapLen   :: {-# UNPACK #-} !Int
  , mcpBindings  :: !(IM.IntMap Value)
  , mcpCutBar    :: {-# UNPACK #-} !Int
  , mcpAggFrame  :: !(Maybe AggFrame)
  , mcpBuiltin   :: !(Maybe BuiltinState)
  }
```

## 2. Function signatures

### run (outer interface stays pure)

```haskell
run :: WamContext -> WamState -> Maybe WamState
run !ctx !s0 = runST $ do
    regs <- newArray (1, maxReg) (Unbound (-1))
    forM_ (IM.toList (wsRegs s0)) $ \(k,v) -> writeArray regs k v
    let pw0 = wamStateToPure s0
    result <- runLoop ctx regs pw0
    case result of
      Nothing -> return Nothing
      Just pw -> do
        finalArr <- freeze regs
        return (Just (pureToWamState pw finalArr))
```

### runLoop (internal, in ST)

```haskell
runLoop :: WamContext -> STArray s Int Value -> PureWamState
        -> ST s (Maybe PureWamState)
runLoop !ctx regs !pw
  | pwPC pw == 0 = return (Just pw)
  | otherwise = do
      let !instr = unsafeFetchInstr (pwPC pw) (wcCode ctx)
      result <- stepST ctx regs pw instr
      case result of
        Just !pw' -> runLoop ctx regs pw'
        Nothing -> do
          bt <- backtrackST regs pw
          case bt of
            Just !pw' -> runLoop ctx regs pw'
            Nothing -> return Nothing
```

### stepST (internal, in ST)

```haskell
stepST :: WamContext -> STArray s Int Value -> PureWamState
       -> Instruction -> ST s (Maybe PureWamState)
```

### backtrackST (internal, in ST)

```haskell
backtrackST :: STArray s Int Value -> PureWamState
            -> ST s (Maybe PureWamState)
```

## 3. Transformation patterns

### Pattern A: Register read with deref

Before:
```haskell
let val = derefVar (wsBindings s) <$> IM.lookup ai (wsRegs s)
```

After:
```haskell
v <- readArray regs ai
let val = Just (derefVar (pwBindings pw) v)
```

Note: `readArray` always succeeds (array is initialized to `Unbound (-1)`),
so `<$>` over Maybe becomes direct application.

### Pattern B: Register read in case

Before:
```haskell
case IM.lookup ai (wsRegs s) of
  Just val -> ...
  Nothing -> Nothing
```

After:
```haskell
val <- readArray regs ai
-- (proceed with val directly; Nothing case eliminated)
```

### Pattern C: Simple register write + record update

Before:
```haskell
Just (s { wsPC = wsPC s + 1, wsRegs = IM.insert ai c (wsRegs s) })
```

After:
```haskell
writeArray regs ai c
return $ Just (pw { pwPC = pwPC pw + 1 })
```

### Pattern D: Multiple register writes

Before:
```haskell
wsRegs = IM.insert xn var (IM.insert ai var (wsRegs s))
```

After:
```haskell
writeArray regs xn var
writeArray regs ai var
```

### Pattern E: Choice point snapshot

Before:
```haskell
cpRegs = wsRegs s    -- O(1) structural sharing
```

After:
```haskell
snap <- freeze regs  -- O(n) copy, n=200
... mcpRegs = snap ...
```

### Pattern F: Backtrack register restore

Before:
```haskell
wsRegs = cpRegs cp   -- O(1) pointer swap
```

After:
```haskell
let snap = mcpRegs mcp
forM_ [1..maxReg] $ \i -> writeArray regs i (snap ! i)
```

### Pattern G: getReg helper

Before:
```haskell
getReg :: Int -> WamState -> Maybe Value
getReg rid s
  | rid >= 200 = findYReg rid (wsStack s)
  | otherwise = derefVar (wsBindings s) <$> IM.lookup rid (wsRegs s)
```

After:
```haskell
getRegST :: STArray s Int Value -> Int -> PureWamState -> ST s (Maybe Value)
getRegST regs rid pw
  | rid >= 200 = return $ findYReg rid (pwStack pw)
  | otherwise = do
      v <- readArray regs rid
      return $ Just (derefVar (pwBindings pw) v)
```

### Pattern H: putReg helper

Before:
```haskell
putReg :: Int -> Value -> WamState -> WamState
putReg rid val s
  | rid >= 200 = s { wsStack = updateTopEnv rid val (wsStack s) }
  | otherwise = s { wsRegs = IM.insert rid val (wsRegs s) }
```

After:
```haskell
putRegST :: STArray s Int Value -> Int -> Value -> PureWamState -> ST s PureWamState
putRegST regs rid val pw
  | rid >= 200 = return $ pw { pwStack = updateTopEnv rid val (pwStack pw) }
  | otherwise = do writeArray regs rid val; return pw
```

## 4. Conversion scope

| Section | Lines | wsRegs refs | Priority |
|---------|-------|-------------|----------|
| getReg/putReg | 3614-3634 | 4 | 1 (foundation) |
| run loop | 2549-2586 | 0 | 2 (wrapper) |
| backtrack | 1175-1356 | 7 | 3 (restore) |
| step function | 1839-2547 | 70 | 4 (bulk) |
| addToBuilder | 3646-3678 | 3 | 5 (helper) |
| lowered emitter | separate file | 4 | 6 (last) |
| foreign call dispatch | 513-830 | ~12 | 7 (codegen) |

Total: ~100 sites. Estimated 2-3 sessions for full conversion.

## 5. Migration strategy

1. Add `PureWamState`, `MutableChoicePoint` types to WamTypes generation
2. Add `getRegST`, `putRegST`, `wamStateToPure`, `pureToWamState` helpers
3. Add `runLoop` and `backtrackST` in ST
4. Convert step arms one group at a time (grouped by pattern)
5. Add codegen option `register_mode(st_array)` to switch between old/new
6. Default to `st_array` once all tests pass
7. Remove old IntMap code path after stabilization

## 6. References

- Philosophy doc: `docs/design/WAM_HASKELL_MUTABLE_REGISTERS_PHILOSOPHY.md`
- POC standalone: `examples/benchmark/haskell_register_bench/Main.hs` (7.9x)
- POC WAM proto: `examples/benchmark/haskell_st_wam_proto/STWam.hs` (7.66x)
- Plan: `docs/design/WAM_FSHARP_CSR_PARALLEL_PLAN.md` Phase 3
- Code generator: `src/unifyweaver/targets/wam_haskell_target.pl` lines 1175-3700
