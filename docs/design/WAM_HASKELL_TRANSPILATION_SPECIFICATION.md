# WAM Haskell Transpilation: Specification

## Data Types

### Value
```haskell
data Value = Atom String
           | Integer Int
           | Float Double
           | VList [Value]
           | Str String [Value]    -- functor + args (e.g., Str "member/2" [x, list])
           | Unbound String        -- logical variable
           | Ref Int               -- heap reference
           deriving (Eq, Ord, Show)
```

### WamState
```haskell
data WamState = WamState
  { wsPC       :: !Int                         -- program counter (1-indexed, 0=halt)
  , wsRegs     :: !(Map.Map String Value)      -- A/X registers
  , wsStack    :: ![EnvFrame]                  -- environment frames
  , wsHeap     :: ![Value]                     -- term construction area
  , wsTrail    :: ![TrailEntry]                -- binding history
  , wsCP       :: !Int                         -- continuation pointer
  , wsCPs      :: ![ChoicePoint]               -- choice point stack
  , wsBindings :: !(Map.Map String Value)      -- variable binding table
  , wsCutBar   :: !Int                         -- cut barrier depth
  , wsLabels   :: !(Map.Map String Int)        -- label → PC mapping
  }
```

### ChoicePoint
```haskell
data ChoicePoint = ChoicePoint
  { cpNextPC   :: !Int
  , cpRegs     :: !(Map.Map String Value)      -- O(1) snapshot via sharing
  , cpStack    :: ![EnvFrame]                  -- O(1) snapshot via sharing
  , cpCP       :: !Int
  , cpTrailLen :: !Int                         -- index-based (O(1))
  , cpHeapLen  :: !Int                         -- index-based (O(1))
  , cpBindings :: !(Map.Map String Value)      -- O(1) snapshot via sharing
  , cpCutBar   :: !Int
  }
```

## WAM Instruction Set

Each instruction is a pure function: `WamState -> Maybe WamState`

### Head Unification
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `get_constant C, Ai` | Unify `regs[Ai]` with constant `C`. Bind if unbound. |
| `get_variable Xn, Ai` | Copy `regs[Ai]` to register `Xn`. |
| `get_value Xn, Ai` | Unify `regs[Ai]` with `regs[Xn]`. |
| `get_structure F/N, Ai` | Match compound term or enter write mode. |
| `get_list Ai` | Match list `[H|T]` or enter write mode. |

### Body Construction
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `put_constant C, Ai` | Set `regs[Ai] = C`. |
| `put_variable Xn, Ai` | Create fresh `Unbound`, store in both `Xn` and `Ai`. |
| `put_value Xn, Ai` | Copy `regs[Xn]` to `regs[Ai]`. |
| `put_structure F/N, Ai` | Begin constructing compound term in `Ai`. |
| `put_list Ai` | Begin constructing list in `Ai`. |
| `set_value Xn` | Add `regs[Xn]` as next sub-argument. |
| `set_constant C` | Add constant `C` as next sub-argument. |

### Control Flow
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `allocate` | Push env frame. Set `cutBarrier = length cps`. |
| `deallocate` | Pop env frame. Restore `CP`. |
| `call P/N` | Save `CP = PC + 1`. Jump to label `P/N`. |
| `execute P/N` | Jump to label `P/N` (tail call). |
| `proceed` | Return to `CP`. If `CP = 0`, halt. |

### Choice Points
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `try_me_else L` | Push `ChoicePoint` with `nextPC = L`. **O(1)** via sharing. |
| `retry_me_else L` | Update top CP's `nextPC = L`. (Do NOT re-snapshot state.) |
| `trust_me` | Pop top `ChoicePoint`. |

### Indexing
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `switch_on_constant` | Binary search dispatch table on `regs[A1]`. |

### Builtins
| Instruction | Haskell Semantics |
|-------------|-------------------|
| `!/0` | Truncate `wsCPs` to `wsCutBar`. |
| `is/2` | Evaluate arithmetic, bind result. |
| `length/2` | Measure list, bind length. |
| `</2`, `>/2`, etc. | Arithmetic comparison. |
| `\+/1` | Negation-as-failure. Fast path for `member/2`. |

## Backtracking Semantics

```haskell
backtrack :: WamState -> Maybe WamState
backtrack s = case wsCPs s of
  [] -> Nothing     -- no choice points: fail
  (cp : _) ->
    -- 1. Unwind binding-only trail entries
    -- 2. Swap in saved state (O(1) for Map fields)
    -- 3. Truncate trail and heap to saved lengths
    Just s { wsPC = cpNextPC cp
           , wsRegs = cpRegs cp           -- O(1)
           , wsStack = cpStack cp         -- O(1)
           , wsBindings = cpBindings cp   -- O(1)
           , ...
           }
```

**Critical difference from Rust:** Steps 1-3 are O(1) + O(k) where k is the
number of trail entries to unwind. In Rust, step 2 was O(n) (HashMap clone).

## Fact Indexing

Facts are loaded as `Map.Map String [(String, ...)]` lookup tables, indexed
by first argument. This replaces the Rust spike's `SwitchOnConstant` +
`TryMeElse` chain with a native Haskell `Map.lookup`.

```haskell
-- category_parent indexed by first argument
categoryParents :: Map.Map String [String]
categoryParents = Map.fromListWith (++) [("Nuclear_physics", ["Subfields_of_physics"]), ...]

-- Query: category_parent(Cat, Parent)
categoryParent :: String -> [(String)]
categoryParent cat = fromMaybe [] $ Map.lookup cat categoryParents
```

## Integration with Native Lowering

The `haskell_target.pl` handles native lowering. When it fails, the WAM
fallback path is:

```prolog
compile_predicate(Pred/Arity, Options, Code) :-
    % Try native lowering first
    (   haskell_target:compile_predicate_to_haskell(Pred/Arity, Options, Code)
    ->  true
    ;   % Fallback to WAM
        wam_target:compile_predicate_to_wam(Pred/Arity, [], WamCode),
        wam_haskell_target:compile_wam_predicate_to_haskell(
            Pred/Arity, WamCode, Options, Code)
    ).
```
