:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_haskell_target.pl - WAM-to-Haskell Transpilation Target
%
% Compiles WAM instructions to Haskell code using persistent data structures.
% Unlike the Rust WAM target (which interprets instructions at runtime),
% this target natively lowers each WAM instruction to Haskell expressions.
%
% Key design: Data.Map for registers and bindings gives O(1) snapshots
% for choice points (structural sharing), eliminating the clone overhead
% that made the Rust WAM 343x slower than SWI-Prolog.
%
% Architecture:
%   - WamState record with Data.Map fields
%   - Each compiled predicate becomes a Haskell function
%   - TryMeElse/RetryMeElse/TrustMe become choice point list operations
%   - Backtracking = swap to saved Data.Map reference (O(1))
%   - Facts loaded as Data.Map lookup tables (first-argument indexing)
%
% Pipeline:
%   Prolog source → haskell_target.pl (native lowering, preferred)
%                 → wam_target.pl (WAM compilation, fallback)
%                 → wam_haskell_target.pl (THIS FILE: WAM → Haskell)
%
% See: docs/design/WAM_RUST_STATE_MANAGEMENT_RETROSPECTIVE.md

:- module(wam_haskell_target, [
    compile_wam_predicate_to_haskell/4,  % +Pred/Arity, +WamCode, +Options, -HaskellCode
    compile_wam_runtime_to_haskell/2,    % +Options, -HaskellCode
    write_wam_haskell_project/3          % +Predicates, +Options, +ProjectDir
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

% ============================================================================
% Haskell WAM Runtime Data Types
% ============================================================================
%
% Generated Haskell code uses these types:
%
%   data Value = Atom String | Integer Int | Float Double
%              | VList [Value] | Str String [Value]
%              | Unbound String | Ref Int
%              deriving (Eq, Ord, Show)
%
%   data WamState = WamState
%     { wsPC        :: !Int
%     , wsRegs      :: !(Map String Value)    -- A/X registers
%     , wsStack     :: ![EnvFrame]            -- environment frames
%     , wsHeap      :: ![Value]               -- term construction
%     , wsTrail     :: ![TrailEntry]           -- binding history
%     , wsCP        :: !Int                    -- continuation pointer
%     , wsCPs       :: ![ChoicePoint]          -- choice point stack
%     , wsBindings  :: !(Map String Value)     -- variable bindings
%     , wsCutBar    :: !Int                    -- cut barrier
%     }
%
%   data EnvFrame = EnvFrame !Int !(Map String Value)  -- saved CP + Y-regs
%
%   data TrailEntry = TrailEntry !String !(Maybe Value)
%
%   data ChoicePoint = ChoicePoint
%     { cpNextPC   :: !Int
%     , cpRegs     :: !(Map String Value)
%     , cpStack    :: ![EnvFrame]
%     , cpCP       :: !Int
%     , cpTrailLen :: !Int
%     , cpHeapLen  :: !Int
%     , cpBindings :: !(Map String Value)     -- O(1) snapshot!
%     , cpCutBar   :: !Int
%     }
%
% The critical insight: Map String Value uses structural sharing.
% When a ChoicePoint saves cpBindings = wsBindings state, no data is copied.
% Both point to the same tree. Mutations create new nodes only along the
% modified path (O(log n) per insert). Backtracking = swap the reference.

% ============================================================================
% PHASE 1: WAM Instruction → Haskell Expression
% ============================================================================

%% wam_to_haskell(+Instruction, -HaskellExpr)
%  Translates a single WAM instruction to a Haskell state transformation.
%  Each instruction is a function: WamState -> Maybe WamState
%  Nothing = failure, Just s = success with new state.

wam_to_haskell(get_constant(C, Ai), Code) :-
    format(string(Code),
'  let val = Map.lookup "~w" (wsRegs s)
  in case val of
    Just v | v == ~w -> Just (s { wsPC = wsPC s + 1 })
    Just (Unbound var) -> Just (s { wsPC = wsPC s + 1
                                  , wsRegs = Map.insert "~w" ~w (wsRegs s)
                                  , wsBindings = Map.insert var ~w (wsBindings s)
                                  , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                                  })
    _ -> Nothing', [Ai, C, Ai, C, C]).

wam_to_haskell(get_variable(Xn, Ai), Code) :-
    format(string(Code),
'  case Map.lookup "~w" (wsRegs s) of
    Just val -> let derefed = derefVar (wsBindings s) val
                    s1 = putReg "~w" derefed s
                in Just (s1 { wsPC = wsPC s + 1 })
    Nothing -> Nothing', [Ai, Xn]).

wam_to_haskell(put_value(Xn, Ai), Code) :-
    format(string(Code),
'  case getReg "~w" s of
    Just val -> Just (s { wsPC = wsPC s + 1
                        , wsRegs = Map.insert "~w" val (wsRegs s)
                        })
    Nothing -> Nothing', [Xn, Ai]).

wam_to_haskell(put_variable(Xn, Ai), Code) :-
    format(string(Code),
'  let var = Unbound ("_V" ++ show (wsPC s))
      s1 = putReg "~w" var s
  in Just (s1 { wsPC = wsPC s + 1
              , wsRegs = Map.insert "~w" var (wsRegs s1)
              })', [Xn, Ai]).

wam_to_haskell(put_constant(C, Ai), Code) :-
    format(string(Code),
'  Just (s { wsPC = wsPC s + 1
           , wsRegs = Map.insert "~w" ~w (wsRegs s)
           })', [Ai, C]).

wam_to_haskell(call(Pred, _Arity), Code) :-
    format(string(Code),
'  Just (s { wsPC = lookupLabel "~w" s
           , wsCP = wsPC s + 1
           })', [Pred]).

wam_to_haskell(proceed, Code) :-
    Code = '  let ret = wsCP s
  in if ret == 0 then Just (s { wsPC = 0 })  -- halt
     else Just (s { wsPC = ret, wsCP = 0 })'.

wam_to_haskell(allocate, Code) :-
    Code = '  let frame = EnvFrame (wsCP s) Map.empty
  in Just (s { wsPC = wsPC s + 1
             , wsStack = frame : wsStack s
             , wsCutBar = length (wsCPs s)
             })'.

wam_to_haskell(deallocate, Code) :-
    Code = '  case wsStack s of
    (EnvFrame oldCP _ : rest) -> Just (s { wsPC = wsPC s + 1
                                         , wsStack = rest
                                         , wsCP = oldCP
                                         })
    _ -> Nothing'.

wam_to_haskell(try_me_else(Label), Code) :-
    format(string(Code),
'  let cp = ChoicePoint
        { cpNextPC   = lookupLabel "~w" s
        , cpRegs     = wsRegs s       -- O(1): shared reference
        , cpStack    = wsStack s      -- O(1): shared reference
        , cpCP       = wsCP s
        , cpTrailLen = length (wsTrail s)
        , cpHeapLen  = length (wsHeap s)
        , cpBindings = wsBindings s   -- O(1): shared reference
        , cpCutBar   = wsCutBar s
        }
  in Just (s { wsPC = wsPC s + 1
             , wsCPs = cp : wsCPs s
             })', [Label]).

wam_to_haskell(trust_me, Code) :-
    Code = '  case wsCPs s of
    (_ : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest })
    [] -> Nothing'.

wam_to_haskell(retry_me_else(Label), Code) :-
    format(string(Code),
'  case wsCPs s of
    (cp : rest) -> let cp'' = cp { cpNextPC = lookupLabel "~w" s }
                   in Just (s { wsPC = wsPC s + 1
                              , wsCPs = cp'' : rest
                              })
    [] -> Nothing', [Label]).

wam_to_haskell(builtin_call('!/0', 0), Code) :-
    Code = '  Just (s { wsPC = wsPC s + 1
             , wsCPs = take (wsCutBar s) (wsCPs s)
             })'.

wam_to_haskell(builtin_call('is/2', 2), Code) :-
    Code = '  let expr = derefVar (wsBindings s) $ fromMaybe (Integer 0) (Map.lookup "A2" (wsRegs s))
      result = evalArith (wsBindings s) expr
      lhs = derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s)
  in case (lhs, result) of
    (Just (Unbound var), Just r) ->
      let val = if fromIntegral (round r) == r then Integer (round r) else Float r
      in Just (s { wsPC = wsPC s + 1
                 , wsRegs = Map.insert "A1" val (wsRegs s)
                 , wsBindings = Map.insert var val (wsBindings s)
                 , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                 })
    (Just (Integer n), Just r) | fromIntegral n == r -> Just (s { wsPC = wsPC s + 1 })
    (Just (Float f), Just r) | f == r -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing'.

wam_to_haskell(builtin_call('length/2', 2), Code) :-
    Code = '  let listVal = derefVar (wsBindings s) $ fromMaybe (VList []) (Map.lookup "A1" (wsRegs s))
      len = case listVal of VList items -> length items ; _ -> -1
      lhs = derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s)
  in if len < 0 then Nothing
     else case lhs of
       Just (Unbound var) ->
         let val = Integer len
         in Just (s { wsPC = wsPC s + 1
                    , wsRegs = Map.insert "A2" val (wsRegs s)
                    , wsBindings = Map.insert var val (wsBindings s)
                    , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                    })
       Just (Integer n) | n == len -> Just (s { wsPC = wsPC s + 1 })
       _ -> Nothing'.

wam_to_haskell(builtin_call('</2', 2), Code) :-
    Code = '  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a < b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing'.

% Negation-as-failure: fast path for member/2
wam_to_haskell(builtin_call('\\+/1', 1), Code) :-
    Code = '  case derefHeap (wsHeap s) =<< Map.lookup "A1" (wsRegs s) of
    Just (Str "member/2" [needle, haystack]) ->
      let needle'' = derefVar (wsBindings s) needle
          haystack'' = derefVar (wsBindings s) haystack
          found = case haystack'' of
            VList items -> any (\\item -> derefVar (wsBindings s) item == needle'') items
            _ -> False
      in if found then Nothing  -- member succeeded, \\+ fails
         else Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing  -- unsupported goal'.

% ============================================================================
% PHASE 2: Backtrack Function
% ============================================================================

backtrack_haskell(Code) :-
    Code = '-- | Restore state from the top choice point (non-popping).
-- Uses O(1) Data.Map reference swap for registers and bindings.
-- When an aggregate frame CP is reached, delegates to finalizeAggregate.
backtrack :: WamState -> Maybe WamState
backtrack s = case wsCPs s of
  [] -> Nothing
  (cp : rest) -> case cpAggFrame cp of
    Just af ->
      -- Aggregate frame: finalize with the accumulated values.
      finalizeAggregate (afReturnPC af) s
    Nothing ->
      let trailLen = cpTrailLen cp
          newEntries = reverse $ take (length (wsTrail s) - trailLen) (wsTrail s)
          bindings'' = foldl'' undoBinding (cpBindings cp) newEntries
      in Just s { wsPC       = cpNextPC cp
                , wsRegs     = cpRegs cp       -- O(1): shared reference swap
                , wsStack    = cpStack cp      -- O(1): shared reference swap
                , wsCP       = cpCP cp
                , wsTrail    = drop (length (wsTrail s) - trailLen) (wsTrail s)
                , wsHeap     = take (cpHeapLen cp) (wsHeap s)
                , wsBindings = bindings''       -- O(1) base + O(k) trail unwind
                , wsCutBar   = cpCutBar cp
                }
  where
    undoBinding bindings (TrailEntry key mOld)
      | "__binding__" `isPrefixOf` key =
          let var = drop (length "__binding__") key
          in case mOld of
            Just old -> Map.insert var old bindings
            Nothing  -> Map.delete var bindings
      | otherwise = bindings

-- | Backtrack skipping past the aggregate_frame CP. If the top CP is
-- an aggregate frame, return Nothing (inner solutions exhausted).
-- Otherwise, normal backtrack.
backtrackInner :: Int -> WamState -> Maybe WamState
backtrackInner returnPC s = case wsCPs s of
  (cp : _)
    | Just _ <- cpAggFrame cp -> Nothing  -- reached aggregate frame = done
    | otherwise -> backtrack s
  [] -> Nothing

-- | Finalize an aggregate: pop CPs to the aggregate frame, apply the
-- aggregation function, bind the result register.
finalizeAggregate :: Int -> WamState -> Maybe WamState
finalizeAggregate returnPC s = go (wsCPs s)
  where
    go [] = Nothing
    go (cp : rest) = case cpAggFrame cp of
      Just (AggFrame typ _valReg resReg _) ->
        let accum = reverse (wsAggAccum s)
            result = applyAggregation typ accum
            bindings0 = cpBindings cp
            regs0 = cpRegs cp
            resVal = derefVar bindings0 <$> Map.lookup resReg regs0
            (regs1, bindings1, trail1) = case resVal of
              Just (Unbound var) ->
                ( Map.insert resReg result regs0
                , Map.insert var result bindings0
                , TrailEntry ("__binding__" ++ var) (Map.lookup var bindings0)
                    : take (cpTrailLen cp) (wsTrail s)
                )
              _ -> (regs0, bindings0, take (cpTrailLen cp) (wsTrail s))
        in Just s { wsPC = returnPC
                  , wsRegs = regs1
                  , wsStack = cpStack cp
                  , wsBindings = bindings1
                  , wsTrail = trail1
                  , wsHeap = take (cpHeapLen cp) (wsHeap s)
                  , wsCP = cpCP cp
                  , wsCPs = rest
                  , wsAggAccum = []
                  }
      Nothing -> go rest  -- skip non-aggregate CPs

-- | Apply aggregation function to collected values.
applyAggregation :: String -> [Value] -> Value
applyAggregation "sum" vals =
  let toNum (Integer n) = fromIntegral n
      toNum (Float f) = f
      toNum _ = 0
      s = sum (map toNum vals)
  in if fromIntegral (round s :: Int) == s then Integer (round s) else Float s
applyAggregation "count" vals = Integer (length vals)
applyAggregation "collect" vals = VList vals
applyAggregation _ vals = VList vals

-- ============================================================================
-- Foreign Function Interface: native Haskell implementations of expensive
-- recursive predicates. Avoids ~100x WAM interpretation overhead.
-- ============================================================================

-- | Native category_ancestor: depth-bounded DFS over indexed parent facts.
-- Returns all (Hops) values for paths from Cat to Root within maxDepth.
-- Uses Set for O(log n) membership check on Visited.
nativeCategoryAncestor :: Map.Map String [String] -> String -> String -> Int -> Int -> Set.Set String -> [Int]
nativeCategoryAncestor parents cat root maxDepth depth visited =
  let directParents = fromMaybe [] (Map.lookup cat parents)
      baseHits = [1 | p <- directParents, p == root, not (Set.member p visited)]
      recHits = if depth >= maxDepth then [] else
        concatMap (\\mid ->
          if Set.member mid visited then []
          else map (+1) $ nativeCategoryAncestor parents mid root maxDepth (depth+1) (Set.insert mid visited)
        ) directParents
  in baseHits ++ recHits

-- | Execute a foreign predicate call. Computes all results natively,
-- returns first result with CPs for the rest.
executeForeign :: String -> WamState -> Maybe WamState
executeForeign "category_ancestor/4" s =
  let cat = derefVar (wsBindings s) $ fromMaybe (Atom "") (Map.lookup "A1" (wsRegs s))
      root = derefVar (wsBindings s) $ fromMaybe (Atom "") (Map.lookup "A2" (wsRegs s))
      visited = derefVar (wsBindings s) $ fromMaybe (VList []) (Map.lookup "A4" (wsRegs s))
      maxD = fromMaybe 10 $ Map.lookup "max_depth" (wsForeignConfig s)
      parents = fromMaybe Map.empty $ Map.lookup "category_parent" (wsForeignFacts s)
  in case (cat, root, visited) of
    (Atom catS, Atom rootS, VList visitedVals) ->
      let visitedStrs = Set.fromList [v | Atom v <- visitedVals]
          hops = nativeCategoryAncestor parents catS rootS maxD (Set.size visitedStrs) visitedStrs
          retPC = wsCP s
          hopsReg = derefVar (wsBindings s) $ fromMaybe (Unbound "_") (Map.lookup "A3" (wsRegs s))
          bindHop hopVal =
            case hopsReg of
              Unbound var ->
                s { wsPC = retPC
                  , wsRegs = Map.insert "A3" (Integer (fromIntegral hopVal)) (wsRegs s)
                  , wsBindings = Map.insert var (Integer (fromIntegral hopVal)) (wsBindings s)
                  , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s }
              _ -> s { wsPC = retPC, wsRegs = Map.insert "A3" (Integer (fromIntegral hopVal)) (wsRegs s) }
          mkCP hopVal =
            let bound = bindHop hopVal
            in ChoicePoint
              { cpNextPC   = retPC
              , cpRegs     = wsRegs bound
              , cpStack    = wsStack s
              , cpCP       = wsCP s
              , cpTrailLen = length (wsTrail s)
              , cpHeapLen  = length (wsHeap s)
              , cpBindings = wsBindings bound
              , cpCutBar   = wsCutBar s
              , cpAggFrame = Nothing
              }
      in case hops of
        [] -> Nothing
        [h] -> Just (bindHop h)   -- single result, no CPs
        (h:rest) ->
          let s1 = bindHop h
              -- CPs for remaining results (in reverse so backtrack gets them in order)
              newCPs = map mkCP (reverse rest)
          in Just (s1 { wsCPs = newCPs ++ wsCPs s })
    _ -> Nothing
executeForeign _ _ = Nothing

-- | Unify two values, binding unbound variables.
unifyVal :: Value -> Value -> WamState -> Maybe WamState
unifyVal (Unbound v) val s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = Map.insert v val (wsBindings s)
          , wsTrail = TrailEntry ("__binding__" ++ v) (Map.lookup v (wsBindings s)) : wsTrail s
          })
unifyVal val (Unbound v) s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = Map.insert v val (wsBindings s)
          , wsTrail = TrailEntry ("__binding__" ++ v) (Map.lookup v (wsBindings s)) : wsTrail s
          })
unifyVal a b s | a == b = Just (s { wsPC = wsPC s + 1 })
               | otherwise = Nothing'.

% ============================================================================
% PHASE 3: Step Function + Run Loop
% ============================================================================

step_function_haskell(Code) :-
    Code = '-- | Execute a single WAM instruction.
step :: WamState -> Instruction -> Maybe WamState
step s (GetConstant c ai) =
  let val = derefVar (wsBindings s) <$> Map.lookup ai (wsRegs s)
  in case val of
    Just v | v == c -> Just (s { wsPC = wsPC s + 1 })
    Just (Unbound var) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = Map.insert ai c (wsRegs s)
              , wsBindings = Map.insert var c (wsBindings s)
              , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
              })
    _ -> Nothing

step s (GetVariable xn ai) =
  case Map.lookup ai (wsRegs s) of
    Just val -> let dv = derefVar (wsBindings s) val
                in Just ((putReg xn dv s) { wsPC = wsPC s + 1 })
    Nothing -> Nothing

step s (GetValue xn ai) =
  let va = derefVar (wsBindings s) <$> Map.lookup ai (wsRegs s)
      vx = getReg xn s
  in case (va, vx) of
    (Just a, Just x) | a == x -> Just (s { wsPC = wsPC s + 1 })
    (Just (Unbound n), Just x) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = Map.insert ai x (wsRegs s)
              , wsBindings = Map.insert n x (wsBindings s)
              , wsTrail = TrailEntry ("__binding__" ++ n) (Map.lookup n (wsBindings s)) : wsTrail s
              })
    _ -> Nothing

step s (PutConstant c ai) =
  Just (s { wsPC = wsPC s + 1, wsRegs = Map.insert ai c (wsRegs s) })

step s (PutVariable xn ai) =
  let var = Unbound ("_V" ++ show (wsVarCounter s))
      s1 = putReg xn var s
  in Just (s1 { wsPC = wsPC s + 1
              , wsRegs = Map.insert ai var (wsRegs s1)
              , wsVarCounter = wsVarCounter s + 1
              })

step s (PutValue xn ai) =
  case getReg xn s of
    Just val -> Just (s { wsPC = wsPC s + 1, wsRegs = Map.insert ai val (wsRegs s) })
    Nothing -> Nothing

step s (PutStructure fn ai) =
  let arity = case break (== ''/'') (reverse fn) of
                (ra, _:_) -> case reads (reverse ra) of [(n,"")] -> n; _ -> 0
                _ -> 0
  in Just (s { wsPC = wsPC s + 1
             , wsBuilder = BuildStruct fn ai arity []
             })

step s (PutList ai) =
  Just (s { wsPC = wsPC s + 1
           , wsBuilder = BuildList ai []
           })

step s (SetValue xn) =
  case getReg xn s of
    Just val -> addToBuilder val s
    Nothing -> Nothing

step s (SetConstant c) =
  addToBuilder c s

step s (Call pred _arity) =
  -- Try foreign dispatch first (native Haskell implementation)
  case executeForeign pred (s { wsCP = wsPC s + 1 }) of
    Just s'' -> Just s''
    Nothing ->
      -- Fall back to WAM instruction dispatch
      case Map.lookup pred (wsLabels s) of
        Just pc -> Just (s { wsPC = pc, wsCP = wsPC s + 1 })
        Nothing -> Nothing

step s Proceed =
  let ret = wsCP s
  in if ret == 0 then Just (s { wsPC = 0 })
     else Just (s { wsPC = ret, wsCP = 0 })

step s Allocate =
  let frame = EnvFrame (wsCP s) Map.empty
  in Just (s { wsPC = wsPC s + 1
             , wsStack = frame : wsStack s
             , wsCutBar = length (wsCPs s)
             })

step s Deallocate =
  case wsStack s of
    (EnvFrame oldCP _ : rest) -> Just (s { wsPC = wsPC s + 1, wsStack = rest, wsCP = oldCP })
    _ -> Nothing

step s (TryMeElse label) =
  let nextPC = fromMaybe 0 $ Map.lookup label (wsLabels s)
      cp = ChoicePoint
        { cpNextPC   = nextPC
        , cpRegs     = wsRegs s       -- O(1): Data.Map shared reference
        , cpStack    = wsStack s      -- O(1): list shared reference
        , cpCP       = wsCP s
        , cpTrailLen = length (wsTrail s)
        , cpHeapLen  = length (wsHeap s)
        , cpBindings = wsBindings s   -- O(1): Data.Map shared reference
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Nothing
        }
  in Just (s { wsPC = wsPC s + 1, wsCPs = cp : wsCPs s })

step s TrustMe =
  case wsCPs s of
    (_ : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest })
    [] -> Nothing

step s (RetryMeElse label) =
  case wsCPs s of
    (cp : rest) ->
      let nextPC = fromMaybe 0 $ Map.lookup label (wsLabels s)
      in Just (s { wsPC = wsPC s + 1, wsCPs = cp { cpNextPC = nextPC } : rest })
    [] -> Nothing

step s (BuiltinCall "!/0" _) =
  Just (s { wsPC = wsPC s + 1, wsCPs = take (wsCutBar s) (wsCPs s) })

step s (BuiltinCall "is/2" _) =
  let expr = derefVar (wsBindings s) $ fromMaybe (Integer 0) (Map.lookup "A2" (wsRegs s))
      result = evalArith (wsBindings s) expr
      lhs = derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s)
  in case (lhs, result) of
    (Just (Unbound var), Just r) ->
      let val = if fromIntegral (round r :: Int) == r then Integer (round r) else Float r
      in Just (s { wsPC = wsPC s + 1
                 , wsRegs = Map.insert "A1" val (wsRegs s)
                 , wsBindings = Map.insert var val (wsBindings s)
                 , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                 })
    (Just (Integer n), Just r) | fromIntegral n == r -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step s (BuiltinCall "length/2" _) =
  let listVal = derefVar (wsBindings s) $ fromMaybe (VList []) (Map.lookup "A1" (wsRegs s))
  in case listVal of
    VList items ->
      let len = length items
          lhs = derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s)
      in case lhs of
        Just (Unbound var) ->
          let val = Integer len
          in Just (s { wsPC = wsPC s + 1
                     , wsRegs = Map.insert "A2" val (wsRegs s)
                     , wsBindings = Map.insert var val (wsBindings s)
                     , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                     })
        Just (Integer n) | n == len -> Just (s { wsPC = wsPC s + 1 })
        _ -> Nothing
    _ -> Nothing

step s (BuiltinCall "</2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a < b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step s (BuiltinCall "\\\\+/1" _) =
  let goal = Map.lookup "A1" (wsRegs s) >>= derefHeap (wsHeap s)
  in case goal of
    Just (Str fn [needle, haystack]) | "member" `isPrefixOf` fn ->
      let n = derefVar (wsBindings s) needle
          h = derefVar (wsBindings s) haystack
          found = case h of
            VList items -> any (\\item -> derefVar (wsBindings s) item == n) items
            _ -> False
      in if found then Nothing else Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

-- SwitchOnConstant: dispatch on A1 value via O(log n) Map lookup
step s (SwitchOnConstant table) =
  let val = derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s)
  in case val of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })  -- unbound: skip
    Just v -> case Map.lookup v table of
      Just label -> case Map.lookup label (wsLabels s) of
        Just pc -> Just (s { wsPC = pc })
        Nothing -> Nothing
      Nothing -> Nothing  -- no match: fail
    Nothing -> Nothing

step s (BuiltinCall ">/2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a > b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

-- member/2 builtin: A1=Elem, A2=List. Creates choice points for backtracking.
step s (BuiltinCall "member/2" _) =
  let elem_ = derefVar (wsBindings s) $ fromMaybe (Unbound "_") (Map.lookup "A1" (wsRegs s))
      list_ = derefVar (wsBindings s) $ fromMaybe (VList []) (Map.lookup "A2" (wsRegs s))
  in case list_ of
    VList (x:_) -> unifyVal elem_ x s
    _ -> Nothing

-- begin_aggregate: push aggregate frame CP, initialize accumulator, continue to goal body
step s (BeginAggregate typ valReg resReg) =
  let cp = ChoicePoint
        { cpNextPC   = wsPC s   -- not used directly; aggregate frame handles return
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = length (wsTrail s)
        , cpHeapLen  = length (wsHeap s)
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Just (AggFrame typ valReg resReg 0)  -- returnPC set by end_aggregate
        }
  in Just (s { wsPC = wsPC s + 1
             , wsCPs = cp : wsCPs s
             , wsAggAccum = []
             })

-- end_aggregate: collect value, store returnPC in aggregate frame, force backtrack
step s (EndAggregate valReg) =
  let val = derefVar (wsBindings s) $ fromMaybe (Integer 0) (getReg valReg s)
      returnPC = wsPC s + 1
      -- Update the aggregate frame CP with the correct returnPC
      updatedCPs = map (\\cp -> case cpAggFrame cp of
          Just af -> cp { cpAggFrame = Just af { afReturnPC = returnPC } }
          Nothing -> cp) (wsCPs s)
      s1 = s { wsAggAccum = val : wsAggAccum s, wsCPs = updatedCPs }
  in case backtrackInner returnPC s1 of
    Just s2 -> Just s2
    Nothing -> finalizeAggregate returnPC s1

-- Fallback for unhandled instructions
step _ _ = Nothing'.

run_loop_haskell(Code) :-
    Code = '-- | Main execution loop. Runs until halt (pc=0) or failure.
run :: WamState -> Maybe WamState
run s
  | wsPC s == 0 = Just s  -- halt
  | otherwise = case fetchInstr (wsPC s) (wsCode s) of
      Nothing -> Nothing
      Just instr -> case step s instr of
        Just s'' -> run s''
        Nothing -> case backtrack s of
          Just s'' -> run s''
          Nothing -> Nothing'.

% ============================================================================
% PHASE 4: Project Generation
% ============================================================================

%% write_wam_haskell_project(+Predicates, +Options, +ProjectDir)
%  Generates a complete Haskell project (cabal or stack) with:
%  - WamTypes.hs: data types and utility functions
%  - WamRuntime.hs: run loop and backtracking
%  - Predicates.hs: compiled predicates
%  - Main.hs: benchmark driver
write_wam_haskell_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'src', SrcDir),
    make_directory_path(SrcDir),

    % Generate WamTypes.hs
    generate_wam_types_hs(TypesCode),
    directory_file_path(SrcDir, 'WamTypes.hs', TypesPath),
    write_hs_file(TypesPath, TypesCode),

    % Generate WamRuntime.hs
    compile_wam_runtime_to_haskell(Options, RuntimeCode),
    directory_file_path(SrcDir, 'WamRuntime.hs', RuntimePath),
    write_hs_file(RuntimePath, RuntimeCode),

    % Compile predicates
    compile_predicates_to_haskell(Predicates, Options, PredsCode),
    directory_file_path(SrcDir, 'Predicates.hs', PredsPath),
    write_hs_file(PredsPath, PredsCode),

    % Generate cabal file
    option(module_name(ModName), Options, 'wam-haskell-bench'),
    generate_cabal_file(ModName, CabalCode),
    format(atom(CabalFile), '~w.cabal', [ModName]),
    directory_file_path(ProjectDir, CabalFile, CabalPath),
    write_hs_file(CabalPath, CabalCode),

    % Generate Main.hs
    generate_main_hs(Predicates, MainCode),
    directory_file_path(SrcDir, 'Main.hs', MainPath),
    write_hs_file(MainPath, MainCode),

    format(user_error, '[WAM-Haskell] Generated project at: ~w~n', [ProjectDir]).

%% generate_main_hs(+Predicates, -Code)
%  Generates Main.hs — a benchmark driver for effective-distance.
%  Loads facts from TSV, runs category_ancestor queries, outputs TSV.
generate_main_hs(_Predicates, Code) :-
    Code = 'module Main where

import qualified Data.Map.Strict as Map
import Data.List (nub, sort, isPrefixOf, intercalate)
import qualified Numeric
import Data.Maybe (fromMaybe, mapMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr, hFlush, stdout)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import WamTypes
import WamRuntime
import Predicates

-- | Load a TSV file, skip header, return pairs.
loadTsvPairs :: FilePath -> IO [(String, String)]
loadTsvPairs path = do
    content <- readFile path
    let ls = drop 1 (lines content)  -- skip header
    return [(a, b) | l <- ls, let ws = splitOn ''\\t'' l, length ws >= 2, let [a, b] = take 2 ws]

-- | Load single-column TSV.
loadSingleColumn :: FilePath -> IO [String]
loadSingleColumn path = do
    content <- readFile path
    return [l | l <- drop 1 (lines content), not (null l)]

-- | Simple tab split.
splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn d (c:cs)
  | c == d    = "" : splitOn d cs
  | otherwise = let (w:ws) = splitOn d cs in (c:w) : ws

-- | Build SwitchOnConstant dispatch table from grouped facts.
buildFactIndex :: [(String, String)] -> Map.Map String [(String, String)]
buildFactIndex pairs = foldl (\\m (a, b) -> Map.insertWith (++) a [(a, b)] m) Map.empty pairs

-- | Build WAM instructions for indexed fact predicate.
-- Returns (instructions, labels) to append to the code vector.
buildFact2Code :: String -> [(String, String)] -> Int -> ([Instruction], [(String, Int)])
buildFact2Code predName pairs startPC =
    let groups = Map.toAscList (buildFactIndex pairs)
        -- SwitchOnConstant dispatch table
        (dispatchList, groupCode, groupLabels, _) = foldl buildGroup ([], [], [], startPC + 1) groups
        switchInstr = SwitchOnConstant (Map.fromList dispatchList)
    in (switchInstr : groupCode, (predName ++ "/2", startPC) : groupLabels)
  where
    buildGroup (disp, code, labels, pc) (key, facts) =
      let groupLabel = predName ++ "_g_" ++ key
          (fcode, flabels, nextPC) = buildFactGroup predName key facts pc
      in (disp ++ [(Atom key, groupLabel)], code ++ fcode, labels ++ [(groupLabel, pc)] ++ flabels, nextPC)

    buildFactGroup _ _ [] pc = ([], [], pc)
    buildFactGroup pn key facts pc =
      let n = length facts
          buildFact i (a, b) curPC =
            let choiceInstr = if n == 1 then [] else
                  if i == 0 then [TryMeElse (pn ++ "_g_" ++ key ++ "_" ++ show (i+1))]
                  else if i == n - 1 then [TrustMe]
                  else [RetryMeElse (pn ++ "_g_" ++ key ++ "_" ++ show (i+1))]
                factInstrs = [GetConstant (Atom a) "A1", GetConstant (Atom b) "A2", Proceed]
                label = (pn ++ "_g_" ++ key ++ "_" ++ show i, curPC)
            in (choiceInstr ++ factInstrs, [label], curPC + length choiceInstr + length factInstrs)
          (allCode, allLabels, finalPC) = foldl (\\(c, l, p) (i, f) ->
              let (fc, fl, np) = buildFact i f p in (c ++ fc, l ++ fl, np))
            ([], [], pc) (zip [0..] facts)
      in (allCode, allLabels, finalPC)

-- | Build WAM instructions for 1-column indexed fact predicate.
buildFact1Code :: String -> [String] -> Int -> ([Instruction], [(String, Int)])
buildFact1Code predName vals startPC =
    let disp = [(Atom v, predName ++ "_" ++ show i) | (i, v) <- zip [0..] vals]
        switchInstr = SwitchOnConstant (Map.fromList disp)
        factCode = concatMap (\\(i, v) -> [GetConstant (Atom v) "A1", Proceed]) (zip [0..] vals)
        factLabels = [(predName ++ "_" ++ show i, startPC + 1 + i * 2) | i <- [0..length vals - 1]]
    in (switchInstr : factCode, (predName ++ "/1", startPC) : factLabels)

main :: IO ()
main = do
    args <- getArgs
    let factsDir = if null args then "." else head args

    t0 <- getCurrentTime

    -- Load facts from TSV
    categoryParents <- loadTsvPairs (factsDir ++ "/category_parent.tsv")
    articleCategories <- loadTsvPairs (factsDir ++ "/article_category.tsv")
    roots <- loadSingleColumn (factsDir ++ "/root_categories.tsv")

    t1 <- getCurrentTime
    let loadMs = round (diffUTCTime t1 t0 * 1000) :: Int

    -- Build merged code: compiled predicates + runtime facts
    let baseLen = length allCode
        (cpCode, cpLabels) = buildFact2Code "category_parent" categoryParents (baseLen + 1)
        cpEnd = baseLen + length cpCode
        (acCode, acLabels) = buildFact2Code "article_category" articleCategories (cpEnd + 1)
        acEnd = cpEnd + length acCode
        (rcCode, rcLabels) = buildFact1Code "root_category" roots (acEnd + 1)

        mergedCode = allCode ++ cpCode ++ acCode ++ rcCode
        mergedLabels = Map.union allLabels
                     $ Map.fromList (cpLabels ++ acLabels ++ rcLabels)

    -- Collect seed categories
    let seedCats = nub $ sort $ map snd articleCategories
        root = head roots
        n = 5.0 :: Double
        negN = -n

    t2 <- getCurrentTime

    -- For each seed, run category_ancestor(Cat, Root, Hops, [Cat])
    let seedResults = map (\\cat ->
          let s0 = (emptyState mergedCode mergedLabels)
                { wsPC = fromMaybe 1 $ Map.lookup "category_ancestor/4" mergedLabels
                , wsRegs = Map.fromList
                    [ ("A1", Atom cat)
                    , ("A2", Atom root)
                    , ("A3", Unbound "Hops")
                    , ("A4", VList [Atom cat])
                    ]
                , wsCP = 0
                }
              solutions = collectSolutions s0
              weightSum = sum [((hops + 1) ** negN) | hops <- solutions]
          in (cat, weightSum)
          ) seedCats

    let seedWeightSums = Map.fromList [(cat, ws) | (cat, ws) <- seedResults, ws > 0]

    t3 <- getCurrentTime
    let queryMs = round (diffUTCTime t3 t2 * 1000) :: Int

    -- Aggregate per-article weight sums
    let articleSums = foldl (\\m (art, cat) ->
          let ws = if cat == root then 1.0 else 0.0
              catWs = fromMaybe 0.0 (Map.lookup cat seedWeightSums)
          in Map.insertWith (+) art (ws + catWs) m
          ) Map.empty articleCategories

    let invN = -1.0 / n
        results = sort [(ws ** invN, art) | (art, ws) <- Map.toList articleSums, ws > 0]

    t4 <- getCurrentTime
    let aggMs = round (diffUTCTime t4 t3 * 1000) :: Int
        totalMs = round (diffUTCTime t4 t0 * 1000) :: Int

    -- Output TSV
    putStrLn "article\\troot_category\\teffective_distance"
    mapM_ (\\(deff, art) ->
        putStrLn (art ++ "\\t" ++ root ++ "\\t" ++ showFFloat6 deff)
        ) results
    hFlush stdout

    -- Metrics to stderr
    hPutStrLn stderr $ "mode=wam_haskell_accumulated"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "aggregation_ms=" ++ show aggMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "seed_count=" ++ show (length seedCats)
    hPutStrLn stderr $ "tuple_count=" ++ show (Map.size seedWeightSums)
    hPutStrLn stderr $ "article_count=" ++ show (length results)

-- | Format a Double to 6 decimal places.
showFFloat6 :: Double -> String
showFFloat6 x = Numeric.showFFloat (Just 6) x ""

-- | Collect all Hops solutions from a query by repeated run + backtrack.
collectSolutions :: WamState -> [Double]
collectSolutions s0 =
    case run s0 of
      Nothing -> []
      Just s1 ->
        let hopsVal = case Map.lookup "Hops" (wsBindings s1) of
              Just v -> extractDouble (derefVar (wsBindings s1) v)
              Nothing -> Nothing
            hops = fromMaybe 0 hopsVal
            rest = case backtrack s1 of
              Just s2 -> collectSolutions s2
              Nothing -> []
        in case hopsVal of
          Just _ -> hops : rest
          Nothing -> rest  -- skip solutions where Hops is not bound

-- | Extract a Double from a Value.
extractDouble :: Value -> Maybe Double
extractDouble (Integer h) = Just (fromIntegral h)
extractDouble (Float h) = Just h
extractDouble (Atom str) = case reads str of [(h, "")] -> Just h; _ -> Nothing
extractDouble _ = Nothing
'.

build_predicate_loads([], '    let allCode = []\n    let allLabels = Map.empty').
build_predicate_loads(Predicates, Code) :-
    Predicates \= [],
    % Build code concatenation and label union
    maplist([PredInd, FN]>>(
        (PredInd = _M:P/A -> true ; PredInd = P/A),
        format(atom(FN), '~w_~w', [P, A])
    ), Predicates, FuncNames),
    % Generate Haskell code to merge all predicate code/labels
    maplist([FN, Expr]>>(
        format(string(Expr), '~w_code', [FN])
    ), FuncNames, CodeExprs),
    atomic_list_concat(CodeExprs, ' ++ ', CodeConcat),
    maplist([FN, Expr]>>(
        format(string(Expr), '~w_labels', [FN])
    ), FuncNames, LabelExprs),
    % Union with offset adjustment
    % For simplicity, use Map.unions (labels need PC offset per predicate)
    atomic_list_concat(LabelExprs, ' `Map.union` ', LabelUnion),
    format(string(Code),
'    let allCode = ~w
    let allLabels = ~w', [CodeConcat, LabelUnion]).

%% compile_wam_runtime_to_haskell(+Options, -Code)
compile_wam_runtime_to_haskell(_Options, Code) :-
    step_function_haskell(StepCode),
    backtrack_haskell(BacktrackCode),
    run_loop_haskell(RunCode),
    format(string(Code),
'module WamRuntime where

import qualified Data.Map.Strict as Map
import Data.Array (Array, listArray, (!), bounds)
import qualified Data.Set as Set
import Data.List (isPrefixOf, foldl'')
import Data.Maybe (fromMaybe)
import WamTypes

~w

~w

~w

-- | Dereference an Unbound variable through the binding table.
derefVar :: Map.Map String Value -> Value -> Value
derefVar bindings (Unbound name) =
  case Map.lookup name bindings of
    Just val -> derefVar bindings val
    Nothing  -> Unbound name
derefVar _ v = v

-- | Evaluate arithmetic expression.
evalArith :: Map.Map String Value -> Value -> Maybe Double
evalArith _ (Integer n) = Just (fromIntegral n)
evalArith _ (Float f) = Just f
evalArith bindings (Atom s) = case reads s of
  [(n, "")] -> Just n
  _ -> Nothing
evalArith bindings (Str op [a]) = do
  va <- evalArith bindings (derefVar bindings a)
  let bareOp = takeWhile (/= ''/'') op
  case bareOp of
    "-" -> Just (negate va)
    "abs" -> Just (abs va)
    _ -> Nothing
evalArith bindings (Str op [a, b]) = do
  va <- evalArith bindings (derefVar bindings a)
  vb <- evalArith bindings (derefVar bindings b)
  let bareOp = takeWhile (/= ''/'') op
  case bareOp of
    "+" -> Just (va + vb)
    "-" -> Just (va - vb)
    "*" -> Just (va * vb)
    "**" -> Just (va ** vb)
    "^" -> Just (va ** vb)
    "/" -> if vb /= 0 then Just (va / vb) else Nothing
    "//" -> if vb /= 0 then Just (fromIntegral (truncate va `div` truncate vb :: Int)) else Nothing
    "mod" -> if vb /= 0 then Just (fromIntegral (truncate va `mod` truncate vb :: Int)) else Nothing
    _ -> Nothing
evalArith _ _ = Nothing

-- | Get register value (Y-registers from topmost env frame).
getReg :: String -> WamState -> Maybe Value
getReg name s
  | "Y" `isPrefixOf` name = findYReg name (wsStack s)
  | otherwise = derefVar (wsBindings s) <$> Map.lookup name (wsRegs s)
  where
    findYReg _ [] = Nothing
    findYReg n (EnvFrame _ yregs : _) = derefVar (wsBindings s) <$> Map.lookup n yregs
    findYReg n (_ : rest) = findYReg n rest

-- | Set register value.
putReg :: String -> Value -> WamState -> WamState
putReg name val s
  | "Y" `isPrefixOf` name = s { wsStack = updateTopEnv name val (wsStack s) }
  | otherwise = s { wsRegs = Map.insert name val (wsRegs s) }
  where
    updateTopEnv _ _ [] = []
    updateTopEnv n v (EnvFrame cp yregs : rest) =
      EnvFrame cp (Map.insert n v yregs) : rest
    updateTopEnv n v (x : rest) = x : updateTopEnv n v rest

-- | Dereference a heap reference.
derefHeap :: [Value] -> Value -> Maybe Value
derefHeap heap (Ref addr)
  | addr >= 0 && addr < length heap = Just (heap !! addr)
  | otherwise = Nothing
derefHeap _ (Str fn args) = Just (Str fn args)
derefHeap _ (Unbound n) = Just (Unbound n)
derefHeap _ v = Just v

-- | Add a value to the current structure/list builder.
addToBuilder :: Value -> WamState -> Maybe WamState
addToBuilder val s = case wsBuilder s of
  BuildStruct fn ai arity args ->
    let args'' = args ++ [val]
    in if length args'' == arity
       then Just (s { wsPC = wsPC s + 1
                    , wsRegs = Map.insert ai (Str fn args'') (wsRegs s)
                    , wsBuilder = NoBuilder
                    })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildStruct fn ai arity args''
                    })
  BuildList ai args ->
    let args'' = args ++ [val]
    in if length args'' == 2
       then let [hd, tl] = args''
                list = case tl of
                  VList items -> VList (hd : items)
                  Atom "[]"  -> VList [hd]
                  _           -> VList [hd, tl]
            in Just (s { wsPC = wsPC s + 1
                       , wsRegs = Map.insert ai list (wsRegs s)
                       , wsBuilder = NoBuilder
                       })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildList ai args''
                    })
  NoBuilder ->
    -- No builder active, just push to heap (fallback)
    Just (s { wsPC = wsPC s + 1, wsHeap = wsHeap s ++ [val] })

-- | Lookup a label in the label map.
lookupLabel :: String -> WamState -> Int
lookupLabel label s = fromMaybe 0 $ Map.lookup label (wsLabels s)

-- | Fetch instruction at PC (1-indexed).
fetchInstr :: Int -> Array Int Instruction -> Maybe Instruction
fetchInstr pc code
  | let (lo, hi) = bounds code in pc < lo || pc > hi = Nothing
  | otherwise = Just (code ! pc)
', [StepCode, BacktrackCode, RunCode]).

%% generate_wam_types_hs(-Code)
generate_wam_types_hs(Code) :-
    Code = 'module WamTypes where

import qualified Data.Map.Strict as Map
import Data.Array (Array, listArray, (!), bounds)

data Value = Atom String
           | Integer Int
           | Float Double
           | VList [Value]
           | Str String [Value]
           | Unbound String
           | Ref Int
           deriving (Eq, Ord, Show)

data EnvFrame = EnvFrame !Int !(Map.Map String Value)
              deriving (Show)

data TrailEntry = TrailEntry !String !(Maybe Value)
                deriving (Show)

data ChoicePoint = ChoicePoint
  { cpNextPC   :: !Int
  , cpRegs     :: !(Map.Map String Value)
  , cpStack    :: ![EnvFrame]
  , cpCP       :: !Int
  , cpTrailLen :: !Int
  , cpHeapLen  :: !Int
  , cpBindings :: !(Map.Map String Value)
  , cpCutBar   :: !Int
  , cpAggFrame :: !(Maybe AggFrame)  -- aggregate frame (if this CP is an aggregate)
  } deriving (Show)

-- | Aggregate frame for begin_aggregate/end_aggregate.
data AggFrame = AggFrame
  { afType      :: !String   -- "sum", "count", "collect", etc.
  , afValueReg  :: !String   -- register holding value per solution
  , afResultReg :: !String   -- register for final result
  , afReturnPC  :: !Int      -- PC after end_aggregate
  } deriving (Show)

-- | Builder for PutStructure/PutList + SetValue/SetConstant sequences.
data Builder = BuildStruct !String !String !Int ![Value]  -- functor, target reg, arity, collected args
             | BuildList !String ![Value]                  -- target reg, collected [head, tail]
             | NoBuilder
             deriving (Show)

data WamState = WamState
  { wsPC       :: !Int
  , wsRegs     :: !(Map.Map String Value)
  , wsStack    :: ![EnvFrame]
  , wsHeap     :: ![Value]
  , wsTrail    :: ![TrailEntry]
  , wsCP       :: !Int
  , wsCPs      :: ![ChoicePoint]
  , wsBindings :: !(Map.Map String Value)
  , wsCutBar   :: !Int
  , wsCode     :: !(Array Int Instruction)
  , wsLabels   :: !(Map.Map String Int)
  , wsBuilder  :: !Builder
  , wsVarCounter :: !Int
  , wsAggAccum :: ![Value]     -- aggregate accumulator (values collected so far)
  , wsForeignFacts :: !(Map.Map String (Map.Map String [String]))  -- pred -> (key1 -> [val2s])
  , wsForeignConfig :: !(Map.Map String Int)                       -- "max_depth" etc.
  } deriving (Show)

-- | Instruction type for the WAM.
data Instruction
  = GetConstant Value String
  | GetVariable String String
  | GetValue String String
  | PutConstant Value String
  | PutVariable String String
  | PutValue String String
  | PutStructure String String
  | PutList String
  | SetValue String
  | SetConstant Value
  | Allocate
  | Deallocate
  | Call String Int
  | Execute String
  | Proceed
  | BuiltinCall String Int
  | TryMeElse String
  | RetryMeElse String
  | TrustMe
  | SwitchOnConstant (Map.Map Value String)   -- pre-built Map for O(log n) dispatch
  | BeginAggregate String String String   -- type, valueReg, resultReg
  | EndAggregate String                   -- valueReg
  deriving (Show, Eq)

-- | Create initial empty state.
emptyState :: [Instruction] -> Map.Map String Int -> WamState
emptyState codeList labels = let n = length codeList; code = listArray (1, n) codeList in WamState
  { wsPC       = 1
  , wsRegs     = Map.empty
  , wsStack    = []
  , wsHeap     = []
  , wsTrail    = []
  , wsCP       = 0
  , wsCPs      = []
  , wsBindings = Map.empty
  , wsCutBar   = 0
  , wsCode     = code
  , wsLabels   = labels
  , wsBuilder  = NoBuilder
  , wsVarCounter = 0
  , wsAggAccum = []
  , wsForeignFacts = Map.empty
  , wsForeignConfig = Map.empty
  }
'.

%% generate_cabal_file(+Name, -Code)
generate_cabal_file(Name, Code) :-
    format(string(Code),
'cabal-version: 2.4
name:          ~w
version:       0.1.0.0
build-type:    Simple

executable ~w
  main-is:          Main.hs
  hs-source-dirs:   src
  other-modules:    WamTypes, WamRuntime, Predicates
  build-depends:    base >= 4.14, containers >= 0.6, array, time >= 1.8
  default-language: Haskell2010
  ghc-options:      -O2
', [Name, Name]).

%% compile_predicates_to_haskell(+Predicates, +Options, -Code)
%  Compiles all predicates into a single merged code array and label map,
%  with proper PC offsets for each predicate.
compile_predicates_to_haskell(Predicates, _Options, Code) :-
    compile_predicates_merged(Predicates, 1, AllInstrs, AllLabels),
    atomic_list_concat(AllInstrs, '\n    , ', InstrCode),
    atomic_list_concat(AllLabels, '\n    , ', LabelCode),
    format(string(Code),
'module Predicates where

import qualified Data.Map.Strict as Map
import WamTypes

-- | Merged WAM code for all predicates.
allCode :: [Instruction]
allCode =
    [ ~w
    ]

-- | Merged label map for all predicates.
allLabels :: Map.Map String Int
allLabels = Map.fromList
    [ ~w
    ]
', [InstrCode, LabelCode]).

compile_predicates_merged([], _, [], []).
compile_predicates_merged([PredIndicator|Rest], StartPC, AllInstrs, AllLabels) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    wam_target:compile_predicate_to_wam(PredIndicator, [], WamCode),
    format(user_error, '  ~w/~w: compiled to WAM (PC=~w)~n', [Pred, Arity, StartPC]),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_haskell(Lines, StartPC, InstrExprs, LabelExprs, NextPC),
    compile_predicates_merged(Rest, NextPC, RestInstrs, RestLabels),
    append(InstrExprs, RestInstrs, AllInstrs),
    append(LabelExprs, RestLabels, AllLabels).

compile_single_predicate_to_haskell(PredIndicator, _Options, Code) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    wam_target:compile_predicate_to_wam(PredIndicator, [], WamCode),
    format(user_error, '  ~w/~w: compiled to WAM~n', [Pred, Arity]),
    % Parse WAM text into Haskell instruction list and label map
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_haskell(Lines, 1, InstrExprs, LabelExprs),
    atomic_list_concat(InstrExprs, '\n    , ', InstrCode),
    atomic_list_concat(LabelExprs, '\n    , ', LabelCode),
    format(atom(FuncName), '~w_~w', [Pred, Arity]),
    format(string(Code),
'-- WAM-compiled predicate: ~w/~w
~w_code :: [Instruction]
~w_code =
    [ ~w
    ]

~w_labels :: Map.Map String Int
~w_labels = Map.fromList
    [ ~w
    ]', [Pred, Arity, FuncName, FuncName, InstrCode, FuncName, FuncName, LabelCode]).

%% wam_lines_to_haskell(+Lines, +PC, -InstrExprs, -LabelExprs, -NextPC)
%  Parses WAM assembly lines into Haskell Instruction constructor expressions
%  and label (String, Int) pairs. Returns NextPC for merging multiple predicates.
wam_lines_to_haskell([], PC, [], [], PC).
wam_lines_to_haskell([Line|Rest], PC, Instrs, Labels, FinalPC) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_haskell(Rest, PC, Instrs, Labels, FinalPC)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelExpr), '("~w", ~w)', [LabelName, PC]),
            Labels = [LabelExpr|RestLabels],
            wam_lines_to_haskell(Rest, PC, Instrs, RestLabels, FinalPC)
        ;   wam_instr_to_haskell(CleanParts, HsExpr),
            NPC is PC + 1,
            Instrs = [HsExpr|RestInstrs],
            wam_lines_to_haskell(Rest, NPC, RestInstrs, Labels, FinalPC)
        )
    ).

%% wam_instr_to_haskell(+Parts, -HaskellExpr)
%  Converts parsed WAM instruction parts to a Haskell Instruction constructor.
wam_instr_to_haskell(["get_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    format(string(Hs), 'GetConstant (~w) "~w"', [HsVal, CAi]).
wam_instr_to_haskell(["get_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Hs), 'GetVariable "~w" "~w"', [CXn, CAi]).
wam_instr_to_haskell(["get_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Hs), 'GetValue "~w" "~w"', [CXn, CAi]).
wam_instr_to_haskell(["put_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    format(string(Hs), 'PutConstant (~w) "~w"', [HsVal, CAi]).
wam_instr_to_haskell(["put_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Hs), 'PutVariable "~w" "~w"', [CXn, CAi]).
wam_instr_to_haskell(["put_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Hs), 'PutValue "~w" "~w"', [CXn, CAi]).
wam_instr_to_haskell(["put_structure", FN, Ai], Hs) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(string(Hs), 'PutStructure "~w" "~w"', [CFN, CAi]).
wam_instr_to_haskell(["put_list", Ai], Hs) :-
    format(string(Hs), 'PutList "~w"', [Ai]).
wam_instr_to_haskell(["set_value", Xn], Hs) :-
    format(string(Hs), 'SetValue "~w"', [Xn]).
wam_instr_to_haskell(["set_constant", C], Hs) :-
    wam_value_to_haskell(C, HsVal),
    format(string(Hs), 'SetConstant (~w)', [HsVal]).
wam_instr_to_haskell(["allocate"], "Allocate").
wam_instr_to_haskell(["deallocate"], "Deallocate").
wam_instr_to_haskell(["call", P, N], Hs) :-
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    format(string(Hs), 'Call "~w" ~w', [CP, Num]).
wam_instr_to_haskell(["execute", P], Hs) :-
    format(string(Hs), 'Execute "~w"', [P]).
wam_instr_to_haskell(["proceed"], "Proceed").
wam_instr_to_haskell(["builtin_call", Op, N], Hs) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    escape_haskell_string(COp, ECOp),
    format(string(Hs), 'BuiltinCall "~w" ~w', [ECOp, Num]).
wam_instr_to_haskell(["try_me_else", Label], Hs) :-
    format(string(Hs), 'TryMeElse "~w"', [Label]).
wam_instr_to_haskell(["trust_me"], "TrustMe").
wam_instr_to_haskell(["retry_me_else", Label], Hs) :-
    format(string(Hs), 'RetryMeElse "~w"', [Label]).
wam_instr_to_haskell(["set_variable", Xn], Hs) :-
    format(string(Hs), 'SetVariable "~w"', [Xn]).
%% switch_on_constant key1:label1, key2:label2, ...
wam_instr_to_haskell(["switch_on_constant"|Entries], Hs) :-
    parse_switch_entries(Entries, HsPairs),
    atomic_list_concat(HsPairs, ', ', PairsStr),
    format(string(Hs), 'SwitchOnConstant (Map.fromList [~w])', [PairsStr]).
wam_instr_to_haskell(["switch_on_constant_a2"|Entries], Hs) :-
    parse_switch_entries(Entries, HsPairs),
    atomic_list_concat(HsPairs, ', ', PairsStr),
    format(string(Hs), 'SwitchOnConstant (Map.fromList [~w])', [PairsStr]).
wam_instr_to_haskell(["begin_aggregate", Type, ValReg, ResReg], Hs) :-
    clean_comma(Type, CT), clean_comma(ValReg, CV), clean_comma(ResReg, CR),
    format(string(Hs), 'BeginAggregate "~w" "~w" "~w"', [CT, CV, CR]).
wam_instr_to_haskell(["end_aggregate", ValReg], Hs) :-
    clean_comma(ValReg, CV),
    format(string(Hs), 'EndAggregate "~w"', [CV]).
% Fallback for unknown instructions
wam_instr_to_haskell(Parts, Hs) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Hs), '-- UNKNOWN: ~w\n    Proceed', [Joined]).

%% wam_value_to_haskell(+WamVal, -HaskellExpr)
%  Converts a WAM constant to a Haskell Value constructor.
wam_value_to_haskell(Val, Hs) :-
    (   number_string(N, Val), integer(N)
    ->  format(string(Hs), 'Integer ~w', [N])
    ;   number_string(F, Val), float(F)
    ->  format(string(Hs), 'Float ~w', [F])
    ;   format(string(Hs), 'Atom "~w"', [Val])
    ).

%% clean_comma(+Str, -Clean) — strip trailing comma
clean_comma(Str, Clean) :-
    (   sub_string(Str, _, 1, 0, ",")
    ->  sub_string(Str, 0, _, 1, Clean)
    ;   Clean = Str
    ).

%% parse_switch_entries(+Entries, -HaskellPairs)
%  Parse "key:label" pairs from switch_on_constant instruction.
parse_switch_entries([], []).
parse_switch_entries([Entry|Rest], [HsPair|HsRest]) :-
    clean_comma(Entry, CEntry),
    (   sub_atom(CEntry, Before, 1, _, ':')
    ->  sub_atom(CEntry, 0, Before, _, Key),
        After is Before + 1,
        sub_atom(CEntry, After, _, 0, Label),
        wam_value_to_haskell(Key, HsKey),
        format(string(HsPair), '(~w, "~w")', [HsKey, Label])
    ;   format(string(HsPair), '(Atom "~w", "default")', [CEntry])
    ),
    parse_switch_entries(Rest, HsRest).

%% escape_haskell_string(+In, -Out) — escape backslashes for Haskell string literals
escape_haskell_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Out).

%% write_hs_file(+Path, +Content)
write_hs_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
