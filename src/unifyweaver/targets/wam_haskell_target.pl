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
    Code = '-- | Restore state from the top choice point.
-- Dispatches: aggregate frame -> finalize, builtin -> resumeBuiltin, normal -> restore.
backtrack :: WamState -> Maybe WamState
backtrack s = case wsCPs s of
  [] -> Nothing
  (cp : rest) ->
    -- 1. Aggregate frame: finalize
    case cpAggFrame cp of { Just af -> finalizeAggregate (afReturnPC af) s; Nothing ->
    -- 2. Builtin state: resume (fact_retry etc.)
    case cpBuiltin cp of { Just bs -> resumeBuiltin bs cp rest s; Nothing ->
    -- 3. Normal: restore from CP
    let trailLen = cpTrailLen cp
        diff = wsTrailLen s - trailLen
        newEntries = reverse $ take diff (wsTrail s)
        restoredBindings = foldl'' undoBinding (cpBindings cp) newEntries
    in Just s { wsPC       = cpNextPC cp
              , wsRegs     = cpRegs cp
              , wsStack    = cpStack cp
              , wsCP       = cpCP cp
              , wsTrail    = drop diff (wsTrail s)
              , wsTrailLen = trailLen
              , wsHeap     = take (cpHeapLen cp) (wsHeap s)
              , wsHeapLen  = cpHeapLen cp
              , wsBindings = restoredBindings
              , wsCutBar   = cpCutBar cp
              } } }
  where
    undoBinding bindings (TrailEntry vid mOld) =
      case mOld of
        Just old -> IM.insert vid old bindings
        Nothing  -> IM.delete vid bindings

-- | Resume a builtin choice point. Tries next match, updates or pops CP.
resumeBuiltin :: BuiltinState -> ChoicePoint -> [ChoicePoint] -> WamState -> Maybe WamState
resumeBuiltin (FactRetry _ [] _) _ rest s =
  backtrack (s { wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
resumeBuiltin (FactRetry vid (v:vs) retPC) cp rest s =
  let newBindings = IM.insert vid (Atom v) (cpBindings cp)
      newRegs = IM.insert 2 (Atom v) (cpRegs cp)
      newCPs = case vs of
        [] -> rest
        _  -> cp { cpBuiltin = Just (FactRetry vid vs retPC) } : rest
      diff = wsTrailLen s - cpTrailLen cp
  in Just s { wsPC = retPC, wsRegs = newRegs, wsStack = cpStack cp
            , wsCP = cpCP cp
            , wsTrail = drop diff (wsTrail s)
            , wsTrailLen = cpTrailLen cp
            , wsHeap = take (cpHeapLen cp) (wsHeap s)
            , wsHeapLen = cpHeapLen cp
            , wsBindings = newBindings, wsCutBar = cpCutBar cp, wsCPs = newCPs }
resumeBuiltin (HopsRetry _ [] _) _ rest s =
  backtrack (s { wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
resumeBuiltin (HopsRetry vid (h:hs) retPC) cp rest s =
  let newBindings = IM.insert vid (Integer (fromIntegral h)) (cpBindings cp)
      newRegs = IM.insert 3 (Integer (fromIntegral h)) (cpRegs cp)
      newCPs = case hs of
        [] -> rest
        _  -> cp { cpBuiltin = Just (HopsRetry vid hs retPC) } : rest
      diff = wsTrailLen s - cpTrailLen cp
  in Just s { wsPC = retPC, wsRegs = newRegs, wsStack = cpStack cp
            , wsCP = cpCP cp
            , wsTrail = drop diff (wsTrail s)
            , wsTrailLen = cpTrailLen cp
            , wsHeap = take (cpHeapLen cp) (wsHeap s)
            , wsHeapLen = cpHeapLen cp
            , wsBindings = newBindings, wsCutBar = cpCutBar cp, wsCPs = newCPs }

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
-- | Update only the nearest aggregate frame CP with returnPC. O(k) where
-- k is the number of inner CPs above the aggregate frame, not O(n) over all CPs.
updateNearestAggFrame :: Int -> [ChoicePoint] -> [ChoicePoint]
updateNearestAggFrame _ [] = []
updateNearestAggFrame rpc (cp:rest) = case cpAggFrame cp of
  Just af -> cp { cpAggFrame = Just af { afReturnPC = rpc } } : rest
  Nothing -> cp : updateNearestAggFrame rpc rest

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
            resVal = derefVar bindings0 <$> IM.lookup resReg regs0
            -- Restore trail to the CP''s snapshot (drop entries added since)
            diff = wsTrailLen s - cpTrailLen cp
            restoredTrail = drop diff (wsTrail s)
            (regs1, bindings1, trail1, trail1Len) = case resVal of
              Just (Unbound vid) ->
                ( IM.insert resReg result regs0
                , IM.insert vid result bindings0
                , TrailEntry vid (IM.lookup vid bindings0) : restoredTrail
                , cpTrailLen cp + 1
                )
              _ -> (regs0, bindings0, restoredTrail, cpTrailLen cp)
        in Just s { wsPC = returnPC
                  , wsRegs = regs1
                  , wsStack = cpStack cp
                  , wsBindings = bindings1
                  , wsTrail = trail1
                  , wsTrailLen = trail1Len
                  , wsHeap = take (cpHeapLen cp) (wsHeap s)
                  , wsHeapLen = cpHeapLen cp
                  , wsCP = cpCP cp
                  , wsCPs = rest
                  , wsCPsLen = wsCPsLen s - 1
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
-- Uses a plain list for visited because depth is bounded (typically <= 10),
-- so O(depth) elem checks are faster than O(log n) Set operations + the
-- per-call Set construction overhead.
nativeCategoryAncestor :: Map.Map String [String] -> String -> String -> Int -> Int -> [String] -> [Int]
nativeCategoryAncestor parents cat root maxDepth depth visited =
  let directParents = fromMaybe [] (Map.lookup cat parents)
      baseHits = [1 | p <- directParents, p == root, p `notElem` visited]
      recHits = if depth >= maxDepth then [] else
        concatMap (\\mid ->
          if mid `elem` visited then []
          else map (+1) $ nativeCategoryAncestor parents mid root maxDepth (depth+1) (mid : visited)
        ) directParents
  in baseHits ++ recHits

-- | Execute a foreign predicate call. Computes all results natively,
-- returns first result with CPs for the rest.
-- | Indexed fact dispatch for 2-arg facts via BuiltinState CP.
-- O(1) Map lookup, first match returned, FactRetry CP for the rest.
callIndexedFact2 :: String -> WamState -> Maybe WamState
callIndexedFact2 pred s =
  let basePred = takeWhile (/= ''/'') pred
      retPC = wsCP s
  in case Map.lookup basePred (wsForeignFacts s) of
    Nothing -> Nothing
    Just factIndex ->
      let a1 = derefVar (wsBindings s) $ fromMaybe (Atom "") (IM.lookup 1 (wsRegs s))
          a2 = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup 2 (wsRegs s))
      in case a1 of
        Atom key -> case Map.lookup key factIndex of
          Just (v:rest) -> case a2 of
            Unbound vid ->
              let newRegs = IM.insert 2 (Atom v) (wsRegs s)
                  newBindings = IM.insert vid (Atom v) (wsBindings s)
                  newTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                  newCPs = case rest of
                    [] -> wsCPs s  -- single match, no CP
                    _  -> ChoicePoint
                            { cpNextPC = retPC, cpRegs = wsRegs s, cpStack = wsStack s
                            , cpCP = wsCP s, cpTrailLen = wsTrailLen s
                            , cpHeapLen = wsHeapLen s, cpBindings = wsBindings s
                            , cpCutBar = wsCutBar s, cpAggFrame = Nothing
                            , cpBuiltin = Just (FactRetry vid rest retPC)
                            } : wsCPs s
                  newCPsLen = case rest of { [] -> wsCPsLen s; _ -> wsCPsLen s + 1 }
              in Just (s { wsPC = retPC, wsRegs = newRegs, wsBindings = newBindings
                         , wsTrail = newTrail, wsTrailLen = wsTrailLen s + 1
                         , wsCPs = newCPs, wsCPsLen = newCPsLen })
            Atom existing ->
              if existing == v then Just (s { wsPC = retPC })
              else case filter (== existing) rest of
                (_:_) -> Just (s { wsPC = retPC })
                [] -> Nothing
            _ -> Nothing
          _ -> Nothing
        _ -> Nothing

executeForeign :: String -> WamState -> Maybe WamState
executeForeign "category_ancestor/4" s =
  let cat = derefVar (wsBindings s) $ fromMaybe (Atom "") (IM.lookup 1 (wsRegs s))
      root = derefVar (wsBindings s) $ fromMaybe (Atom "") (IM.lookup 2 (wsRegs s))
      visited = derefVar (wsBindings s) $ fromMaybe (VList []) (IM.lookup 4 (wsRegs s))
      maxD = fromMaybe 10 $ Map.lookup "max_depth" (wsForeignConfig s)
      parents = fromMaybe Map.empty $ Map.lookup "category_parent" (wsForeignFacts s)
  in case (cat, root, visited) of
    (Atom catS, Atom rootS, VList visitedVals) ->
      let visitedStrs = [v | Atom v <- visitedVals]
          hops = nativeCategoryAncestor parents catS rootS maxD (length visitedStrs) visitedStrs
          retPC = wsCP s
          hopsReg = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup 3 (wsRegs s))
          bindHop hopVal =
            case hopsReg of
              Unbound vid ->
                s { wsPC = retPC
                  , wsRegs = IM.insert 3 (Integer (fromIntegral hopVal)) (wsRegs s)
                  , wsBindings = IM.insert vid (Integer (fromIntegral hopVal)) (wsBindings s)
                  , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                  , wsTrailLen = wsTrailLen s + 1 }
              _ -> s { wsPC = retPC, wsRegs = IM.insert 3 (Integer (fromIntegral hopVal)) (wsRegs s) }
      in case hops of
        [] -> Nothing
        [h] -> Just (bindHop h)   -- single result, no CPs
        (h:restHops) ->
          let s1 = bindHop h
              -- Single HopsRetry CP for all remaining matches (self-popping)
              hopsVar = case hopsReg of { Unbound v -> v; _ -> -1 }
              cp = ChoicePoint
                { cpNextPC = retPC, cpRegs = wsRegs s, cpStack = wsStack s
                , cpCP = wsCP s, cpTrailLen = wsTrailLen s
                , cpHeapLen = wsHeapLen s, cpBindings = wsBindings s
                , cpCutBar = wsCutBar s, cpAggFrame = Nothing
                , cpBuiltin = Just (HopsRetry hopsVar (map fromIntegral restHops) retPC)
                }
          in Just (s1 { wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })
    _ -> Nothing
executeForeign _ _ = Nothing

-- | Unify two values, binding unbound variables.
unifyVal :: Value -> Value -> WamState -> Maybe WamState
unifyVal (Unbound vid) val s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = IM.insert vid val (wsBindings s)
          , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
          , wsTrailLen = wsTrailLen s + 1
          })
unifyVal val (Unbound vid) s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = IM.insert vid val (wsBindings s)
          , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
          , wsTrailLen = wsTrailLen s + 1
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
  let val = derefVar (wsBindings s) <$> IM.lookup ai (wsRegs s)
  in case val of
    Just v | v == c -> Just (s { wsPC = wsPC s + 1 })
    Just (Unbound vid) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai c (wsRegs s)
              , wsBindings = IM.insert vid c (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              })
    _ -> Nothing

step s (GetVariable xn ai) =
  case IM.lookup ai (wsRegs s) of
    Just val -> let dv = derefVar (wsBindings s) val
                in Just ((putReg xn dv s) { wsPC = wsPC s + 1 })
    Nothing -> Nothing

step s (GetValue xn ai) =
  let va = derefVar (wsBindings s) <$> IM.lookup ai (wsRegs s)
      vx = getReg xn s
  in case (va, vx) of
    (Just a, Just x) | a == x -> Just (s { wsPC = wsPC s + 1 })
    (Just (Unbound vid), Just x) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai x (wsRegs s)
              , wsBindings = IM.insert vid x (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              })
    _ -> Nothing

step s (PutConstant c ai) =
  Just (s { wsPC = wsPC s + 1, wsRegs = IM.insert ai c (wsRegs s) })

step s (PutVariable xn ai) =
  let vid = wsVarCounter s
      var = Unbound vid
      s1 = putReg xn var s
  in Just (s1 { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai var (wsRegs s1)
              , wsVarCounter = vid + 1
              })

step s (PutValue xn ai) =
  case getReg xn s of
    Just val -> Just (s { wsPC = wsPC s + 1, wsRegs = IM.insert ai val (wsRegs s) })
    Nothing -> Nothing

step s (PutStructure fn ai arity) =
  Just (s { wsPC = wsPC s + 1
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

-- Fast path: call has been pre-resolved to a target PC at load time.
-- No string lookup, no foreign/indexed dispatch — just a jump.
step s (CallResolved pc _arity) =
  Just (s { wsPC = pc, wsCP = wsPC s + 1 })

step s (Call pred _arity) =
  let sc = s { wsCP = wsPC s + 1 }
  in case executeForeign pred sc of
    Just sr -> Just sr
    Nothing -> case callIndexedFact2 pred sc of
      Just sr -> Just sr
      Nothing -> case Map.lookup pred (wsLabels s) of
        Just pc -> Just (s { wsPC = pc, wsCP = wsPC s + 1 })
        Nothing -> Nothing

step s Proceed =
  let ret = wsCP s
  in if ret == 0 then Just (s { wsPC = 0 })
     else Just (s { wsPC = ret, wsCP = 0 })

step s Allocate =
  let frame = EnvFrame (wsCP s) IM.empty
  in Just (s { wsPC = wsPC s + 1
             , wsStack = frame : wsStack s
             , wsCutBar = wsCPsLen s
             })

step s Deallocate =
  case wsStack s of
    (EnvFrame oldCP _ : rest) -> Just (s { wsPC = wsPC s + 1, wsStack = rest, wsCP = oldCP })
    _ -> Nothing

step s (TryMeElse label) =
  let nextPC = fromMaybe 0 $ Map.lookup label (wsLabels s)
      cp = ChoicePoint
        { cpNextPC   = nextPC
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = wsTrailLen s
        , cpHeapLen  = wsHeapLen s
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Nothing, cpBuiltin = Nothing
        }
  in Just (s { wsPC = wsPC s + 1, wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })

step s TrustMe =
  case wsCPs s of
    (_ : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
    [] -> Nothing

step s (RetryMeElse label) =
  case wsCPs s of
    (cp : rest) ->
      let nextPC = fromMaybe 0 $ Map.lookup label (wsLabels s)
      in Just (s { wsPC = wsPC s + 1, wsCPs = cp { cpNextPC = nextPC } : rest })
    [] -> Nothing

step s (BuiltinCall "!/0" _) =
  -- Cut: truncate wsCPs to the barrier depth saved at clause Allocate.
  Just (s { wsPC = wsPC s + 1, wsCPs = take (wsCutBar s) (wsCPs s), wsCPsLen = wsCutBar s })

step s (BuiltinCall "is/2" _) =
  let expr = derefVar (wsBindings s) $ fromMaybe (Integer 0) (IM.lookup 2 (wsRegs s))
      result = evalArith (wsBindings s) expr
      lhs = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case (lhs, result) of
    (Just (Unbound vid), Just r) ->
      let val = if fromIntegral (round r :: Int) == r then Integer (round r) else Float r
      in Just (s { wsPC = wsPC s + 1
                 , wsRegs = IM.insert 1 val (wsRegs s)
                 , wsBindings = IM.insert vid val (wsBindings s)
                 , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                 , wsTrailLen = wsTrailLen s + 1
                 })
    (Just (Integer n), Just r) | fromIntegral n == r -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step s (BuiltinCall "length/2" _) =
  let listVal = derefVar (wsBindings s) $ fromMaybe (VList []) (IM.lookup 1 (wsRegs s))
  in case listVal of
    VList items ->
      let len = length items
          lhs = derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s)
      in case lhs of
        Just (Unbound vid) ->
          let val = Integer len
          in Just (s { wsPC = wsPC s + 1
                     , wsRegs = IM.insert 2 val (wsRegs s)
                     , wsBindings = IM.insert vid val (wsBindings s)
                     , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                     , wsTrailLen = wsTrailLen s + 1
                     })
        Just (Integer n) | n == len -> Just (s { wsPC = wsPC s + 1 })
        _ -> Nothing
    _ -> Nothing

step s (BuiltinCall "</2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a < b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step s (BuiltinCall "\\\\+/1" _) =
  let goal = IM.lookup 1 (wsRegs s) >>= derefHeap (wsHeap s)
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
  let val = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case val of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })  -- unbound: skip
    Just v -> case Map.lookup v table of
      Just label -> case Map.lookup label (wsLabels s) of
        Just pc -> Just (s { wsPC = pc })
        Nothing -> Nothing
      Nothing -> Nothing  -- no match: fail
    Nothing -> Nothing

step s (BuiltinCall ">/2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a > b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

-- member/2 builtin: A1=Elem, A2=List. Creates choice points for backtracking.
step s (BuiltinCall "member/2" _) =
  let elem_ = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup 1 (wsRegs s))
      list_ = derefVar (wsBindings s) $ fromMaybe (VList []) (IM.lookup 2 (wsRegs s))
  in case list_ of
    VList (x:_) -> unifyVal elem_ x s
    _ -> Nothing

-- begin_aggregate: push aggregate frame CP, initialize accumulator, continue to goal body
step s (BeginAggregate typ valReg resReg) =
  let cp = ChoicePoint
        { cpNextPC   = wsPC s
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = wsTrailLen s
        , cpHeapLen  = wsHeapLen s
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Just (AggFrame typ valReg resReg 0), cpBuiltin = Nothing
        }
  in Just (s { wsPC = wsPC s + 1
             , wsCPs = cp : wsCPs s
             , wsCPsLen = wsCPsLen s + 1
             , wsAggAccum = []
             })

-- end_aggregate: collect value, store returnPC in nearest aggregate frame, force backtrack
step s (EndAggregate valReg) =
  let val = derefVar (wsBindings s) $ fromMaybe (Integer 0) (getReg valReg s)
      returnPC = wsPC s + 1
      -- Update only the nearest (first) aggregate frame CP, not all CPs
      updatedCPs = updateNearestAggFrame returnPC (wsCPs s)
      s1 = s { wsAggAccum = val : wsAggAccum s, wsCPs = updatedCPs }
  in case backtrackInner returnPC s1 of
    Just s2 -> Just s2
    Nothing -> finalizeAggregate returnPC s1

-- Fallback for unhandled instructions
step _ _ = Nothing'.

run_loop_haskell(Code) :-
    Code = '-- | Main execution loop. Runs until halt (pc=0) or failure.
-- Uses unsafeFetchInstr to avoid Maybe wrapping in the hot path.
-- Bounds are guaranteed by the WAM compiler: PC=0 is halt, otherwise PC
-- always points to a valid instruction within the code array.
run :: WamState -> Maybe WamState
run s
  | wsPC s == 0 = Just s  -- halt
  | otherwise =
      let instr = unsafeFetchInstr (wsPC s) (wsCode s)
      in case step s instr of
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

    % Determine map backend: HashMap (faster) or Map (default fallback)
    option(use_hashmap(UseHM), Options, true),

    % Generate WamTypes.hs
    generate_wam_types_hs(TypesCode0),
    apply_hashmap_rewrite(UseHM, types, TypesCode0, TypesCode),
    directory_file_path(SrcDir, 'WamTypes.hs', TypesPath),
    write_hs_file(TypesPath, TypesCode),

    % Generate WamRuntime.hs
    compile_wam_runtime_to_haskell(Options, RuntimeCode0),
    apply_hashmap_rewrite(UseHM, runtime, RuntimeCode0, RuntimeCode),
    directory_file_path(SrcDir, 'WamRuntime.hs', RuntimePath),
    write_hs_file(RuntimePath, RuntimeCode),

    % Compile predicates
    compile_predicates_to_haskell(Predicates, Options, PredsCode0),
    apply_hashmap_rewrite(UseHM, generic, PredsCode0, PredsCode),
    directory_file_path(SrcDir, 'Predicates.hs', PredsPath),
    write_hs_file(PredsPath, PredsCode),

    % Generate cabal file
    option(module_name(ModName), Options, 'wam-haskell-bench'),
    generate_cabal_file(ModName, UseHM, CabalCode),
    format(atom(CabalFile), '~w.cabal', [ModName]),
    directory_file_path(ProjectDir, CabalFile, CabalPath),
    write_hs_file(CabalPath, CabalCode),

    % Generate Main.hs
    generate_main_hs(Predicates, MainCode0),
    apply_hashmap_rewrite(UseHM, main, MainCode0, MainCode),
    directory_file_path(SrcDir, 'Main.hs', MainPath),
    write_hs_file(MainPath, MainCode),

    format(user_error, '[WAM-Haskell] Generated project at: ~w (hashmap=~w)~n', [ProjectDir, UseHM]).

%% apply_hashmap_rewrite(+UseHM, +Module, +InCode, -OutCode)
%  When UseHM=true, rewrite Data.Map.Strict references to Data.HashMap.Strict.
apply_hashmap_rewrite(false, _, Code, Code) :- !.
apply_hashmap_rewrite(true, Module, Code0, Code) :-
    % Replace import line
    replace_substr(Code0, "import qualified Data.Map.Strict as Map",
                   "import qualified Data.HashMap.Strict as Map", Code1),
    % Replace Map.Map type constructor with Map.HashMap
    replace_substr(Code1, "Map.Map ", "Map.HashMap ", Code2),
    % HashMap has no toAscList — use toList instead (loses ordering, but
    % the only use site builds a SwitchOnConstant which doesn''t need it)
    replace_substr(Code2, "Map.toAscList", "Map.toList", Code3),
    % For WamTypes, add Hashable instance for Value (needed for HashMap keys)
    (   Module == types
    ->  replace_substr(Code3,
            "module WamTypes where\n\nimport qualified Data.HashMap.Strict as Map",
            "{-# LANGUAGE DeriveGeneric #-}\nmodule WamTypes where\n\nimport qualified Data.HashMap.Strict as Map\nimport Data.Hashable (Hashable)\nimport GHC.Generics (Generic)",
            Code4),
        replace_substr(Code4,
            "deriving (Eq, Ord, Show)",
            "deriving (Eq, Ord, Show, Generic)\ninstance Hashable Value",
            Code)
    ;   Code = Code3
    ).

%% replace_substr(+Str, +From, +To, -Result)
%  Replace all occurrences of From with To in Str.
replace_substr(Str, From, To, Result) :-
    atom_string(Str, S),
    atom_string(From, FS),
    atom_string(To, TS),
    split_string(S, "", "", [_]),  % normalize
    re_split_replace(S, FS, TS, R),
    atom_string(Result, R).

re_split_replace(S, From, To, Result) :-
    (   sub_string(S, B, L, A, From)
    ->  sub_string(S, 0, B, _, Before),
        sub_string(S, _, A, 0, After),
        re_split_replace(After, From, To, AfterR),
        atom_string(BeforeA, Before),
        atom_string(ToA, To),
        atom_string(AfterRA, AfterR),
        atomic_list_concat([BeforeA, ToA, AfterRA], R),
        atom_string(R, Result)
    ;   Result = S
    ).

%% generate_main_hs(+Predicates, -Code)
%  Generates Main.hs — a benchmark driver for effective-distance.
%  Loads facts from TSV, runs category_ancestor queries, outputs TSV.
generate_main_hs(_Predicates, Code) :-
    Code = '{-# LANGUAGE BangPatterns #-}
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
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
                factInstrs = [GetConstant (Atom a) 1, GetConstant (Atom b) 2, Proceed]
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
        factCode = concatMap (\\(i, v) -> [GetConstant (Atom v) 1, Proceed]) (zip [0..] vals)
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

        mergedCodeRaw = allCode ++ cpCode ++ acCode ++ rcCode
        mergedLabels = Map.union allLabels
                     $ Map.fromList (cpLabels ++ acLabels ++ rcLabels)
        -- Pre-resolve Call instructions to CallResolved at startup so the
        -- hot path skips wsLabels string lookups. Foreign/indexed predicates
        -- are kept as Call so runtime dispatch (executeForeign/callIndexedFact2)
        -- still applies.
        foreignPreds = ["category_ancestor/4"]
        mergedCode = resolveCallInstrs mergedLabels foreignPreds mergedCodeRaw

    -- Collect seed categories
    let seedCats = nub $ sort $ map snd articleCategories
        root = head roots
        n = 5.0 :: Double
        negN = -n

    t2 <- getCurrentTime

    -- For each seed, run category_ancestor(Cat, Root, Hops, [Cat])
    -- IMPORTANT: use mapM in IO with strict bang patterns to force actual
    -- computation BEFORE timing endpoint, otherwise lazy evaluation will
    -- defer the work until output and the timing will be artificially low.
    seedResultsForced <- mapM (\\cat -> do
        let hopsVarId = 1000000  -- reserved query var ID, won''t collide with wsVarCounter
            s0 = (emptyState mergedCode mergedLabels)
                { wsPC = fromMaybe 1 $ Map.lookup "category_ancestor/4" mergedLabels
                , wsRegs = IM.fromList
                    [ (1, Atom cat)
                    , (2, Atom root)
                    , (3, Unbound hopsVarId)
                    , (4, VList [Atom cat])
                    ]
                , wsCP = 0
                }
            !solutions = collectSolutions s0
            !weightSum = sum [((hops + 1) ** negN) | hops <- solutions]
        return (cat, weightSum)
        ) seedCats

    let !seedWeightSums = Map.fromList [(cat, ws) | (cat, ws) <- seedResultsForced, ws > 0]
        !forcedSize = Map.size seedWeightSums  -- force the map

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

-- | Collect all Hops solutions by reading A3 (hops register) after each run.
-- A3 holds the hops result from category_ancestor/4.
collectSolutions :: WamState -> [Double]
collectSolutions s0 =
    case run s0 of
      Nothing -> []
      Just s1 ->
        let hopsVal = case IM.lookup 3 (wsRegs s1) of
              Just v -> extractDouble (derefVar (wsBindings s1) v)
              Nothing -> Nothing
            hops = fromMaybe 0 hopsVal
            rest = case backtrack s1 of
              Just s2 -> collectSolutions s2
              Nothing -> []
        in case hopsVal of
          Just _ -> hops : rest
          Nothing -> rest

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
import qualified Data.IntMap.Strict as IM
import Data.Array (Array, listArray, (!), bounds)
import qualified Data.Set as Set
import Data.List (isPrefixOf, foldl'')
import Data.Maybe (fromMaybe)
import WamTypes

~w

~w

~w

-- | Dereference an Unbound variable through the binding table.
derefVar :: IM.IntMap Value -> Value -> Value
derefVar bindings (Unbound vid) =
  case IM.lookup vid bindings of
    Just val -> derefVar bindings val
    Nothing  -> Unbound vid
derefVar _ v = v

-- | Evaluate arithmetic expression.
evalArith :: IM.IntMap Value -> Value -> Maybe Double
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

-- | Get register value. Y-registers (id >= 200) come from the env frame.
getReg :: Int -> WamState -> Maybe Value
getReg rid s
  | rid >= 200 = findYReg rid (wsStack s)
  | otherwise = derefVar (wsBindings s) <$> IM.lookup rid (wsRegs s)
  where
    findYReg _ [] = Nothing
    findYReg r (EnvFrame _ yregs : _) = derefVar (wsBindings s) <$> IM.lookup r yregs
    findYReg r (_ : rest) = findYReg r rest

-- | Set register value. Y-registers go to the topmost env frame.
putReg :: Int -> Value -> WamState -> WamState
putReg rid val s
  | rid >= 200 = s { wsStack = updateTopEnv rid val (wsStack s) }
  | otherwise = s { wsRegs = IM.insert rid val (wsRegs s) }
  where
    updateTopEnv _ _ [] = []
    updateTopEnv r v (EnvFrame cp yregs : rest) =
      EnvFrame cp (IM.insert r v yregs) : rest
    updateTopEnv r v (x : rest) = x : updateTopEnv r v rest

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
    -- Cons to front (O(1)) and reverse only on finalize. Track count via list length
    -- but only when finalizing — args grows from 0 to arity, max arity is small.
    let args'' = val : args
    in if length args'' == arity
       then Just (s { wsPC = wsPC s + 1
                    , wsRegs = IM.insert ai (Str fn (reverse args'')) (wsRegs s)
                    , wsBuilder = NoBuilder
                    })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildStruct fn ai arity args''
                    })
  BuildList ai args ->
    -- BuildList always has exactly 2 args [head, tail]
    let args'' = val : args
    in if length args'' == 2
       then let [tl, hd] = args''   -- reversed because we cons-built
                list = case tl of
                  VList items -> VList (hd : items)
                  Atom "[]"  -> VList [hd]
                  _           -> VList [hd, tl]
            in Just (s { wsPC = wsPC s + 1
                       , wsRegs = IM.insert ai list (wsRegs s)
                       , wsBuilder = NoBuilder
                       })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildList ai args''
                    })
  NoBuilder ->
    -- No builder active, just push to heap (fallback)
    Just (s { wsPC = wsPC s + 1, wsHeap = wsHeap s ++ [val], wsHeapLen = wsHeapLen s + 1 })

-- | Lookup a label in the label map.
lookupLabel :: String -> WamState -> Int
lookupLabel label s = fromMaybe 0 $ Map.lookup label (wsLabels s)

-- | Fetch instruction at PC (1-indexed). Bounds-checked, returns Maybe.
fetchInstr :: Int -> Array Int Instruction -> Maybe Instruction
fetchInstr pc code
  | let (lo, hi) = bounds code in pc < lo || pc > hi = Nothing
  | otherwise = Just (code ! pc)

-- | Unsafe fetch — no bounds check, no Maybe wrapping. Use only when the
-- caller can prove PC is in bounds (the run loop handles PC=0 as halt
-- separately, and a well-formed WAM program never jumps out of bounds).
{-# INLINE unsafeFetchInstr #-}
unsafeFetchInstr :: Int -> Array Int Instruction -> Instruction
unsafeFetchInstr pc code = code ! pc

-- | Resolve Call instructions to CallResolved by looking up labels once at
-- project load time. Calls to predicates that have foreign/indexed handlers
-- (e.g., category_ancestor/4 via FFI, category_parent/2 via indexed facts)
-- are LEFT as Call so the runtime dispatch chain still applies. Only Call
-- targets that have a matching label and no foreign/indexed override are
-- pre-resolved to a direct PC jump.
resolveCallInstrs :: Map.Map String Int -> [String] -> [Instruction] -> [Instruction]
resolveCallInstrs labels foreignPreds = map resolve
  where
    resolve (Call pred arity)
      | pred `elem` foreignPreds = Call pred arity   -- keep dispatch
      | otherwise = case Map.lookup pred labels of
          Just pc -> CallResolved pc arity
          Nothing -> Call pred arity   -- unresolvable, leave as-is
    resolve i = i
', [StepCode, BacktrackCode, RunCode]).

%% generate_wam_types_hs(-Code)
generate_wam_types_hs(Code) :-
    Code = 'module WamTypes where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (Array, listArray, (!), bounds)

data Value = Atom String
           | Integer Int
           | Float Double
           | VList [Value]
           | Str String [Value]
           | Unbound !Int   -- variable ID (interned via wsVarCounter)
           | Ref Int
           deriving (Eq, Ord, Show)

data EnvFrame = EnvFrame !Int !(IM.IntMap Value)   -- saved CP + Y-regs (IntMap)
              deriving (Show)

data TrailEntry = TrailEntry !Int !(Maybe Value)   -- variable ID, old value
                deriving (Show)

data ChoicePoint = ChoicePoint
  { cpNextPC   :: !Int
  , cpRegs     :: !(IM.IntMap Value)    -- IntMap (registers)
  , cpStack    :: ![EnvFrame]
  , cpCP       :: !Int
  , cpTrailLen :: !Int
  , cpHeapLen  :: !Int
  , cpBindings :: !(IM.IntMap Value)    -- IntMap (variable bindings)
  , cpCutBar   :: !Int
  , cpAggFrame :: !(Maybe AggFrame)
  , cpBuiltin  :: !(Maybe BuiltinState)
  } deriving (Show)

-- | Builtin state for choice points that need custom retry logic.
data BuiltinState
  = FactRetry !Int ![String] !Int  -- variable ID, remaining values, returnPC
  | HopsRetry !Int ![Int] !Int     -- variable ID, remaining Hops values, returnPC
  deriving (Show)

-- | Aggregate frame for begin_aggregate/end_aggregate.
data AggFrame = AggFrame
  { afType      :: !String   -- "sum", "count", "collect", etc.
  , afValueReg  :: !Int      -- register ID holding value per solution
  , afResultReg :: !Int      -- register ID for final result
  , afReturnPC  :: !Int      -- PC after end_aggregate
  } deriving (Show)

-- | Builder for PutStructure/PutList + SetValue/SetConstant sequences.
data Builder = BuildStruct !String !Int !Int ![Value]  -- functor, target reg ID, arity, collected args
             | BuildList !Int ![Value]                  -- target reg ID, collected [head, tail]
             | NoBuilder
             deriving (Show)

data WamState = WamState
  { wsPC       :: !Int
  , wsRegs     :: !(IM.IntMap Value)
  , wsStack    :: ![EnvFrame]
  , wsHeap     :: ![Value]
  , wsHeapLen  :: !Int                          -- cached length(wsHeap), avoids O(n) length calls
  , wsTrail    :: ![TrailEntry]
  , wsTrailLen :: !Int                          -- cached length(wsTrail)
  , wsCP       :: !Int
  , wsCPs      :: ![ChoicePoint]
  , wsCPsLen   :: !Int                          -- cached length(wsCPs) for cut barrier
  , wsBindings :: !(IM.IntMap Value)
  , wsCutBar   :: !Int
  , wsCode     :: !(Array Int Instruction)
  , wsLabels   :: !(Map.Map String Int)
  , wsBuilder  :: !Builder
  , wsVarCounter :: !Int
  , wsAggAccum :: ![Value]
  , wsForeignFacts :: !(Map.Map String (Map.Map String [String]))
  , wsForeignConfig :: !(Map.Map String Int)
  } deriving (Show)

-- | Instruction type for the WAM.
-- | Register IDs are pre-interned at compile time as Ints to avoid string
-- hashing on register access. Encoding:
--   A1-A99: 1-99
--   X1-X99: 101-199
--   Y1-Y99: 201-299
type RegId = Int

data Instruction
  = GetConstant Value !RegId
  | GetVariable !RegId !RegId
  | GetValue !RegId !RegId
  | PutConstant Value !RegId
  | PutVariable !RegId !RegId
  | PutValue !RegId !RegId
  | PutStructure String !RegId !Int   -- functor, target reg, arity (pre-parsed)
  | PutList !RegId
  | SetValue !RegId
  | SetConstant Value
  | Allocate
  | Deallocate
  | Call String !Int                  -- pre-resolution form (string-keyed)
  | CallResolved !Int !Int            -- post-resolution: target PC + arity
  | Execute String
  | Proceed
  | BuiltinCall String !Int
  | TryMeElse String
  | RetryMeElse String
  | TrustMe
  | SwitchOnConstant (Map.Map Value String)   -- pre-built Map for O(log n) dispatch
  | BeginAggregate String !RegId !RegId   -- type, valueReg, resultReg
  | EndAggregate !RegId                   -- valueReg
  deriving (Show, Eq)

-- | Create initial empty state.
emptyState :: [Instruction] -> Map.Map String Int -> WamState
emptyState codeList labels = let n = length codeList; code = listArray (1, n) codeList in WamState
  { wsPC       = 1
  , wsRegs     = IM.empty
  , wsStack    = []
  , wsHeap     = []
  , wsHeapLen  = 0
  , wsTrail    = []
  , wsTrailLen = 0
  , wsCP       = 0
  , wsCPs      = []
  , wsCPsLen   = 0
  , wsBindings = IM.empty
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

%% generate_cabal_file(+Name, +UseHM, -Code)
generate_cabal_file(Name, UseHM, Code) :-
    (   UseHM == true
    ->  Deps = "base >= 4.12, containers >= 0.6, array, time >= 1.8, unordered-containers >= 0.2, hashable >= 1.2"
    ;   Deps = "base >= 4.12, containers >= 0.6, array, time >= 1.8"
    ),
    format(string(Code),
'cabal-version: 2.4
name:          ~w
version:       0.1.0.0
build-type:    Simple

executable ~w
  main-is:          Main.hs
  hs-source-dirs:   src
  other-modules:    WamTypes, WamRuntime, Predicates
  build-depends:    ~w
  default-language: Haskell2010
  ghc-options:      -O2
', [Name, Name, Deps]).

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

%% reg_name_to_int(+RegName, -Int)
%  Encode register name string to integer ID for IntMap-based register storage.
%  A1-A99 -> 1-99, X1-X99 -> 101-199, Y1-Y99 -> 201-299.
reg_name_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Bank),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Bank == 'A' -> Int = Num
    ;   Bank == 'X' -> Int is Num + 100
    ;   Bank == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% wam_instr_to_haskell(+Parts, -HaskellExpr)
%  Converts parsed WAM instruction parts to a Haskell Instruction constructor.
wam_instr_to_haskell(["get_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetConstant (~w) ~w', [HsVal, AiI]).
wam_instr_to_haskell(["get_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetVariable ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["get_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetValue ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutConstant (~w) ~w', [HsVal, AiI]).
wam_instr_to_haskell(["put_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutVariable ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutValue ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_structure", FN, Ai], Hs) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    parse_functor_arity(CFN, Arity),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutStructure "~w" ~w ~w', [CFN, AiI, Arity]).

%% parse_functor_arity(+FunctorString, -Arity)
%  Extract the arity from "name/N" format. Defaults to 0 if no slash.
parse_functor_arity(FN, Arity) :-
    atom_string(FNA, FN),
    (   sub_atom(FNA, Before, 1, _, '/'),
        After is Before + 1,
        sub_atom(FNA, After, _, 0, ArityStr),
        atom_number(ArityStr, Arity)
    ->  true
    ;   Arity = 0
    ).
wam_instr_to_haskell(["put_list", Ai], Hs) :-
    clean_comma(Ai, CAi), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutList ~w', [AiI]).
wam_instr_to_haskell(["set_value", Xn], Hs) :-
    clean_comma(Xn, CXn), reg_name_to_int(CXn, XnI),
    format(string(Hs), 'SetValue ~w', [XnI]).
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
    reg_name_to_int(CV, VI), reg_name_to_int(CR, RI),
    format(string(Hs), 'BeginAggregate "~w" ~w ~w', [CT, VI, RI]).
wam_instr_to_haskell(["end_aggregate", ValReg], Hs) :-
    clean_comma(ValReg, CV),
    reg_name_to_int(CV, VI),
    format(string(Hs), 'EndAggregate ~w', [VI]).
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
