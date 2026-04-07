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
backtrack :: WamState -> Maybe WamState
backtrack s = case wsCPs s of
  [] -> Nothing
  (cp : _) ->
    let -- Unwind only binding-table trail entries
        trailLen = cpTrailLen cp
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
      | otherwise = bindings'.

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
  let var = Unbound ("_V" ++ show (wsPC s))
      s1 = putReg xn var s
  in Just (s1 { wsPC = wsPC s + 1, wsRegs = Map.insert ai var (wsRegs s1) })

step s (PutValue xn ai) =
  case getReg xn s of
    Just val -> Just (s { wsPC = wsPC s + 1, wsRegs = Map.insert ai val (wsRegs s) })
    Nothing -> Nothing

step s (Call pred _arity) =
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

-- SwitchOnConstant: dispatch on A1 value
step s (SwitchOnConstant table) =
  let val = derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s)
  in case val of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })  -- unbound: skip
    Just v -> case lookup v table of
      Just label -> case Map.lookup label (wsLabels s) of
        Just pc -> Just (s { wsPC = pc })
        Nothing -> Nothing
      Nothing -> Nothing  -- no match: fail
    Nothing -> Nothing

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
%  Generates a Main.hs that loads predicates and runs a simple query.
generate_main_hs(Predicates, Code) :-
    % Build the combined code and labels from all predicates
    build_predicate_loads(Predicates, LoadCode),
    format(string(Code),
'module Main where

import qualified Data.Map.Strict as Map
import WamTypes
import WamRuntime
import Predicates

main :: IO ()
main = do
    -- Build combined instruction list and label map
~w
    let state0 = emptyState allCode allLabels
    -- Query: set A1 to unbound, run dimension_n/1
    let queryState = state0
          { wsPC = case Map.lookup "dimension_n/1" allLabels of
                     Just pc -> pc
                     Nothing -> 1
          , wsRegs = Map.fromList [("A1", Unbound "X")]
          , wsCP = 0
          }
    case run queryState of
      Just result -> do
        let a1 = derefVar (wsBindings result) <$> Map.lookup "A1" (wsRegs result)
        putStrLn $ "Result: A1 = " ++ show a1
      Nothing -> putStrLn "Query failed"
', [LoadCode]).

build_predicate_loads([], '    let allCode = []\n    let allLabels = Map.empty').
build_predicate_loads([PredIndicator|_], Code) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    format(atom(FuncName), '~w_~w', [Pred, Arity]),
    format(string(Code),
'    let allCode = ~w_code
    let allLabels = ~w_labels', [FuncName, FuncName]).
% TODO: merge multiple predicate code/labels for multi-predicate projects

%% compile_wam_runtime_to_haskell(+Options, -Code)
compile_wam_runtime_to_haskell(_Options, Code) :-
    step_function_haskell(StepCode),
    backtrack_haskell(BacktrackCode),
    run_loop_haskell(RunCode),
    format(string(Code),
'module WamRuntime where

import qualified Data.Map.Strict as Map
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
evalArith bindings (Str op [a, b]) = do
  va <- evalArith bindings (derefVar bindings a)
  vb <- evalArith bindings (derefVar bindings b)
  let bareOp = takeWhile (/= ''/'') op
  case bareOp of
    "+" -> Just (va + vb)
    "-" -> Just (va - vb)
    "*" -> Just (va * vb)
    "**" -> Just (va ** vb)
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

-- | Lookup a label in the label map.
lookupLabel :: String -> WamState -> Int
lookupLabel label s = fromMaybe 0 $ Map.lookup label (wsLabels s)

-- | Fetch instruction at PC (1-indexed).
fetchInstr :: Int -> [Instruction] -> Maybe Instruction
fetchInstr pc code
  | pc < 1 || pc > length code = Nothing
  | otherwise = Just (code !! (pc - 1))
', [StepCode, BacktrackCode, RunCode]).

%% generate_wam_types_hs(-Code)
generate_wam_types_hs(Code) :-
    Code = 'module WamTypes where

import qualified Data.Map.Strict as Map

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
  } deriving (Show)

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
  , wsCode     :: ![Instruction]
  , wsLabels   :: !(Map.Map String Int)
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
  | SwitchOnConstant [(Value, String)]
  deriving (Show, Eq)

-- | Create initial empty state.
emptyState :: [Instruction] -> Map.Map String Int -> WamState
emptyState code labels = WamState
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
  build-depends:    base >= 4.14, containers >= 0.6
  default-language: Haskell2010
  ghc-options:      -O2
', [Name, Name]).

%% compile_predicates_to_haskell(+Predicates, +Options, -Code)
compile_predicates_to_haskell(Predicates, Options, Code) :-
    maplist({Options}/[Pred, PredCode]>>(
        compile_single_predicate_to_haskell(Pred, Options, PredCode)
    ), Predicates, PredCodes),
    atomic_list_concat(PredCodes, '\n\n', AllPreds),
    format(string(Code),
'module Predicates where

import qualified Data.Map.Strict as Map
import WamTypes
import WamRuntime

~w
', [AllPreds]).

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

%% wam_lines_to_haskell(+Lines, +PC, -InstrExprs, -LabelExprs)
%  Parses WAM assembly lines into Haskell Instruction constructor expressions
%  and label (String, Int) pairs.
wam_lines_to_haskell([], _, [], []).
wam_lines_to_haskell([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_haskell(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelExpr), '("~w", ~w)', [LabelName, PC]),
            Labels = [LabelExpr|RestLabels],
            wam_lines_to_haskell(Rest, PC, Instrs, RestLabels)
        ;   wam_instr_to_haskell(CleanParts, HsExpr),
            NPC is PC + 1,
            Instrs = [HsExpr|RestInstrs],
            wam_lines_to_haskell(Rest, NPC, RestInstrs, Labels)
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
    format(string(Hs), 'BuiltinCall "~w" ~w', [COp, Num]).
wam_instr_to_haskell(["try_me_else", Label], Hs) :-
    format(string(Hs), 'TryMeElse "~w"', [Label]).
wam_instr_to_haskell(["trust_me"], "TrustMe").
wam_instr_to_haskell(["retry_me_else", Label], Hs) :-
    format(string(Hs), 'RetryMeElse "~w"', [Label]).
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

%% write_hs_file(+Path, +Content)
write_hs_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
