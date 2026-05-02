{-# LANGUAGE BangPatterns #-}
-- | put_structure_dyn end-to-end runtime smoke.
--
-- Imports the actual generated WamRuntime + WamTypes (compiled by
-- cabal under the same build configuration as a real benchmark
-- project) and drives the runtime `step` function on a real
-- `PutStructureDyn` instruction. This exercises the runtime
-- contract documented at wam_haskell_target.pl:1890-1898:
--
--   * Atom name + non-negative Integer arity =>
--     wsBuilder = BuildStruct fnId targetReg arity []
--     and wsPC advances by 1.
--   * Anything else (non-atom name, negative arity, missing reg,
--     non-integer arity) => Nothing (signals backtrack).
--   * Source registers may hold Unbound variables that resolve to
--     valid values via wsBindings — derefVar must be applied.
--
-- This file is dropped into the generated project's src/ directory
-- by the Prolog smoke driver alongside the regular Main.hs (which
-- the smoke does not exercise — only WamTypes + WamRuntime).
--
-- Output protocol:
--   One PASS/FAIL line per case, then a summary line
--     RESULT <ok>/<total>
--   that the Prolog driver greps. Any deviation from N/N is a fail.
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (listArray)
import WamTypes
import WamRuntime (step)

mkCtx :: InternTable -> WamContext
mkCtx tbl = WamContext
  { wcCode = listArray (0, -1) []
  , wcLabels = Map.empty
  , wcForeignFacts = Map.empty
  , wcForeignConfig = Map.empty
  , wcLoweredPredicates = Map.empty
  , wcInternTable = tbl
  , wcFfiFacts = Map.empty
  , wcFfiWeightedFacts = Map.empty
  , wcInlineFacts = Map.empty
  , wcFactSources = Map.empty
  , wcEdgeLookups = Map.empty
  }

mkState :: [(Int, Value)] -> WamState
mkState regs = WamState
  { wsPC = 10
  , wsRegs = IM.fromList regs
  , wsStack = []
  , wsHeap = []
  , wsHeapLen = 0
  , wsTrail = []
  , wsTrailLen = 0
  , wsCP = 0
  , wsCPs = []
  , wsCPsLen = 0
  , wsBindings = IM.empty
  , wsCutBar = 0
  , wsBuilder = NoBuilder
  , wsVarCounter = 1000
  , wsAggAccum = []
  }

regA1, regA2, regTarget :: Int
regA1     = 1     -- A1
regA2     = 2     -- A2
regTarget = 104   -- X4 (101 + 3)

check :: String -> Bool -> IO Bool
check name ok = do
  putStrLn ((if ok then "PASS " else "FAIL ") ++ name)
  return ok

main :: IO ()
main = do
  let (fooId, tbl) = internAtom emptyInternTable "foo"
      ctx   = mkCtx tbl
      instr = PutStructureDyn regA1 regA2 regTarget

  r1 <- check "atom_name + nonneg_arity => BuildStruct" $
    case step ctx (mkState [(regA1, Atom fooId), (regA2, Integer 2)]) instr of
      Just s' ->
        wsPC s' == 11 &&
        case wsBuilder s' of
          BuildStruct fn target arity args ->
            fn == fooId && target == regTarget && arity == 2 && null args
          _ -> False
      Nothing -> False

  r2 <- check "atom_name + zero_arity => BuildStruct (zero arity allowed)" $
    case step ctx (mkState [(regA1, Atom fooId), (regA2, Integer 0)]) instr of
      Just s' -> case wsBuilder s' of
        BuildStruct fn target arity args ->
          fn == fooId && target == regTarget && arity == 0 && null args
        _ -> False
      Nothing -> False

  r3 <- check "negative_arity => Nothing (backtrack)" $
    case step ctx (mkState [(regA1, Atom fooId), (regA2, Integer (-1))]) instr of
      Just _  -> False
      Nothing -> True

  r4 <- check "non_atom_name => Nothing (backtrack)" $
    case step ctx (mkState [(regA1, Integer 99), (regA2, Integer 1)]) instr of
      Just _  -> False
      Nothing -> True

  r5 <- check "missing_arity_register => Nothing (backtrack)" $
    case step ctx (mkState [(regA1, Atom fooId)]) instr of
      Just _  -> False
      Nothing -> True

  r6 <- check "float_arity => Nothing (backtrack)" $
    case step ctx (mkState [(regA1, Atom fooId), (regA2, Float 2.0)]) instr of
      Just _  -> False
      Nothing -> True

  let st7 = (mkState [(regA1, Unbound 500), (regA2, Integer 3)])
              { wsBindings = IM.fromList [(500, Atom fooId)] }
  r7 <- check "deref_through_binding_chain => BuildStruct" $
    case step ctx st7 instr of
      Just s' -> case wsBuilder s' of
        BuildStruct fn target arity _ ->
          fn == fooId && target == regTarget && arity == 3
        _ -> False
      Nothing -> False

  let oks = length (filter id [r1, r2, r3, r4, r5, r6, r7])
  putStrLn ("RESULT " ++ show oks ++ "/7")
