{-# LANGUAGE BangPatterns #-}
-- | \+ member(X, L) lowering microbenchmark.
--
-- Compares two predicates with IDENTICAL Prolog source, distinguished
-- only by a `:- mode/1` declaration:
--
--     :- mode bench_notmember_lowered(+, +).
--     bench_notmember_lowered(X, V) :- \+ member(X, V).
--
--     bench_notmember_unlowered(X, V) :- \+ member(X, V).
--
-- The mode declaration triggers the NotMemberList lowering (X bound,
-- V bound -> emit `not_member_list XReg, VReg`). The unannotated copy
-- compiles to put_structure member/2 + builtin_call \+/1 (which then
-- fast-paths to a list walk inside the runtime).
--
-- Both paths walk the same list at runtime; the savings are:
--   * No heap allocation for the goal term `member(X, V)`.
--   * No put_structure + 2 set_value instruction dispatches.
--   * No builtin_call \+/1 dispatch (pattern match on Str fnId).
-- That is ~4 instruction dispatches saved per call, plus one heap
-- allocation.
--
-- Output: same protocol as wam_term_construction_bench's Bench.hs.

module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (listArray)
import Data.Maybe (fromMaybe)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import System.Environment (getArgs)
import System.IO (hFlush, stdout)
import WamTypes
import WamRuntime (run)
import Predicates (allCode, allLabels, compileTimeAtomTable)

mkCtx :: WamContext
mkCtx =
  let n = length allCode
  in WamContext
        { wcCode = listArray (1, n) allCode
        , wcLabels = allLabels
        , wcForeignFacts = Map.empty
        , wcForeignConfig = Map.empty
        , wcLoweredPredicates = Map.empty
        , wcInternTable = compileTimeAtomTable
        , wcFfiFacts = Map.empty
        , wcFfiWeightedFacts = Map.empty
        , wcInlineFacts = Map.empty
        , wcFactSources = Map.empty
        , wcEdgeLookups = Map.empty
        }

-- | bench_notmember_*(X, V) where X = Integer k (varies per iteration
-- to defeat GHC constant-folding) and V is a VList of N atoms.
-- X is never an atom in V so \+ member always succeeds; the runtime
-- walks all N items.
mkInitState :: Int -> Int -> Value -> WamState
mkInitState !pc !k !vList = WamState
  { wsPC = pc
  , wsRegs = IM.fromList
      [ (1, Integer (fromIntegral k))
      , (2, vList)
      ]
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
  , wsVarCounter = 1
  , wsAggAccum = []
  }

benchN :: Int -> WamContext -> Int -> Value -> Int
benchN !n !ctx !pc !vList = go 0 n
  where
    go !acc 0 = acc
    go !acc !k =
      let !s0  = mkInitState pc k vList
          add  = case run ctx s0 of
                   Just _  -> 1
                   Nothing -> 0
      in go (acc + add) (k - 1)

timeIt :: String -> Int -> WamContext -> Int -> Value -> IO ()
timeIt !caseName !n !ctx !pc !vList = do
  start <- getCurrentTime
  let !ok = benchN n ctx pc vList
  ok `seq` return ()
  end <- getCurrentTime
  let elapsed = realToFrac (diffUTCTime end start) :: Double
  putStrLn $
       "RESULT " ++ caseName
    ++ "  " ++ show n
    ++ "  " ++ show elapsed
    ++ "s  " ++ show ok ++ "/" ++ show n
  hFlush stdout

main :: IO ()
main = do
  args <- getArgs
  let n = case args of
            (x : _) -> read x :: Int
            []      -> 50000
  -- Build a 50-item visited list of distinct atoms. The X probe is an
  -- Integer that varies per iteration (k), so it can never be ==
  -- to any of the atoms in vList — \+ member always succeeds and the
  -- runtime walks the whole list.
  let listLen = 50
      tbl0    = compileTimeAtomTable
      (vAtomIds, _tblF) = foldr step ([], tbl0) [0 .. listLen - 1 :: Int]
        where step i (acc, t) =
                let (aid, t') = internAtom t ("v" ++ show i)
                in (aid : acc, t')
      vList = VList (map Atom vAtomIds)
      !ctx  = mkCtx
      pcLow   = fromMaybe 1 $ Map.lookup "bench_notmember_lowered/2"   allLabels
      pcUnlow = fromMaybe 1 $ Map.lookup "bench_notmember_unlowered/2" allLabels

  putStrLn $ "[INFO] N=" ++ show n
          ++ "  list_len=" ++ show listLen
          ++ "  bench_notmember_lowered/2 PC=" ++ show pcLow
          ++ "  bench_notmember_unlowered/2 PC=" ++ show pcUnlow
  hFlush stdout

  -- Warm-up.
  let !w1 = benchN 500 ctx pcLow   vList
      !w2 = benchN 500 ctx pcUnlow vList
  w1 `seq` w2 `seq` return ()

  -- Two trial blocks, alternating order.
  timeIt "lowered  " n ctx pcLow   vList
  timeIt "unlowered" n ctx pcUnlow vList
  timeIt "unlowered" n ctx pcUnlow vList
  timeIt "lowered  " n ctx pcLow   vList
