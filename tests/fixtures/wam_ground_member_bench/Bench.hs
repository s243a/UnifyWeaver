{-# LANGUAGE BangPatterns #-}
-- | \+ member(X, [a, b, c, ...]) ground-list lowering microbenchmark.
--
-- Runs `bench_ground/1` N times in a tight loop. The same Bench.hs is
-- dropped into TWO projects:
--
--   * lowered:   the ground-member lowering is active, so `bench_ground/1`
--                compiles to a single NotMemberConstAtoms dispatch.
--   * unlowered: the lowering is disabled at codegen via
--                `lowering_disabled(ground_member)` in wam_target, so
--                `bench_ground/1` compiles to put_structure cons cells +
--                builtin_call \\+/1 + member walk.
--
-- The probe value is `Integer k` (varies per iteration to defeat GHC
-- constant-folding) and is never an atom in the ground list, so the
-- check always succeeds.
--
-- Output protocol mirrors wam_not_member_bench's Bench.hs:
--   `RESULT <case>  <n>  <elapsed_s>s  <ok>/<n>`

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

mkInitState :: Int -> Int -> WamState
mkInitState !pc !k = WamState
  { wsPC = pc
  , wsRegs = IM.fromList [(1, Integer (fromIntegral k))]
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

benchN :: Int -> WamContext -> Int -> Int
benchN !n !ctx !pc = go 0 n
  where
    go !acc 0 = acc
    go !acc !k =
      let !s0  = mkInitState pc k
          add  = case run ctx s0 of
                   Just _  -> 1
                   Nothing -> 0
      in go (acc + add) (k - 1)

timeIt :: String -> Int -> WamContext -> Int -> IO ()
timeIt !caseName !n !ctx !pc = do
  start <- getCurrentTime
  let !ok = benchN n ctx pc
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
            []      -> 200000
  let !ctx = mkCtx
      pc   = fromMaybe 1 $ Map.lookup "bench_ground/1" allLabels

  putStrLn $ "[INFO] N=" ++ show n ++ "  bench_ground/1 PC=" ++ show pc
  hFlush stdout

  -- Warm-up.
  let !w = benchN 500 ctx pc
  w `seq` return ()

  -- 4 trials so noise is visible.
  timeIt "bench_ground" n ctx pc
  timeIt "bench_ground" n ctx pc
  timeIt "bench_ground" n ctx pc
  timeIt "bench_ground" n ctx pc
