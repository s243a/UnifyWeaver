{-# LANGUAGE BangPatterns #-}
-- | Term-construction lowering microbenchmark.
--
-- Compares two predicates that have IDENTICAL Prolog source but differ
-- only in whether a `:- mode/1` declaration is present:
--
--     :- mode bench_lowered(+, +, -).
--     bench_lowered(Name, Arg, T) :- T =.. [Name, Arg].
--
--     bench_unlowered(Name, Arg, T) :- T =.. [Name, Arg].
--
-- The mode declaration makes the binding-state analyser prove
-- T `unbound` and Name `bound` at the =../2 goal site, which triggers
-- the PutStructureDyn lowering. The unannotated copy goes through the
-- list-build + BuiltinCall "=../2" runtime path.
--
-- This file is dropped into the generated project's src/ directory by
-- the Prolog driver; the cabal file is overridden to point at this
-- file as the executable's main-is.
--
-- Usage:
--   cabal v2-run uw-term-bench -- <N>
--
-- Output protocol (one line per case):
--   RESULT <case>  <n>  <seconds>  <ok>/<n>
-- where <ok> counts iterations that returned a Just final state.

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

-- | Build a read-only WamContext from the project's compiled bytecode.
-- The bytecode is 1-indexed (PC=0 means halt; PC=1 is the first
-- instruction). This matches mkContext in WamTypes.hs.
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

-- | Initial state for `bench_*(foo, k, T)` at the predicate's entry PC.
mkInitState :: Int -> Int -> Int -> WamState
mkInitState !pc !fooId !k = WamState
  { wsPC = pc
  , wsRegs = IM.fromList
      [ (1, Atom fooId)
      , (2, Integer (fromIntegral k))
      , (3, Unbound 0)
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

-- | Run the predicate at PC `pc` exactly `n` times, counting successes.
-- Bang patterns and a strict accumulator force GHC to actually run the
-- loop instead of optimising it away.
benchN :: Int -> WamContext -> Int -> Int -> Int
benchN !n !ctx !pc !fooId = go 0 n
  where
    go !acc 0 = acc
    go !acc !k =
      let !s0   = mkInitState pc fooId k
          add   = case run ctx s0 of
                    Just _  -> 1
                    Nothing -> 0
      in go (acc + add) (k - 1)

timeIt :: String -> Int -> WamContext -> Int -> Int -> IO ()
timeIt !caseName !n !ctx !pc !fooId = do
  start <- getCurrentTime
  let !ok = benchN n ctx pc fooId
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
            []      -> 100000
  let !ctx    = mkCtx
      !fooId  = fst (internAtom compileTimeAtomTable "foo")
      pcLow   = fromMaybe 1 $ Map.lookup "bench_lowered/3" allLabels
      pcUnlow = fromMaybe 1 $ Map.lookup "bench_unlowered/3" allLabels

  putStrLn $ "[INFO] N=" ++ show n
          ++ "  bench_lowered/3 PC=" ++ show pcLow
          ++ "  bench_unlowered/3 PC=" ++ show pcUnlow
  hFlush stdout

  -- Warm-up: one untimed pass each to fault in code pages and JIT hot paths.
  let !w1 = benchN 1000 ctx pcLow   fooId
      !w2 = benchN 1000 ctx pcUnlow fooId
  w1 `seq` w2 `seq` return ()

  -- Alternate the order across two trial blocks to reduce measurement
  -- bias from cache and branch-predictor warming.
  timeIt "lowered  " n ctx pcLow   fooId
  timeIt "unlowered" n ctx pcUnlow fooId
  timeIt "unlowered" n ctx pcUnlow fooId
  timeIt "lowered  " n ctx pcLow   fooId
