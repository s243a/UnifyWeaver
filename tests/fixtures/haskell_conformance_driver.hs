{-# LANGUAGE BangPatterns #-}
-- Cross-target conformance driver for the WAM Haskell backend.
--
-- Runs a single ground 0-arity wrapper predicate (e.g. "ctw_1/0",
-- synthesised by the conformance harness as `ctw_N :- pred(args).`) and
-- prints "true" or "false". Because the wrapper bakes every atom into the
-- compiled instruction stream at codegen time, there is NO runtime atom
-- parsing here — the boolean is just whether `run` finds a solution.
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (listArray)
import System.Environment (getArgs)
import WamTypes
import WamRuntime
import Predicates

main :: IO ()
main = do
    args <- getArgs
    let key = case args of (k:_) -> k; _ -> ""
        code = allCode
        labels = allLabels
        n = length code
        codeArr = listArray (1, n) code
        ctx = WamContext
          { wcCode = codeArr
          , wcLabels = labels
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
    case Map.lookup key labels of
      Nothing -> putStrLn "false"
      Just pc -> do
        let s0 = emptyState { wsPC = pc, wsRegs = IM.empty, wsCP = 0 }
        case run ctx s0 of
          Just _  -> putStrLn "true"
          Nothing -> putStrLn "false"
