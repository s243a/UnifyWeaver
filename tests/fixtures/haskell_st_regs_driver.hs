{-# LANGUAGE BangPatterns #-}
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (listArray)
import Data.Maybe (fromMaybe)
import Text.Printf
import WamTypes
import WamRuntime
import Predicates

main :: IO ()
main = do
    let code = allCode
        labels = allLabels
        n = length code
        codeArr = listArray (1, n) code

    let ctx = WamContext
          { wcCode = codeArr
          , wcLabels = labels
          , wcForeignFacts = Map.empty
          , wcForeignConfig = Map.empty
          , wcLoweredPredicates = Map.empty
          , wcInternTable = emptyInternTable
          , wcFfiFacts = Map.empty
          , wcFfiWeightedFacts = Map.empty
          , wcInlineFacts = Map.empty
          , wcFactSources = Map.empty
          , wcEdgeLookups = Map.empty
          }

    -- Query: st_probe(X) — should unify X with 42
    let s0 = emptyState
          { wsPC = fromMaybe 1 $ Map.lookup "st_probe/1" labels
          , wsRegs = IM.fromList [(1, Unbound 999)]
          , wsCP = 0
          }

    putStrLn "Haskell ST Mutable Registers E2E Test"
    putStrLn "======================================"
    putStrLn ""
    printf "Code: %d instructions, %d labels\n" n (Map.size labels)
    printf "Query: st_probe(X) at PC=%d\n\n"
      (fromMaybe 0 $ Map.lookup "st_probe/1" labels :: Int)

    -- Test IntMap path
    let !resultIM = run ctx s0
    putStr "IntMap run:       "
    case resultIM of
      Nothing -> putStrLn "Nothing (no solution)"
      Just s  -> do
        let binding = IM.lookup 999 (wsBindings s)
        printf "PC=%d binding[999]=%s\n" (wsPC s) (show binding)

    -- Test STArray path
    let !resultST = runMutableRegs ctx s0
    putStr "STArray run:      "
    case resultST of
      Nothing -> putStrLn "Nothing (no solution)"
      Just s  -> do
        let binding = IM.lookup 999 (wsBindings s)
        printf "PC=%d binding[999]=%s\n" (wsPC s) (show binding)

    -- Compare results
    putStrLn ""
    case (resultIM, resultST) of
      (Just sIM, Just sST) -> do
        let bIM = IM.lookup 999 (wsBindings sIM)
            bST = IM.lookup 999 (wsBindings sST)
        if bIM == bST
          then printf "RESULT OK (both agree: %s)\n" (show bIM)
          else printf "RESULT MISMATCH (im=%s st=%s)\n" (show bIM) (show bST)
      (Nothing, Nothing) ->
        putStrLn "RESULT OK (both returned Nothing)"
      _ ->
        putStrLn "RESULT MISMATCH (one succeeded, other failed)"
