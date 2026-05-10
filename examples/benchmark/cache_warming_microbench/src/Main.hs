-- SPDX-License-Identifier: MIT
-- Phase M (Measurement) — cache warming microbench driver.
--
-- Usage:
--   cache-warming-microbench <lmdb_dir> [N=10000] [seed=42]
--
-- Reads from a Phase 1 LMDB directory (containing category_child
-- as the dupsort sub-db). Runs M1, M2, M3 and prints a markdown
-- summary table to stdout.

{-# LANGUAGE BangPatterns #-}

module Main where

import Control.DeepSeq (deepseq, force)
import Control.Exception (evaluate)
import Control.Monad (forM_, when)
import Data.Int (Int32)
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.List (sort, sortBy)
import Data.Ord (comparing, Down(..))
import Data.Time.Clock (UTCTime, diffUTCTime, getCurrentTime)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import System.Environment (getArgs)
import System.IO (hFlush, stdout)
import System.Random (mkStdGen, randoms)
import Database.LMDB.Raw

import Patterns (Pattern(..), runPatternBench)
import Rankers (collectAllEdges, pprIteration, streamingHop, semanticSim)
import Crossover (runCrossover, RunResult(..))

main :: IO ()
main = do
    args <- getArgs
    let (lmdbDir, nKeys, seed) = case args of
          [d]       -> (d, 10000 :: Int, 42 :: Int)
          [d, n]    -> (d, read n, 42)
          [d, n, s] -> (d, read n, read s)
          _         -> error "usage: cache-warming-microbench <lmdb_dir> [N] [seed]"

    putStrLn $ "# Phase M microbench"
    putStrLn $ ""
    putStrLn $ "- LMDB: " ++ lmdbDir
    putStrLn $ "- N (keys): " ++ show nKeys
    putStrLn $ "- seed: " ++ show seed
    putStrLn $ ""

    env <- mdb_env_create
    mdb_env_set_mapsize env (16 * 1024 * 1024 * 1024)  -- 16 GB ceiling
    mdb_env_set_maxdbs env 8
    mdb_env_set_maxreaders env 126
    mdb_env_open env lmdbDir [MDB_RDONLY]

    -- Sample N distinct keys from category_child
    sampleKeys <- sampleDistinctKeys env "category_child" nKeys seed
    let nActual = length sampleKeys
    putStrLn $ "Sampled " ++ show nActual ++ " distinct keys from category_child"
    putStrLn ""

    -- =====================================================================
    -- M1: read patterns
    -- =====================================================================
    putStrLn "## M1: read patterns"
    putStrLn ""
    putStrLn "| pattern | wall_ms | edges_collected |"
    putStrLn "|---|---|---|"
    forM_ [PerLookupCursor, SharedInOrder, SharedSorted, SharedShuffled, FullScan] $ \pat -> do
      (ms, n) <- runPatternBench env "category_child" sampleKeys pat
      putStrLn $ "| " ++ show pat ++ " | " ++ fmt ms ++ " | " ++ show n ++ " |"
      hFlush stdout
    putStrLn ""

    -- M1.b: selectivity sweep — scan vs sorted seeks crossover.
    -- Take a fraction of the sampled keys and time both patterns.
    putStrLn "## M1.b: selectivity sweep (sorted seeks vs full scan)"
    putStrLn ""
    putStrLn "| key_fraction | n_keys | sorted_ms | scan_ms | scan/sorted |"
    putStrLn "|---|---|---|---|---|"
    forM_ [0.01, 0.05, 0.1, 0.25, 0.5, 1.0 :: Double] $ \frac -> do
      let k = max 1 $ round (frac * fromIntegral nActual)
          subset = take k sampleKeys
      (ms_sorted, _) <- runPatternBench env "category_child" subset SharedSorted
      (ms_scan, _)   <- runPatternBench env "category_child" subset FullScan
      let ratio = if ms_sorted > 0 then ms_scan / ms_sorted else 0
      putStrLn $ "| " ++ fmt frac ++ " | " ++ show k
              ++ " | " ++ fmt ms_sorted ++ " | " ++ fmt ms_scan
              ++ " | " ++ fmt ratio ++ "× |"
      hFlush stdout
    putStrLn ""

    -- =====================================================================
    -- M2: ranker overhead
    -- =====================================================================
    putStrLn "## M2: ranker overhead"
    putStrLn ""
    putStrLn "Loading full edge list once for shared-input rankers..."
    t0 <- getCurrentTime
    edges <- collectAllEdges env "category_child"
    let !edgeCount = length edges
    edges `deepseq` return ()
    t1 <- getCurrentTime
    putStrLn $ "  edges loaded: " ++ show edgeCount ++ " in " ++ fmt (diffMs t0 t1) ++ " ms"
    putStrLn ""

    -- Initial scores: uniform 1.0 over all source nodes.
    let !sources = IS.fromList [u | (u, _) <- edges]
        !initScores = IM.fromList [(s, 1.0 :: Double) | s <- IS.toList sources]
        !outdeg = IM.fromListWith (+) [(u, 1 :: Int) | (u, _) <- edges]

    putStrLn "| ranker | wall_ms | output_size |"
    putStrLn "|---|---|---|"

    -- PPR one iteration
    tA <- getCurrentTime
    let !ppr = pprIteration initScores outdeg edges
    evaluate (force ppr) >>= \_ -> return ()
    tB <- getCurrentTime
    putStrLn $ "| PPR/Flux 1 iter | " ++ fmt (diffMs tA tB) ++ " | " ++ show (IM.size ppr) ++ " |"
    hFlush stdout

    -- Streaming hop, frontier seeded from first 10 sources
    let !seedFrontier = IM.fromList [(s, 0) | s <- take 10 (IS.toList sources)]
    tC <- getCurrentTime
    let !hops = streamingHop seedFrontier edges
    evaluate (force hops) >>= \_ -> return ()
    tD <- getCurrentTime
    putStrLn $ "| streaming-hop 1 pass | " ++ fmt (diffMs tC tD) ++ " | " ++ show (IM.size hops) ++ " |"
    hFlush stdout

    -- Semantic similarity stub on K=1000 candidates
    let !candidates = take 1000 (IS.toList sources)
    tE <- getCurrentTime
    let !sims = semanticSim candidates
    evaluate (force sims) >>= \_ -> return ()
    tF <- getCurrentTime
    putStrLn $ "| semantic-sim K=1000 D=128 | " ++ fmt (diffMs tE tF) ++ " | " ++ show (length sims) ++ " |"
    putStrLn ""

    -- =====================================================================
    -- M3: warming-vs-JIT crossover
    -- =====================================================================
    putStrLn "## M3: warming-vs-JIT crossover"
    putStrLn ""
    putStrLn $ "BFS-shaped queries (max_hops=3) from random roots."
    putStrLn ""

    -- Warm cache: top-K source nodes by out-degree (a proxy for "hot" edges).
    let warmK = min 5000 (IM.size outdeg)
        topK = take warmK $ sortBy (comparing (Down . snd)) $ IM.toList outdeg
        !warmKeys = IS.fromList (map fst topK)
    -- Pre-load their edges into IntMap.
    let !warmMap = IM.fromListWith (++) [(u, [v]) | (u, v) <- edges, u `IS.member` warmKeys]
    evaluate (force warmMap) >>= \_ -> return ()
    putStrLn $ "Warmed top-" ++ show warmK ++ " source nodes into IntMap."
    putStrLn ""

    putStrLn "| M (queries) | cold_ms | warm_ms | cold_per_q | warm_per_q | speedup |"
    putStrLn "|---|---|---|---|---|---|"
    forM_ [1, 5, 10, 25, 50, 100, 250] $ \m -> do
      let roots = take m (sampleKeys ++ cycle sampleKeys)
      cold <- runCrossover env "category_child" roots 3 Nothing
      warm <- runCrossover env "category_child" roots 3 (Just warmMap)
      let cms = rrTotalMs cold
          wms = rrTotalMs warm
          cpq = cms / fromIntegral m
          wpq = wms / fromIntegral m
          sp  = if wms > 0 then cms / wms else 0
      putStrLn $ "| " ++ show m ++ " | " ++ fmt cms ++ " | " ++ fmt wms
              ++ " | " ++ fmt cpq ++ " | " ++ fmt wpq
              ++ " | " ++ fmt sp ++ "× |"
      hFlush stdout
    putStrLn ""

    mdb_env_close env
    putStrLn "## End"

fmt :: Double -> String
fmt x
  | x >= 100   = show (round x :: Int)
  | x >= 1     = show (fromIntegral (round (x * 10) :: Int) / 10 :: Double)
  | otherwise  = show (fromIntegral (round (x * 1000) :: Int) / 1000 :: Double)

-- | Difference between two UTCTimes in milliseconds.
diffMs :: UTCTime -> UTCTime -> Double
diffMs t0 t1 = realToFrac (diffUTCTime t1 t0) * 1000.0

-- | Sample N distinct keys from a dupsort sub-db using MDB_FIRST + skip stride.
-- Deterministic for a given (seed, N, db_size).
sampleDistinctKeys :: MDB_env -> String -> Int -> Int -> IO [Int]
sampleDistinctKeys env dbName n seed = do
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    !allKeys <- alloca $ \kvPtr -> alloca $ \dvPtr -> do
                  poke kvPtr (MDB_val 0 nullPtr)
                  poke dvPtr (MDB_val 0 nullPtr)
                  found <- mdb_cursor_get' MDB_FIRST cursor kvPtr dvPtr
                  collectKeys cursor kvPtr dvPtr found IS.empty
    mdb_txn_abort txn
    let allList = IS.toAscList allKeys
        size = length allList
        gen  = mkStdGen seed
        idxs = take n . IS.toAscList . IS.fromList . map (`mod` size) . take (n * 2) $ randoms gen
        chosen = [ allList !! i | i <- idxs ]
    return (take n chosen)
  where
    collectKeys _ _ _ False acc = return acc
    collectKeys cur kvPtr dvPtr True acc = do
      MDB_val ksz kp <- peek kvPtr
      k <- if ksz >= 4 then fromIntegral <$> peekElemOff (castPtr kp :: Ptr Int32) 0 else return 0
      -- Skip duplicate keys: jump to next *distinct* key via MDB_NEXT_NODUP.
      more <- mdb_cursor_get' MDB_NEXT_NODUP cur kvPtr dvPtr
      collectKeys cur kvPtr dvPtr more (IS.insert k acc)
