-- SPDX-License-Identifier: MIT
-- enwiki-dfs-benchmark: IntMap vs LMDB DFS comparison at enwiki scale.
--
-- Usage:
--   enwiki-dfs <lmdb-path> <backend> <n-seeds>
--   backend = intmap | lmdb
--
-- Samples N random seeds (from a list of valid keys in the LMDB,
-- using a fixed RNG seed for reproducibility), runs a depth-limited
-- DFS from each seed, reports per-seed timing statistics.

{-# LANGUAGE BangPatterns #-}

module Main where

import Control.Monad (forM, forM_, when)
import Data.IORef
import Data.Int (Int32)
import qualified Data.IntSet as IS
import Data.List (foldl')
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, peekElemOff)
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.Random (mkStdGen, random, StdGen)
import Database.LMDB.Raw

import Common
import qualified IntMapBackend as IM
import qualified LmdbBackend as L

maxDfsDepth :: Int
maxDfsDepth = 12

rngSeedFixed :: Int
rngSeedFixed = 0xCA7C0DE  -- fixed for reproducibility

main :: IO ()
main = do
    args <- getArgs
    case args of
      [dbPath, backend, nSeedsStr] -> do
        let !nSeeds = read nSeedsStr :: Int
        putStrLn $ "[main] db=" ++ dbPath ++ " backend=" ++ backend
                ++ " seeds=" ++ show nSeeds ++ " max_depth=" ++ show maxDfsDepth

        -- Sample N random keys from the LMDB key space.
        keys <- sampleKeys dbPath nSeeds
        putStrLn $ "[main] sampled " ++ show (length keys) ++ " seeds"

        -- Open the selected backend and run DFS for each seed.
        edges <- case backend of
          "intmap" -> IM.openIntMapFromLmdb dbPath
          "lmdb"   -> L.openLmdbBackend dbPath
          _ -> do
            putStrLn $ "unknown backend: " ++ backend
            exitFailure

        putStrLn "[main] warming up..."
        _ <- forM (take (min 100 nSeeds) keys) $ \k -> do
               let !br = dfsFromSeed edges maxDfsDepth k
               return br

        putStrLn "[main] measuring..."
        -- Time the whole sweep.  Individual per-seed timings are below
        -- clock resolution for fast queries and produce garbage stats.
        -- The reliable number is total-time ÷ seed-count.
        (brs, elapsed) <- timeIt $ do
          rs <- forM keys $ \k -> do
                  let !br = dfsFromSeed edges maxDfsDepth k
                  return br
          return rs

        let !totalVisited = foldl' (\a br -> a + brVisited br) 0 brs
            !maxDepthSeen = foldl' (\a br -> max a (brMaxDepth br)) 0 brs
            !n = length brs
            !meanMs = (elapsed / fromIntegral n) * 1000
            !seedsPerSec = fromIntegral n / elapsed :: Double

        putStrLn ""
        putStrLn $ "=== " ++ backend ++ " DFS ==="
        putStrLn $ "  seeds:                 " ++ show n
        putStrLn $ "  total_elapsed_sec:     " ++ show elapsed
        putStrLn $ "  per_seed_mean_ms:      " ++ show meanMs
        putStrLn $ "  seeds_per_sec:         " ++ show seedsPerSec
        putStrLn $ "  total_nodes_visited:   " ++ show totalVisited
        putStrLn $ "  avg_nodes_per_seed:    " ++ show (fromIntegral totalVisited / fromIntegral n :: Double)
        putStrLn $ "  max_depth_observed:    " ++ show maxDepthSeen

      _ -> do
        putStrLn "usage: enwiki-dfs <lmdb-path> <intmap|lmdb> <n-seeds>"
        exitFailure

-- | Sample N distinct keys from an LMDB database via reservoir sampling.
-- One pass over the LMDB (using MDB_NEXT_NODUP to skip dupsort values),
-- O(N) memory, fixed RNG for reproducibility.
sampleKeys :: FilePath -> Int -> IO [Int]
sampleKeys dbPath nSeeds = do
    env <- mdb_env_create
    mdb_env_set_mapsize env (2 * 1024 * 1024 * 1024)
    mdb_env_set_maxdbs env 4
    mdb_env_set_maxreaders env 126
    mdb_env_open env dbPath [MDB_RDONLY]
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just "main") [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi

    -- Reservoir sampling (Algorithm R): keep the first N items; for
    -- the i-th item (i > N), replace a random slot with probability N/i.
    -- Deterministic with a fixed RNG stream.
    reservoir <- newIORef ([] :: [Int])     -- stored in reverse order
    seen <- newIORef (0 :: Int)
    -- Precompute an infinite stream of random Ints from the fixed seed.
    -- We'll take one per visited key (cheap for 2.6M keys).
    rngRef <- newIORef (mkStdGen rngSeedFixed)

    let walk !step = do
          more <- alloca $ \kvPtr -> alloca $ \dvPtr -> do
            r <- mdb_cursor_get' step cursor kvPtr dvPtr
            if r
              then do
                MDB_val kSz kPtr <- peek kvPtr
                !k <- if kSz >= 4
                        then fromIntegral <$> peekElemOff (castPtr kPtr :: Ptr Int32) 0
                        else return (0 :: Int)
                !i <- readIORef seen
                writeIORef seen (i + 1)
                if i < nSeeds
                  then modifyIORef' reservoir (k:)
                  else do
                    -- reservoir is full; replace a slot with prob N/(i+1)
                    g <- readIORef rngRef
                    let (r', g') = random g :: (Int, StdGen)
                        j = (abs r') `mod` (i + 1)
                    writeIORef rngRef g'
                    when (j < nSeeds) $ do
                      cur <- readIORef reservoir
                      writeIORef reservoir (replaceAt j k cur)
                return True
              else return False
          when more (walk MDB_NEXT_NODUP)
    walk MDB_FIRST
    mdb_cursor_close' cursor
    mdb_txn_abort txn
    mdb_env_close env

    total <- readIORef seen
    putStrLn $ "[sample] distinct keys in LMDB: " ++ show total
    readIORef reservoir

replaceAt :: Int -> a -> [a] -> [a]
replaceAt _ _ []     = []
replaceAt 0 y (_:xs) = y : xs
replaceAt n y (x:xs) = x : replaceAt (n-1) y xs
