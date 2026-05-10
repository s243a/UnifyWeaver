-- SPDX-License-Identifier: MIT
-- M3: warming-vs-JIT crossover.
--
-- Run M random queries against:
--   (a) cold path: per-query LMDB cursor seeks for the keys it touches
--   (b) warmed:    pre-load top-K keys into an IntMap; fall through to
--                  cursor only on miss.
--
-- Find the break-even M (queries-per-warmup) where (b) beats (a) by
-- a margin worth a cost-model decision. The "K" budget we choose is
-- arbitrary for Phase M — we just want the crossover shape.

{-# LANGUAGE BangPatterns #-}

module Crossover
  ( runCrossover
  , RunResult(..)
  ) where

import Control.Monad (forM, forM_)
import Data.Int (Int32)
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.List (foldl')
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Foreign.Marshal.Alloc (alloca, allocaBytes)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import Database.LMDB.Raw
import System.Random (mkStdGen, randoms)

data RunResult = RunResult
  { rrTotalMs   :: !Double
  , rrEdgeReads :: !Int
  , rrCacheHits :: !Int
  } deriving Show

-- | Run M BFS-shaped queries from random roots, max-hops=3.
runCrossover :: MDB_env -> String -> [Int] -> Int -> Maybe (IM.IntMap [Int]) -> IO RunResult
runCrossover env dbName roots maxHops mWarm = do
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    t0 <- getCurrentTime
    (!totalReads, !totalHits) <- foldQueries cursor mWarm roots
    t1 <- getCurrentTime
    mdb_txn_abort txn
    let !ms = realToFrac (diffUTCTime t1 t0) * 1000.0
    return $ RunResult ms totalReads totalHits
  where
    foldQueries _ _ [] = return (0, 0)
    foldQueries cur warm (r:rs) = do
      (rd, hi) <- bfs cur warm r maxHops
      (rd', hi') <- foldQueries cur warm rs
      return (rd + rd', hi + hi')

    bfs cur warm root hops = go IS.empty (IS.singleton root) 0 0 0
      where
        go _ frontier !depth !reads !hits
          | IS.null frontier || depth >= hops = return (reads, hits)
          | otherwise = do
              let nodes = IS.toList frontier
              (!newEdges, !rd, !hi) <- expand cur warm nodes
              let nextFrontier = IS.fromList newEdges `IS.difference` frontier
              go (frontier `IS.union` nextFrontier) nextFrontier (depth+1) (reads + rd) (hits + hi)

    expand _ _ []     = return ([], 0, 0)
    expand cur warm (n:ns) = do
      (xs, !rd, !hi) <- lookupOne cur warm n
      (ys, !rd', !hi') <- expand cur warm ns
      return (xs ++ ys, rd + rd', hi + hi')

    lookupOne cur warm n = case warm of
      Just m | Just vs <- IM.lookup n m -> return (vs, 0, 1)
      _                                 -> do
        vs <- countDupsCollect cur n
        return (vs, length vs, 0)

countDupsCollect :: MDB_cursor' -> Int -> IO [Int]
countDupsCollect cursor key =
    allocaBytes 4 $ \kp -> do
      poke (castPtr kp :: Ptr Int32) (fromIntegral key)
      alloca $ \kvPtr -> alloca $ \dvPtr -> do
        poke kvPtr (MDB_val 4 kp)
        found <- mdb_cursor_get' MDB_SET cursor kvPtr dvPtr
        if not found then return [] else walk kvPtr dvPtr []
  where
    walk kvPtr dvPtr acc = do
      MDB_val vsz vp <- peek dvPtr
      v <- if vsz >= 4 then fromIntegral <$> peekElemOff (castPtr vp :: Ptr Int32) 0 else return 0
      more <- mdb_cursor_get' MDB_NEXT_DUP cursor kvPtr dvPtr
      if more then walk kvPtr dvPtr (v : acc) else return (v : acc)
