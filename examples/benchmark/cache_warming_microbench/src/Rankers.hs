-- SPDX-License-Identifier: MIT
-- M2: ranker overhead microbench.
--
-- Three rankers, all consuming the same edge list. The cost classes
-- are very different even though all three feed into "pick top-K
-- edges to warm":
--
--   - PPR/Flux iteration:   O(E) per iter (sequential pass + score update).
--   - Streaming hop:        O(E log F) one pass, multi-pass optional.
--                           Frontier in IntMap; relax each (u,v) seen.
--   - Stub semantic sim:    O(K * D) — dot product per candidate against
--                           a fixed query embedding (D=128 floats here).
--
-- We're not fighting over correctness here — these are cost-class
-- microbenches. The ranker's *quality* is a separate question.

{-# LANGUAGE BangPatterns #-}

module Rankers
  ( pprIteration
  , streamingHop
  , semanticSim
  , collectAllEdges
  ) where

import Control.DeepSeq (deepseq)
import Control.Monad (forM_, when)
import Data.Int (Int32)
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.List (foldl')
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import Database.LMDB.Raw

-- | Pull all (u, v) edges from a dupsort sub-db via MDB_FIRST + MDB_NEXT.
-- This is the Phase M baseline scan — used by all three rankers.
collectAllEdges :: MDB_env -> String -> IO [(Int, Int)]
collectAllEdges env dbName = do
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    edges <- alloca $ \kvPtr -> alloca $ \dvPtr -> do
              poke kvPtr (MDB_val 0 nullPtr)
              poke dvPtr (MDB_val 0 nullPtr)
              found <- mdb_cursor_get' MDB_FIRST cursor kvPtr dvPtr
              walk cursor kvPtr dvPtr found []
    mdb_txn_abort txn
    return (reverse edges)
  where
    walk _ _ _ False acc = return acc
    walk cur kvPtr dvPtr True acc = do
      MDB_val ksz kp <- peek kvPtr
      MDB_val vsz vp <- peek dvPtr
      k <- if ksz >= 4 then fromIntegral <$> peekElemOff (castPtr kp :: Ptr Int32) 0 else return 0
      v <- if vsz >= 4 then fromIntegral <$> peekElemOff (castPtr vp :: Ptr Int32) 0 else return 0
      more <- mdb_cursor_get' MDB_NEXT cur kvPtr dvPtr
      walk cur kvPtr dvPtr more ((k,v) : acc)

-- | One PPR-style iteration: for each edge (u, v), accumulate score[u] += score[v] / outdeg[v].
-- Returns updated score map. Pure / no IO.
pprIteration :: IM.IntMap Double -> IM.IntMap Int -> [(Int, Int)] -> IM.IntMap Double
pprIteration scores outdeg edges =
    foldl' relax IM.empty edges
  where
    relax !acc (u, v) =
      let sv = IM.findWithDefault 0.0 v scores
          dv = IM.findWithDefault 1 v outdeg
          contrib = sv / fromIntegral dv
      in IM.insertWith (+) u contrib acc

-- | Streaming hop-distance: one pass over edges, propagate (id, hop) via IntMap frontier.
-- Per the user's algorithm: if u is in frontier at hop h and v isn't (or has higher hop),
-- record v at h+1.  Single pass; multi-pass convergence not implemented here (Phase M
-- measures the cost of ONE pass, which is what matters for the resolver decision).
streamingHop :: IM.IntMap Int -> [(Int, Int)] -> IM.IntMap Int
streamingHop initialFrontier edges =
    foldl' relax initialFrontier edges
  where
    relax !frontier (u, v) =
      case IM.lookup u frontier of
        Nothing -> frontier
        Just hu ->
          let hv = hu + 1
          in case IM.lookup v frontier of
               Just hv' | hv' <= hv -> frontier
               _                    -> IM.insert v hv frontier

-- | Stub semantic similarity: dot product of each candidate's embedding
-- against a fixed query embedding. We don't have real embeddings — we
-- simulate by treating each Int as a deterministic 128-d vector.
-- Cost is what we measure; quality is irrelevant for Phase M.
semanticSim :: [Int] -> [Double]
semanticSim candidates =
    [ dotProduct (embedding c) queryEmbedding | c <- candidates ]
  where
    !queryEmbedding = [ sin (fromIntegral i * 0.7) | i <- [0 .. 127 :: Int] ]
    embedding :: Int -> [Double]
    embedding c = [ sin (fromIntegral (c * (i + 1))) | i <- [0 .. 127 :: Int] ]
    dotProduct as bs = sum (zipWith (*) as bs)
