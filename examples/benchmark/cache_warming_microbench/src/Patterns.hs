-- SPDX-License-Identifier: MIT
-- M1: LMDB read-pattern microbench.
--
-- Same workload (resolve dupsort values for N keys) under five access
-- patterns. The comparison the cost model needs is "scan time vs
-- cumulative seek time" — i.e. when does E (full-table sequential scan)
-- beat C (sorted-key seeks) for a given N?

{-# LANGUAGE BangPatterns #-}

module Patterns
  ( runPatternBench
  , Pattern(..)
  ) where

import Control.DeepSeq (deepseq)
import Control.Monad (forM_, when)
import Data.Int (Int32)
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.List (sort)
import Data.Time.Clock (diffUTCTime, getCurrentTime)
import Foreign.Marshal.Alloc (alloca, allocaBytes)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import Database.LMDB.Raw

data Pattern = PerLookupCursor | SharedInOrder | SharedSorted | SharedShuffled | FullScan
  deriving (Eq, Show)

-- | Run one read pattern against a dupsort sub-db, return (wall_ms, edges_collected).
runPatternBench :: MDB_env -> String -> [Int] -> Pattern -> IO (Double, Int)
runPatternBench env dbName keys pat = do
    t0 <- getCurrentTime
    !edgeCount <- case pat of
      PerLookupCursor -> perLookup env dbName keys
      SharedInOrder   -> sharedCursor env dbName keys
      SharedSorted    -> sharedCursor env dbName (sort keys)
      SharedShuffled  -> sharedCursor env dbName (shuffleDeterministic keys)
      FullScan        -> fullScan env dbName (IS.fromList keys)
    t1 <- getCurrentTime
    let !ms = realToFrac (diffUTCTime t1 t0) * 1000.0
    return (ms, edgeCount)

-- Pattern A: open and close a fresh cursor per key (worst case for setup overhead).
perLookup :: MDB_env -> String -> [Int] -> IO Int
perLookup env dbName keys = go 0 keys
  where
    go !acc []     = return acc
    go !acc (k:ks) = do
      txn <- mdb_txn_begin env Nothing True
      dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
      cursor <- mdb_cursor_open' txn dbi
      n <- countDups cursor k
      mdb_txn_abort txn
      go (acc + n) ks

-- Patterns B/C/D: open one cursor up front, MDB_SET + MDB_NEXT_DUP per key.
sharedCursor :: MDB_env -> String -> [Int] -> IO Int
sharedCursor env dbName keys = do
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    !n <- foldKeys cursor 0 keys
    mdb_txn_abort txn
    return n
  where
    foldKeys _ !acc []     = return acc
    foldKeys cur !acc (k:ks) = do
      m <- countDups cur k
      foldKeys cur (acc + m) ks

-- Count dups for one key (MDB_SET + MDB_NEXT_DUP walk).
countDups :: MDB_cursor' -> Int -> IO Int
countDups cursor key =
    allocaBytes 4 $ \kp -> do
      poke (castPtr kp :: Ptr Int32) (fromIntegral key)
      alloca $ \kvPtr -> alloca $ \dvPtr -> do
        poke kvPtr (MDB_val 4 kp)
        found <- mdb_cursor_get' MDB_SET cursor kvPtr dvPtr
        if not found then return 0 else walk kvPtr dvPtr 1
  where
    walk kvPtr dvPtr !n = do
      more <- mdb_cursor_get' MDB_NEXT_DUP cursor kvPtr dvPtr
      if more then walk kvPtr dvPtr (n+1) else return n

-- Pattern E: full-table sequential scan via MDB_FIRST + MDB_NEXT, only count
-- entries whose key is in the wanted set.
fullScan :: MDB_env -> String -> IS.IntSet -> IO Int
fullScan env dbName wantedKeys = do
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just dbName) [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    !n <- alloca $ \kvPtr -> alloca $ \dvPtr -> do
            poke kvPtr (MDB_val 0 nullPtr)
            poke dvPtr (MDB_val 0 nullPtr)
            found0 <- mdb_cursor_get' MDB_FIRST cursor kvPtr dvPtr
            walkAll cursor kvPtr dvPtr found0 0
    mdb_txn_abort txn
    return n
  where
    walkAll _ _ _ False !acc = return acc
    walkAll cur kvPtr dvPtr True !acc = do
      MDB_val ksz kp <- peek kvPtr
      kVal <- if ksz >= 4
              then fromIntegral <$> peekElemOff (castPtr kp :: Ptr Int32) 0
              else return (0 :: Int)
      let !acc' = if IS.member kVal wantedKeys then acc + 1 else acc
      more <- mdb_cursor_get' MDB_NEXT cur kvPtr dvPtr
      walkAll cur kvPtr dvPtr more acc'

-- Deterministic shuffle for reproducibility (Fisher-Yates with a fixed seed-ish swap).
shuffleDeterministic :: [Int] -> [Int]
shuffleDeterministic xs =
    let arr = IM.fromList (zip [0..] xs)
        n   = IM.size arr
        swaps = [(i, ((i * 2654435761) `mod` n)) | i <- [0 .. n-1]]
        go !m [] = m
        go !m ((i,j):rest) =
          let a = m IM.! i
              b = m IM.! j
              !m' = IM.insert i b (IM.insert j a m)
          in go m' rest
    in IM.elems (go arr swaps)
