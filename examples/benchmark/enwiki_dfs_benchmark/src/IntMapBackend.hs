-- SPDX-License-Identifier: MIT
-- IntMap backend.  Reads the entire LMDB at startup, materializes an
-- adjacency list as an IntMap [Int] in the GHC heap, returns a pure
-- EdgeLookup closure.  After startup the LMDB is closed; queries hit
-- only memory.

{-# LANGUAGE BangPatterns #-}

module IntMapBackend
  ( openIntMapFromLmdb
  ) where

import Control.Monad (when)
import qualified Data.IntMap.Strict as IM
import Data.IORef
import Data.Int (Int32)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, peekElemOff)
import Database.LMDB.Raw

import Common (EdgeLookup)

-- | Open the LMDB, iterate every (key, value) pair with a cursor,
-- accumulate into an IntMap [Int] (dupsort keys → lists of values).
-- Close the LMDB at the end.  Returns a pure EdgeLookup.
openIntMapFromLmdb :: FilePath -> IO EdgeLookup
openIntMapFromLmdb path = do
    env <- mdb_env_create
    mdb_env_set_mapsize env (2 * 1024 * 1024 * 1024)  -- 2 GB
    mdb_env_set_maxdbs env 4
    mdb_env_set_maxreaders env 126
    mdb_env_open env path [MDB_RDONLY]
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just "main") [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi

    mapRef  <- newIORef IM.empty
    countRef <- newIORef (0 :: Int)

    let loop !step = do
          hasMore <- alloca $ \kvPtr -> alloca $ \dvPtr -> do
            r <- mdb_cursor_get' step cursor kvPtr dvPtr
            if r
              then do
                MDB_val kSz kPtr <- peek kvPtr
                MDB_val vSz vPtr <- peek dvPtr
                !k <- if kSz >= 4
                        then fromIntegral <$> peekElemOff (castPtr kPtr :: Ptr Int32) 0
                        else return (0 :: Int)
                !v <- if vSz >= 4
                        then fromIntegral <$> peekElemOff (castPtr vPtr :: Ptr Int32) 0
                        else return (0 :: Int)
                modifyIORef' mapRef (IM.alter (addValue v) k)
                modifyIORef' countRef (+1)
                return True
              else return False
          when hasMore (loop MDB_NEXT)

    loop MDB_FIRST
    mdb_cursor_close' cursor
    mdb_txn_abort txn
    mdb_env_close env

    n <- readIORef countRef
    putStrLn $ "[intmap] loaded " ++ show n ++ " edges into IntMap"
    !m <- readIORef mapRef
    putStrLn $ "[intmap] distinct keys: " ++ show (IM.size m)

    return $ \k -> IM.findWithDefault [] k m

  where
    addValue v Nothing   = Just [v]
    addValue v (Just vs) = Just (v : vs)
