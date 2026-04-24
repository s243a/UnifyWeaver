-- SPDX-License-Identifier: MIT
-- LMDB backend.  Opens the database with a long-lived read txn and
-- reuses a single cursor across lookups to iterate duplicate values
-- for each key (dupsort).  The EdgeLookup closure captures the
-- cursor and does one MDB_SET + many MDB_NEXT_DUP per query.

{-# LANGUAGE BangPatterns #-}

module LmdbBackend
  ( openLmdbBackend
  ) where

import Control.Monad (when)
import Data.Int (Int32)
import Foreign.Marshal.Alloc (alloca, allocaBytes)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import System.IO.Unsafe (unsafePerformIO)
import Database.LMDB.Raw

import Common (EdgeLookup)

-- | Open the LMDB database read-only and return an EdgeLookup backed
-- by a long-lived read txn + a reusable cursor.
openLmdbBackend :: FilePath -> IO EdgeLookup
openLmdbBackend path = do
    env <- mdb_env_create
    mdb_env_set_mapsize env (2 * 1024 * 1024 * 1024)  -- 2 GB
    mdb_env_set_maxdbs env 4
    mdb_env_set_maxreaders env 126
    mdb_env_open env path [MDB_RDONLY]
    txn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' txn (Just "main") [MDB_DUPSORT]
    cursor <- mdb_cursor_open' txn dbi
    return (lookupDups cursor)

-- | Dupsort lookup: position the cursor at `key` via MDB_SET, read
-- the first matching value, then walk MDB_NEXT_DUP until the key
-- changes (mdb_cursor_get' returns False).
--
-- NOT thread-safe: the cursor is shared state.  The DFS benchmark is
-- single-threaded so this is fine; a multithreaded consumer would
-- need one cursor per thread.
lookupDups :: MDB_cursor' -> EdgeLookup
lookupDups cursor key = unsafePerformIO $
    allocaBytes 4 $ \kp -> do
      poke (castPtr kp :: Ptr Int32) (fromIntegral key :: Int32)
      alloca $ \kvPtr -> alloca $ \dvPtr -> do
        poke kvPtr (MDB_val 4 kp)
        found <- mdb_cursor_get' MDB_SET cursor kvPtr dvPtr
        if not found
          then return []
          else collectDups kvPtr dvPtr
  where
    collectDups kvPtr dvPtr = do
      v <- readInt32Val dvPtr
      more <- mdb_cursor_get' MDB_NEXT_DUP cursor kvPtr dvPtr
      if more
        then do
          rest <- collectDups kvPtr dvPtr
          return (v : rest)
        else return [v]

    readInt32Val :: Ptr MDB_val -> IO Int
    readInt32Val ptr = do
      MDB_val sz dataPtr <- peek ptr
      if sz >= 4
        then fromIntegral <$> peekElemOff (castPtr dataPtr :: Ptr Int32) 0
        else return 0
