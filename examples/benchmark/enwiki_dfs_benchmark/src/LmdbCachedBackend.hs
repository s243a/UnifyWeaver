-- SPDX-License-Identifier: MIT
-- LmdbCachedBackend: demand-driven memoising wrapper around LMDB dupsort.
--
-- Motivation
-- ----------
-- Naive upfront prefetch doesn't work for this workload: the DFS in
-- dfsFromSeed drives which nodes are visited — you cannot know the
-- reachable set without first running the traversal you are trying to
-- prefetch for.  See NOTE [Prefetch bootstrapping problem] below.
--
-- Instead we use demand-driven memoisation: each node's dupsort values
-- are fetched from LMDB exactly once (on first access, by whichever
-- thread gets there first), then stored in a shared pure IntMap.  All
-- subsequent accesses — by the same seed or any other parallel seed
-- that shares the node — read the pure IntMap with no FFI and no IORef
-- contention.
--
-- This gives the IntMap parallelism profile on the hot path while
-- keeping LMDB as the authoritative data source.
--
-- NOTE [Prefetch bootstrapping problem]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- The DFS in Common.dfsFromSeed calls `edges node` only on nodes it
-- actually visits.  Which nodes get visited depends on the edge data
-- returned by `edges`.  So the reachable working set is not known until
-- the traversal is complete — making upfront prefetch a chicken-and-egg
-- problem.  Demand-driven caching resolves this naturally.
--
-- NOTE [Thread-local cursors]
-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~
-- LMDB cursors are NOT thread-safe; a cursor must only be used by the
-- thread that created it.  We keep a Map ThreadId MDB_cursor' under a
-- MVar so that each thread opens its own cursor once and reuses it for
-- all its cache-miss FFI calls.
--
-- NOTE [Main.hs wiring]
-- ~~~~~~~~~~~~~~~~~~~~~
-- To activate this backend, add to Main.hs:
--
--   import qualified LmdbCachedBackend as LC
--   ...
--   "lmdb_cached" -> LC.openLmdbCachedBackend dbPath
--
-- No changes to the parallel harness or Common.hs are needed.

{-# LANGUAGE BangPatterns #-}

module LmdbCachedBackend
  ( openLmdbCachedBackend
  ) where

import Control.Concurrent (myThreadId, ThreadId)
import Control.Concurrent.MVar
import Data.Int (Int32)
import Data.IORef
import qualified Data.IntMap.Strict as IM
import qualified Data.Map.Strict as Map
import Foreign.Marshal.Alloc (alloca, allocaBytes)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peek, poke, peekElemOff)
import System.IO.Unsafe (unsafePerformIO)
import Database.LMDB.Raw

import Common (EdgeLookup)

-- ---------------------------------------------------------------------------
-- Public API
-- ---------------------------------------------------------------------------

-- | Open the LMDB database read-only.  Returns an EdgeLookup that
--   memoises each node's dupsort neighbours on first access and serves
--   all subsequent accesses from a pure IntMap cache.
--
--   Thread-safe: multiple Haskell threads (sparks) may call the
--   returned EdgeLookup concurrently.  Cache hits are pure IntMap
--   lookups (no synchronisation); cache misses use a per-thread cursor
--   and atomicModifyIORef' to update the shared cache.
openLmdbCachedBackend :: FilePath -> IO EdgeLookup
openLmdbCachedBackend path = do
    -- Shared LMDB environment + dbi (read-only; safe to share across txns).
    env <- mdb_env_create
    mdb_env_set_mapsize env (2 * 1024 * 1024 * 1024)
    mdb_env_set_maxdbs env 4
    mdb_env_set_maxreaders env 126
    mdb_env_open env path [MDB_RDONLY]

    -- Bootstrap the dbi handle.  IMPORTANT: the txn must be COMMITTED
    -- (not aborted) so the dbi handle persists across subsequent
    -- transactions.  Per LMDB docs, "if the transaction is aborted the
    -- handle will be closed automatically", which would invalidate
    -- every per-thread cursor opened later.
    bootstrapTxn <- mdb_txn_begin env Nothing True
    dbi <- mdb_dbi_open' bootstrapTxn (Just "main") [MDB_DUPSORT]
    mdb_txn_commit bootstrapTxn

    -- Shared memoisation cache: IntMap [Int], starts empty.
    -- IORef: reads are cheap; writes (cache misses) are rare once warm.
    cacheRef <- newIORef (IM.empty :: IM.IntMap [Int])

    -- Per-thread cursor registry.  Wrapped in MVar for thread-safe
    -- insertion of new entries.  Each thread only reads/writes its own
    -- ThreadId key so contention is limited to the rare first-miss
    -- per thread.
    cursorMapVar <- newMVar (Map.empty :: Map.Map ThreadId MDB_cursor')

    return $ cachedLookup env dbi cacheRef cursorMapVar

-- ---------------------------------------------------------------------------
-- Internal implementation
-- ---------------------------------------------------------------------------

-- | The EdgeLookup closure.
cachedLookup
    :: MDB_env
    -> MDB_dbi'
    -> IORef (IM.IntMap [Int])
    -> MVar (Map.Map ThreadId MDB_cursor')
    -> EdgeLookup
cachedLookup env dbi cacheRef cursorMapVar key = unsafePerformIO $ do
    -- Fast path: pure cache hit (no synchronisation).
    cache <- readIORef cacheRef
    case IM.lookup key cache of
      Just vs -> return vs
      Nothing -> do
        -- Slow path: cache miss.  Fetch from LMDB, populate cache.
        vs <- fetchFromLmdb env dbi cursorMapVar key
        -- Atomically insert into cache.  If another thread raced us,
        -- we just overwrite with the same value (dupsort data is
        -- immutable).  The double-write is harmless and cheaper than
        -- a check-then-set loop.
        atomicModifyIORef' cacheRef $ \m -> (IM.insert key vs m, ())
        return vs

-- | Fetch dupsort values for `key` from LMDB using this thread's cursor.
--   Opens a new read txn + cursor on first call per thread; reuses on
--   subsequent calls.
fetchFromLmdb
    :: MDB_env
    -> MDB_dbi'
    -> MVar (Map.Map ThreadId MDB_cursor')
    -> Int
    -> IO [Int]
fetchFromLmdb env dbi cursorMapVar key = do
    tid <- myThreadId
    cursorMap <- readMVar cursorMapVar
    cursor <- case Map.lookup tid cursorMap of
      Just c  -> return c
      Nothing -> do
        -- First cache-miss for this thread: open a dedicated read txn
        -- and cursor.  Both stay alive for the life of the thread (no
        -- explicit cleanup; acceptable for batch jobs).
        txn <- mdb_txn_begin env Nothing True
        cur <- mdb_cursor_open' txn dbi
        modifyMVar_ cursorMapVar $ \m -> return (Map.insert tid cur m)
        return cur
    lookupDupsSafe cursor key

-- | Dupsort lookup using an already-positioned cursor.  Same logic as
--   LmdbBackend.lookupDups but safe to call with any thread-local cursor.
lookupDupsSafe :: MDB_cursor' -> Int -> IO [Int]
lookupDupsSafe cursor key =
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
        then (v :) <$> collectDups kvPtr dvPtr
        else return [v]

    readInt32Val :: Ptr MDB_val -> IO Int
    readInt32Val ptr = do
      MDB_val sz dataPtr <- peek ptr
      if sz >= 4
        then fromIntegral <$> peekElemOff (castPtr dataPtr :: Ptr Int32) 0
        else return 0
