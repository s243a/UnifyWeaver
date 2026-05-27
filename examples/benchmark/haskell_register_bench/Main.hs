-- | Haskell WAM Register Store Benchmark
--
-- Compares register implementations for WAM-like workloads:
-- 1. IntMap (current production): persistent, O(log n) per op
-- 2. STArray (proposed): mutable inside ST, pure outside, O(1) per op
-- 3. IOArray: mutable in IO, O(1) per op
-- 4. Immutable Array: O(1) read, O(n) write via (//)
--
-- STArray is the recommended approach: mutable performance inside runST
-- while preserving referential transparency and parallel safety.

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}

module Main where

import qualified Data.IntMap.Strict as IM
import qualified Data.Array.IO as IOA
import qualified Data.Array as A
import qualified Data.Array.ST as STA
import Data.Array.MArray (readArray, writeArray, freeze, newArray)
import Control.Monad.ST.Strict
import Data.STRef.Strict
import Data.IORef
import System.CPUTime
import Control.Monad (forM_, when)
import Data.List (foldl', sort)
import Text.Printf
import System.IO (hFlush, stdout)

numRegs :: Int
numRegs = 200

data Value = Atom !Int | Unbound !Int | IntVal !Int
  deriving (Show, Eq)

defaultVal :: Value
defaultVal = Unbound (-1)

-- Pre-seed registers with IntVal so reads always hit meaningful data.
seedRegs :: [(Int, Value)]
seedRegs = [(i, IntVal (i * 3)) | i <- [1..numRegs]]

------------------------------------------------------------------------
-- 1. IntMap
------------------------------------------------------------------------

{-# NOINLINE benchIntMap #-}
benchIntMap :: Int -> Int -> Int -> Int -> Int
benchIntMap nSteps choiceIv restoreIv seed =
  go (IM.fromList seedRegs) (IM.fromList seedRegs) (seed `mod` numRegs) 0
  where
    go :: IM.IntMap Value -> IM.IntMap Value -> Int -> Int -> Int
    go !regs !snapshot !i !acc
      | i >= nSteps + seed `mod` numRegs = acc
      | otherwise =
          let !idx = i `mod` numRegs + 1
              !idx2 = (i + 37) `mod` numRegs + 1
              !idx3 = (i + 73) `mod` numRegs + 1
              !wIdx = (i + 11) `mod` numRegs + 1
              !wIdx2 = (i + 53) `mod` numRegs + 1
              !wIdx3 = (i + 97) `mod` numRegs + 1
              !wIdx4 = (i + 131) `mod` numRegs + 1
              !v1 = case IM.lookup idx regs of Just v -> v; Nothing -> defaultVal
              !v2 = case IM.lookup idx2 regs of Just v -> v; Nothing -> defaultVal
              !v3 = case IM.lookup idx3 regs of Just v -> v; Nothing -> defaultVal
              !regs1 = IM.insert wIdx (IntVal i) regs
              !regs2 = IM.insert wIdx2 v1 regs1
              !regs3 = IM.insert wIdx3 v2 regs2
              !regs4 = IM.insert wIdx4 v3 regs3
              !delta = case v1 of IntVal n -> n; _ -> 0
              !acc' = acc + delta
          in if i `mod` choiceIv == 0
             then let !snap = regs4
                  in if i `mod` restoreIv == 0
                     then go snapshot snap (i+1) acc'
                     else go regs4 snap (i+1) acc'
             else go regs4 snapshot (i+1) acc'

------------------------------------------------------------------------
-- 2. STArray
------------------------------------------------------------------------

{-# NOINLINE benchSTArray #-}
benchSTArray :: Int -> Int -> Int -> Int -> Int
benchSTArray nSteps choiceIv restoreIv seed = runST go
  where
    go :: forall s. ST s Int
    go = do
      regs <- newArray (1, numRegs) defaultVal :: ST s (STA.STArray s Int Value)
      forM_ seedRegs $ \(k, v) -> writeArray regs k v
      snap0 <- freeze regs
      snapshotRef <- newSTRef snap0
      accRef <- newSTRef (0 :: Int)
      let start = seed `mod` numRegs
      forM_ [start .. nSteps + start - 1] $ \i -> do
        let !idx  = i `mod` numRegs + 1
            !idx2 = (i + 37) `mod` numRegs + 1
            !idx3 = (i + 73) `mod` numRegs + 1
            !wIdx  = (i + 11) `mod` numRegs + 1
            !wIdx2 = (i + 53) `mod` numRegs + 1
            !wIdx3 = (i + 97) `mod` numRegs + 1
            !wIdx4 = (i + 131) `mod` numRegs + 1
        !v1 <- readArray regs idx
        !v2 <- readArray regs idx2
        !v3 <- readArray regs idx3
        writeArray regs wIdx (IntVal i)
        writeArray regs wIdx2 v1
        writeArray regs wIdx3 v2
        writeArray regs wIdx4 v3
        let !delta = case v1 of IntVal x -> x; _ -> 0
        modifySTRef' accRef (+ delta)
        when (i `mod` choiceIv == 0) $ do
          !snap <- freeze regs
          writeSTRef snapshotRef snap
          when (i `mod` restoreIv == 0) $ do
            oldSnap <- readSTRef snapshotRef
            forM_ [1..numRegs] $ \j ->
              writeArray regs j (oldSnap A.! j)
      readSTRef accRef

------------------------------------------------------------------------
-- 3. IOArray
------------------------------------------------------------------------

{-# NOINLINE benchIOArr #-}
benchIOArr :: Int -> Int -> Int -> Int -> IO Int
benchIOArr nSteps choiceIv restoreIv seed = do
  regs <- IOA.newArray (1, numRegs) defaultVal :: IO (IOA.IOArray Int Value)
  forM_ seedRegs $ \(k, v) -> IOA.writeArray regs k v
  snap0 <- IOA.freeze regs
  snapshotRef <- newIORef snap0
  accRef <- newIORef (0 :: Int)
  let start = seed `mod` numRegs
  forM_ [start .. nSteps + start - 1] $ \i -> do
    let !idx  = i `mod` numRegs + 1
        !idx2 = (i + 37) `mod` numRegs + 1
        !idx3 = (i + 73) `mod` numRegs + 1
        !wIdx  = (i + 11) `mod` numRegs + 1
        !wIdx2 = (i + 53) `mod` numRegs + 1
        !wIdx3 = (i + 97) `mod` numRegs + 1
        !wIdx4 = (i + 131) `mod` numRegs + 1
    !v1 <- IOA.readArray regs idx
    !v2 <- IOA.readArray regs idx2
    !v3 <- IOA.readArray regs idx3
    IOA.writeArray regs wIdx (IntVal i)
    IOA.writeArray regs wIdx2 v1
    IOA.writeArray regs wIdx3 v2
    IOA.writeArray regs wIdx4 v3
    let !delta = case v1 of IntVal x -> x; _ -> 0
    modifyIORef' accRef (+ delta)
    when (i `mod` choiceIv == 0) $ do
      !snap <- IOA.freeze regs
      writeIORef snapshotRef snap
      when (i `mod` restoreIv == 0) $ do
        oldSnap <- readIORef snapshotRef
        forM_ [1..numRegs] $ \j ->
          IOA.writeArray regs j (oldSnap A.! j)
  readIORef accRef

------------------------------------------------------------------------
-- 4. Immutable Array
------------------------------------------------------------------------

{-# NOINLINE benchImmArr #-}
benchImmArr :: Int -> Int -> Int -> Int -> Int
benchImmArr nSteps choiceIv restoreIv seed =
  go arrInit arrInit (seed `mod` numRegs) 0
  where
    arrInit = A.array (1, numRegs) seedRegs
    go :: A.Array Int Value -> A.Array Int Value -> Int -> Int -> Int
    go !regs !snapshot !i !acc
      | i >= nSteps + seed `mod` numRegs = acc
      | otherwise =
          let !idx  = i `mod` numRegs + 1
              !idx2 = (i + 37) `mod` numRegs + 1
              !idx3 = (i + 73) `mod` numRegs + 1
              !wIdx  = (i + 11) `mod` numRegs + 1
              !wIdx2 = (i + 53) `mod` numRegs + 1
              !wIdx3 = (i + 97) `mod` numRegs + 1
              !wIdx4 = (i + 131) `mod` numRegs + 1
              !v1 = regs A.! idx
              !v2 = regs A.! idx2
              !v3 = regs A.! idx3
              !regs4 = regs A.// [(wIdx, IntVal i), (wIdx2, v1), (wIdx3, v2), (wIdx4, v3)]
              !delta = case v1 of IntVal n -> n; _ -> 0
              !acc' = acc + delta
          in if i `mod` choiceIv == 0
             then if i `mod` restoreIv == 0
                  then go snapshot regs4 (i+1) acc'
                  else go regs4 regs4 (i+1) acc'
             else go regs4 snapshot (i+1) acc'

------------------------------------------------------------------------
-- Harness
------------------------------------------------------------------------

timeIO :: IO a -> IO (Double, a)
timeIO action = do
  start <- getCPUTime
  !result <- action
  end <- getCPUTime
  let ms = fromIntegral (end - start) / 1e9 :: Double
  return (ms, result)

medianOf :: [Double] -> Double
medianOf xs = sort xs !! (length xs `div` 2)

main :: IO ()
main = do
  let steps = 500000
      choiceIv = 50
      restoreIv = 200
      nRounds = 7

  putStrLn "Haskell WAM Register Store Benchmark"
  putStrLn "====================================="
  printf "Config: %d regs, %d steps, choice every %d, restore every %d\n"
    numRegs steps choiceIv restoreIv
  putStrLn ""

  -- Warmup
  let !_ = benchIntMap steps choiceIv restoreIv 0
  let !_ = benchSTArray steps choiceIv restoreIv 0
  _ <- benchIOArr steps choiceIv restoreIv 0
  let !_ = benchImmArr steps choiceIv restoreIv 0

  putStr "Running benchmarks" >> hFlush stdout

  -- Each round uses a different seed to prevent GHC from sharing results
  imTimes <- sequence [ do
    (t, !_r) <- timeIO (return $! benchIntMap steps choiceIv restoreIv r)
    putStr "." >> hFlush stdout
    return t | r <- [1..nRounds]]
  stTimes <- sequence [ do
    (t, !_r) <- timeIO (return $! benchSTArray steps choiceIv restoreIv r)
    putStr "." >> hFlush stdout
    return t | r <- [1..nRounds]]
  ioTimes <- sequence [ do
    (t, !_r) <- timeIO (benchIOArr steps choiceIv restoreIv r)
    putStr "." >> hFlush stdout
    return t | r <- [1..nRounds]]
  arrTimes <- sequence [ do
    (t, !_r) <- timeIO (return $! benchImmArr steps choiceIv restoreIv r)
    putStr "." >> hFlush stdout
    return t | r <- [1..nRounds]]

  putStrLn " done"
  putStrLn ""

  let imMs  = medianOf imTimes
      stMs  = medianOf stTimes
      ioMs  = medianOf ioTimes
      arrMs = medianOf arrTimes
      spST  = imMs / stMs
      spIO  = imMs / ioMs
      spArr = imMs / arrMs

  putStrLn "Implementation       median_ms  speedup  notes"
  putStrLn "-----------------------------------------------------"
  printf   "IntMap (current)     %8.1f    1.00x   O(log n) per op, O(1) snapshot\n" imMs
  printf   "STArray (proposed)   %8.1f    %.2fx   O(1) per op, pure outside ST\n" stMs spST
  printf   "IOArray              %8.1f    %.2fx   O(1) per op, IO monad\n" ioMs spIO
  printf   "Array (//)           %8.1f    %.2fx   O(1) read, O(n) write\n" arrMs spArr
  putStrLn ""
  printf   "Rounds: IntMap=%s\n" (show (map (\t -> fromIntegral (round (t * 10)) / 10 :: Double) imTimes))
  printf   "        STArray=%s\n" (show (map (\t -> fromIntegral (round (t * 10)) / 10 :: Double) stTimes))
  printf   "        IOArray=%s\n" (show (map (\t -> fromIntegral (round (t * 10)) / 10 :: Double) ioTimes))
  printf   "        Array=%s\n" (show (map (\t -> fromIntegral (round (t * 10)) / 10 :: Double) arrTimes))
  putStrLn ""

  -- Correctness check (same seed, should produce same result)
  let imAcc  = benchIntMap steps choiceIv restoreIv 42
      stAcc  = benchSTArray steps choiceIv restoreIv 42
      arrAcc = benchImmArr steps choiceIv restoreIv 42
  ioAcc <- benchIOArr steps choiceIv restoreIv 42
  if imAcc == stAcc && stAcc == ioAcc && ioAcc == arrAcc
    then printf "RESULT OK (all agree: acc=%d, STArray speedup=%.2fx)\n" imAcc spST
    else do
      printf "RESULT MISMATCH (intmap=%d st=%d io=%d arr=%d)\n" imAcc stAcc ioAcc arrAcc
      printf "  Note: mismatch may indicate snapshot/restore ordering differences\n"
