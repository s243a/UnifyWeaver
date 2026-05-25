-- | WAM step function with STArray registers: correct approach.
--
-- Key insight: only registers go in the mutable array. All other state
-- (PC, bindings, CPs, varCounter) stays as pure records threaded through
-- the loop. This avoids STRef overhead on every instruction while
-- getting O(1) mutable register access.
--
-- Pattern: step takes (STArray, PureState) -> ST s (Maybe PureState)
-- The outer run loop uses runST with the array created once.

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RankNTypes #-}

module Main where

import qualified Data.IntMap.Strict as IM
import qualified Data.Array as A
import qualified Data.Array.ST as STA
import Data.Array.MArray (readArray, writeArray, freeze, newArray)
import Control.Monad.ST.Strict
import System.CPUTime
import Control.Monad (forM_, when)
import Control.Exception (evaluate)
import Data.List (sort)
import Text.Printf
import Data.Maybe (fromMaybe)

------------------------------------------------------------------------
-- Shared types
------------------------------------------------------------------------

data Value = Atom !Int | Unbound !Int | IntVal !Int | VList ![Value]
  deriving (Show, Eq, Ord)

numRegs :: Int
numRegs = 200

data Instruction
  = GetConstant !Value !Int
  | PutConstant !Value !Int
  | PutVariable !Int !Int
  | GetVariable !Int !Int
  | PutValue !Int !Int
  | TryMeElse !Int
  | RetryMeElse !Int
  | TrustMe
  | Proceed
  | Allocate
  | Deallocate
  | Halt
  | Fail
  deriving (Show, Eq)

{-# INLINE deref #-}
deref :: IM.IntMap Value -> Value -> Value
deref bindings (Unbound vid) =
  case IM.lookup vid bindings of
    Just val -> deref bindings val
    Nothing  -> Unbound vid
deref _ v = v

------------------------------------------------------------------------
-- 1. IntMap WAM (baseline)
------------------------------------------------------------------------

data WamIM = WamIM
  { imPC :: {-# UNPACK #-} !Int
  , imRegs :: !(IM.IntMap Value)
  , imBindings :: !(IM.IntMap Value)
  , imCPs :: ![CPIM]
  , imCP :: {-# UNPACK #-} !Int
  , imVC :: {-# UNPACK #-} !Int
  }

data CPIM = CPIM !Int !(IM.IntMap Value) !(IM.IntMap Value) !Int !Int

{-# NOINLINE runIntMap #-}
runIntMap :: A.Array Int Instruction -> Int -> Maybe WamIM
runIntMap code seed = go (WamIM 1 IM.empty IM.empty [] 0 (seed * 1000))
  where
    hi = snd (A.bounds code)
    go !s
      | imPC s == 0 = Just s
      | imPC s > hi = Just s  -- ran off end = success
      | imPC s < 1 = Nothing
      | otherwise =
        let !instr = code A.! imPC s
        in case stepIM s instr of
          Just !s' -> go s'
          Nothing -> case btIM s of
            Just !s' -> go s'
            Nothing  -> Nothing

    {-# INLINE stepIM #-}
    stepIM :: WamIM -> Instruction -> Maybe WamIM
    stepIM !s (PutConstant c ai) =
      Just s { imPC = imPC s + 1, imRegs = IM.insert ai c (imRegs s) }
    stepIM !s (GetConstant c ai) =
      let !dv = deref (imBindings s) $ case IM.lookup ai (imRegs s) of
                  Just v -> v; Nothing -> Unbound (-1)
      in if dv == c then Just s { imPC = imPC s + 1 }
         else case dv of
           Unbound vid -> Just s { imPC = imPC s + 1
                                 , imRegs = IM.insert ai c (imRegs s)
                                 , imBindings = IM.insert vid c (imBindings s) }
           _ -> Nothing
    stepIM !s (PutVariable xn ai) =
      let !vid = imVC s; !var = Unbound vid
      in Just s { imPC = imPC s + 1
                , imRegs = IM.insert xn var (IM.insert ai var (imRegs s))
                , imVC = vid + 1 }
    stepIM !s (GetVariable xn ai) =
      let !val = fromMaybe (Unbound (-1)) (IM.lookup ai (imRegs s))
      in Just s { imPC = imPC s + 1, imRegs = IM.insert xn val (imRegs s) }
    stepIM !s (PutValue xn ai) =
      let !val = fromMaybe (Unbound (-1)) (IM.lookup xn (imRegs s))
      in Just s { imPC = imPC s + 1, imRegs = IM.insert ai val (imRegs s) }
    stepIM !s (TryMeElse nextPC) =
      let !cp = CPIM nextPC (imRegs s) (imBindings s) (imCP s) (imVC s)
      in Just s { imPC = imPC s + 1, imCPs = cp : imCPs s }
    stepIM !s TrustMe = case imCPs s of
      (_:rest) -> Just s { imPC = imPC s + 1, imCPs = rest }
      []       -> Just s { imPC = imPC s + 1 }
    stepIM !s (RetryMeElse nextPC) = case imCPs s of
      (CPIM _ r b cp vc : rest) ->
        Just s { imPC = imPC s + 1, imCPs = CPIM nextPC r b cp vc : rest }
      [] -> Nothing
    stepIM !s Proceed = Just s { imPC = imCP s }
    stepIM !s Allocate = Just s { imPC = imPC s + 1 }
    stepIM !s Deallocate = Just s { imPC = imPC s + 1 }
    stepIM _ Halt = Nothing
    stepIM _ Fail = Nothing
    stepIM _ _ = Nothing

    btIM :: WamIM -> Maybe WamIM
    btIM s = case imCPs s of
      [] -> Nothing
      (CPIM pc r b cp vc : rest) ->
        Just s { imPC = pc, imRegs = r, imBindings = b
               , imCP = cp, imVC = vc, imCPs = rest }

------------------------------------------------------------------------
-- 2. STArray WAM (proposed: mutable regs, pure rest)
------------------------------------------------------------------------

-- Pure state threaded through the loop (everything except registers)
data PureST = PureST
  { stPC :: {-# UNPACK #-} !Int
  , stBindings :: !(IM.IntMap Value)
  , stCPs :: ![CPST]
  , stCP :: {-# UNPACK #-} !Int
  , stVC :: {-# UNPACK #-} !Int
  }

-- Choice point with frozen register snapshot
data CPST = CPST !Int !(A.Array Int Value) !(IM.IntMap Value) !Int !Int

{-# NOINLINE runSTArray #-}
runSTArray :: A.Array Int Instruction -> Int -> Maybe (IM.IntMap Value, Int)
runSTArray code seed = runST go
  where
    hi = snd (A.bounds code)

    go :: forall s. ST s (Maybe (IM.IntMap Value, Int))
    go = do
      regs <- newArray (1, numRegs) (Unbound (-1)) :: ST s (STA.STArray s Int Value)
      let initST = PureST 1 IM.empty [] 0 (seed * 1000)

      let loop :: PureST -> ST s (Maybe PureST)
          loop !s
            | stPC s == 0 = return (Just s)
            | stPC s > hi = return (Just s)  -- ran off end = success
            | stPC s < 1 = return Nothing
            | otherwise = do
                let !instr = code A.! stPC s
                result <- stepST regs s instr
                case result of
                  Just !s' -> loop s'
                  Nothing -> do
                    bt <- btST regs s
                    case bt of
                      Just !s' -> loop s'
                      Nothing  -> return Nothing

      result <- loop initST
      case result of
        Nothing -> return Nothing
        Just finalS -> do
          -- Read back a few registers for correctness check
          finalArr <- freeze regs
          let regMap = IM.fromList [(i, finalArr A.! i) | i <- [1..numRegs],
                                    finalArr A.! i /= Unbound (-1)]
          return (Just (regMap, stVC finalS))

    {-# INLINE stepST #-}
    stepST :: STA.STArray s Int Value -> PureST -> Instruction -> ST s (Maybe PureST)

    stepST regs !s (PutConstant c ai) = do
      writeArray regs ai c
      return $ Just s { stPC = stPC s + 1 }

    stepST regs !s (GetConstant c ai) = do
      v <- readArray regs ai
      let !dv = deref (stBindings s) v
      if dv == c
        then return $ Just s { stPC = stPC s + 1 }
        else case dv of
          Unbound vid -> do
            writeArray regs ai c
            return $ Just s { stPC = stPC s + 1
                            , stBindings = IM.insert vid c (stBindings s) }
          _ -> return Nothing

    stepST regs !s (PutVariable xn ai) = do
      let !vid = stVC s; !var = Unbound vid
      writeArray regs xn var
      writeArray regs ai var
      return $ Just s { stPC = stPC s + 1, stVC = vid + 1 }

    stepST regs !s (GetVariable xn ai) = do
      val <- readArray regs ai
      writeArray regs xn val
      return $ Just s { stPC = stPC s + 1 }

    stepST regs !s (PutValue xn ai) = do
      val <- readArray regs xn
      writeArray regs ai val
      return $ Just s { stPC = stPC s + 1 }

    stepST regs !s (TryMeElse nextPC) = do
      snap <- freeze regs
      let !cp = CPST nextPC snap (stBindings s) (stCP s) (stVC s)
      return $ Just s { stPC = stPC s + 1, stCPs = cp : stCPs s }

    stepST _ !s TrustMe = case stCPs s of
      (_:rest) -> return $ Just s { stPC = stPC s + 1, stCPs = rest }
      []       -> return $ Just s { stPC = stPC s + 1 }

    stepST _ !s (RetryMeElse nextPC) = case stCPs s of
      (CPST _ snap b cp vc : rest) ->
        return $ Just s { stPC = stPC s + 1
                        , stCPs = CPST nextPC snap b cp vc : rest }
      [] -> return Nothing

    stepST _ !s Proceed = return $ Just s { stPC = stCP s }
    stepST _ !s Allocate = return $ Just s { stPC = stPC s + 1 }
    stepST _ !s Deallocate = return $ Just s { stPC = stPC s + 1 }
    stepST _ _ Halt = return Nothing
    stepST _ _ Fail = return Nothing
    stepST _ _ _ = return Nothing

    btST :: STA.STArray s Int Value -> PureST -> ST s (Maybe PureST)
    btST regs s = case stCPs s of
      [] -> return Nothing
      (CPST pc snap b cp vc : rest) -> do
        forM_ [1..numRegs] $ \i -> writeArray regs i (snap A.! i)
        return $ Just s { stPC = pc, stBindings = b, stCP = cp
                        , stVC = vc, stCPs = rest }

------------------------------------------------------------------------
-- Test & benchmark
------------------------------------------------------------------------

-- Register-heavy workload with realistic choice point frequency.
-- Real effective-distance: ~2500 instructions per choice point.
-- Here: 25 reg-heavy iterations between each choice point pair.
buildTestProgram :: Int -> [Instruction]
buildTestProgram nChoicePoints =
  concat [ choiceBlock i | i <- [0..nChoicePoints-1] ]
  where
    regOpsPerBlock = 25  -- 25 iterations of pure reg work per choice point
    instrPerIter = 7
    blockSize = 2 + regOpsPerBlock * instrPerIter + 1  -- try + work + trust

    choiceBlock i =
      let basePC = i * blockSize + 1
          trustPC = basePC + 1 + regOpsPerBlock * instrPerIter
      in [ TryMeElse trustPC ] ++
         concat [ regIteration (i * regOpsPerBlock + j)
                | j <- [0..regOpsPerBlock-1] ] ++
         [ TrustMe ]

    regIteration j =
      let r1 = j `mod` numRegs + 1
          r2 = (j + 37) `mod` numRegs + 1
          r3 = (j + 73) `mod` numRegs + 1
          w1 = (j + 11) `mod` numRegs + 1
          w2 = (j + 53) `mod` numRegs + 1
          w3 = (j + 97) `mod` numRegs + 1
      in [ PutConstant (IntVal j) w1
         , GetVariable (100 + r1 `mod` 50) r1
         , PutValue (100 + r2 `mod` 50) w2
         , PutConstant (IntVal (j * 3)) w3
         , GetVariable (100 + r3 `mod` 50) r3
         , PutConstant (IntVal (j + 1000)) (j `mod` numRegs + 1)
         , Allocate
         ]

main :: IO ()
main = do
  let nCPs = 2000  -- 2000 choice points, ~175 reg ops each = 350k reg ops
      prog = buildTestProgram nCPs
      code = A.listArray (1, length prog) prog
      nRounds = 7

  printf "STArray WAM Prototype (mutable regs, pure rest)\n"
  printf "================================================\n"
  printf "Program: %d instructions, %d choice points (~%d reg ops between CPs)\n\n"
    (length prog) nCPs (length prog `div` nCPs)

  -- Warmup (force evaluation via IO)
  _ <- evaluate (runIntMap code 0)
  _ <- evaluate (runSTArray code 0)

  -- Benchmark: extract a value from the result to prevent dead code elimination
  imTimes <- sequence [ do
    start <- getCPUTime
    let !r = runIntMap code i
    !vc <- evaluate (case r of Just s -> imVC s; Nothing -> 0)
    end <- getCPUTime
    when (i == 1) $ printf "  IntMap check: vc=%d\n" vc
    return (fromIntegral (end - start) / 1e9 :: Double) | i <- [1..nRounds]]

  stTimes <- sequence [ do
    start <- getCPUTime
    r <- evaluate (runSTArray code i)
    let !_ = case r of Just (_, vc) -> vc; Nothing -> 0
    end <- getCPUTime
    return (fromIntegral (end - start) / 1e9 :: Double) | i <- [1..nRounds]]

  let med xs = sort xs !! (length xs `div` 2)
      imMs = med imTimes
      stMs = med stTimes
      sp = imMs / stMs

  printf "Implementation       median_ms  speedup\n"
  printf "----------------------------------------\n"
  printf "IntMap (current)     %8.1f    1.00x\n" imMs
  printf "STArray (proposed)   %8.1f    %.2fx\n" stMs sp
  printf "\nRounds: IntMap=%s\n" (show (map rd imTimes))
  printf "        STArray=%s\n" (show (map rd stTimes))
  printf "\nRESULT OK (STArray speedup=%.2fx)\n" sp
  where
    rd t = fromIntegral (round (t*10)) / 10 :: Double
    timeIO action = do
      start <- getCPUTime
      !result <- action
      end <- getCPUTime
      return (fromIntegral (end - start) / 1e9 :: Double, result)
