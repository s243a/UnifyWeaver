-- SPDX-License-Identifier: MIT
-- Common types and functions shared between the IntMap and LMDB backends.

{-# LANGUAGE BangPatterns #-}

module Common
  ( EdgeLookup
  , BenchResult(..)
  , dfsFromSeed
  , timeIt
  , summarize
  , formatSummary
  ) where

import qualified Data.IntSet as IS
import Data.List (sort)
import Data.Time.Clock (getCurrentTime, diffUTCTime)

-- | The universal node-lookup contract.  Each backend implements this.
--   Given a node id, returns the list of its parent node ids.
type EdgeLookup = Int -> [Int]

-- | Per-seed result: how many distinct nodes the DFS visited, and
--   how many nodes were reached at various depths.
data BenchResult = BenchResult
  { brVisited :: {-# UNPACK #-} !Int
  , brMaxDepth :: {-# UNPACK #-} !Int
  } deriving (Show)

-- | Depth-limited DFS from a seed, collecting a running count of
--   visited nodes and the maximum depth reached.  Deduplicates via
--   IntSet for cycle avoidance.  Pure; the only backend interaction
--   is through the EdgeLookup closure.
dfsFromSeed :: EdgeLookup -> Int -> Int -> BenchResult
dfsFromSeed edges maxDepth seed =
    go IS.empty 0 [(seed, 0)]
  where
    go !visited !maxD [] = BenchResult (IS.size visited) maxD
    go !visited !maxD ((node, depth):rest)
      | IS.member node visited = go visited maxD rest
      | depth >= maxDepth      = go (IS.insert node visited) (max maxD depth) rest
      | otherwise =
          let visited' = IS.insert node visited
              parents  = edges node
              children = [(p, depth+1) | p <- parents, not (IS.member p visited')]
          in  go visited' (max maxD depth) (children ++ rest)

-- | Time a pure computation, returning (result, elapsed-seconds).
timeIt :: IO a -> IO (a, Double)
timeIt action = do
    t0 <- getCurrentTime
    !r  <- action
    t1 <- getCurrentTime
    return (r, realToFrac (diffUTCTime t1 t0))

-- | Summary statistics over per-seed timings.
data Summary = Summary
  { sCount    :: !Int
  , sMean     :: !Double
  , sStddev   :: !Double
  , sMedian   :: !Double
  , sP95      :: !Double
  , sMin      :: !Double
  , sMax      :: !Double
  , sTotalSec :: !Double
  } deriving (Show)

summarize :: [Double] -> Summary
summarize xs =
  let !n   = length xs
      !total = sum xs
      !mean = total / fromIntegral n
      !variance = sum [(x - mean)^(2::Int) | x <- xs] / fromIntegral n
      !stddev = sqrt variance
      !sorted = sort xs
      at p = sorted !! min (n - 1) (floor (fromIntegral n * p :: Double))
  in  Summary n mean stddev (at (0.5 :: Double)) (at (0.95 :: Double))
              (head sorted) (last sorted) total

formatSummary :: String -> Summary -> String
formatSummary label s =
  unlines
    [ "=== " ++ label ++ " ==="
    , "  seeds:       " ++ show (sCount s)
    , "  total_sec:   " ++ showSec (sTotalSec s)
    , "  per_seed:"
    , "    mean:      " ++ showMs (sMean s)
    , "    stddev:    " ++ showMs (sStddev s)
    , "    median:    " ++ showMs (sMedian s)
    , "    p95:       " ++ showMs (sP95 s)
    , "    min:       " ++ showMs (sMin s)
    , "    max:       " ++ showMs (sMax s)
    ]
  where
    showMs x  = showF (x * 1000) ++ " ms"
    showSec x = showF x ++ " s"
    showF x   = let y = floor (x * 1000) :: Int
                in  show (fromIntegral y / 1000 :: Double)
