{-# LANGUAGE BangPatterns #-}
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.List (group, sort, foldl')
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr, hFlush, stdout)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Control.DeepSeq (NFData(..), deepseq)
import WamTypes
import WamRuntime
import Predicates
import qualified Lowered

loadTsvPairs :: FilePath -> IO [(String, String)]
loadTsvPairs path = do
    content <- readFile path
    let ls = drop 1 (lines content)
    return [(a, b) | l <- ls, let ws = splitOn '\t' l, length ws >= 2, let [a, b] = take 2 ws]

loadSingleColumn :: FilePath -> IO [String]
loadSingleColumn path = do
    content <- readFile path
    return [l | l <- drop 1 (lines content), not (null l)]

splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn d (c:cs)
  | c == d    = "" : splitOn d cs
  | otherwise = let (w:ws) = splitOn d cs in (c:w) : ws

-- Collect solutions using IntMap run (current)
collectSolutionsIM :: WamContext -> WamState -> Int -> [Double]
collectSolutionsIM !ctx s0 hopsVarId =
    case run ctx s0 of
      Nothing -> []
      Just s1 ->
        let hopsVal = case IM.lookup hopsVarId (wsBindings s1) of
              Just v -> extractDouble (wcInternTable ctx) (derefVar (wsBindings s1) v)
              Nothing -> Nothing
            hops = fromMaybe 0 hopsVal
            rest = case backtrack s1 of
              Just s2 -> collectSolutionsIM ctx s2 hopsVarId
              Nothing -> []
        in case hopsVal of
          Just _ -> hops : rest
          Nothing -> rest

-- Collect solutions using STArray runMutableRegs (proposed)
collectSolutionsST :: WamContext -> WamState -> Int -> [Double]
collectSolutionsST !ctx s0 hopsVarId =
    case runMutableRegs ctx s0 of
      Nothing -> []
      Just s1 ->
        let hopsVal = case IM.lookup hopsVarId (wsBindings s1) of
              Just v -> extractDouble (wcInternTable ctx) (derefVar (wsBindings s1) v)
              Nothing -> Nothing
            hops = fromMaybe 0 hopsVal
            -- backtrack via runMutableRegs too
            rest = case runMutableRegs ctx (s1 { wsPC = 0 }) of
              -- Can't easily backtrack via ST without re-entering run.
              -- Use pure backtrack for continuation.
              _ -> case backtrack s1 of
                Just s2 -> collectSolutionsST ctx s2 hopsVarId
                Nothing -> []
        in case hopsVal of
          Just _ -> hops : rest
          Nothing -> rest

main :: IO ()
main = do
    args <- getArgs
    let factsDir = if null args then "." else head args
        numReps = if length args > 1 then read (args !! 1) else 5

    categoryParents <- loadTsvPairs (factsDir ++ "/category_parent.tsv")
    articleCategories <- loadTsvPairs (factsDir ++ "/article_category.tsv")
    roots <- loadSingleColumn (factsDir ++ "/root_categories.tsv")

    let atomStrings =
            [child | (child, _) <- categoryParents] ++
            [parent | (_, parent) <- categoryParents] ++
            [cat | (_, cat) <- articleCategories] ++
            roots
        !fullInternTable = foldl'
            (\tbl s -> snd (internAtom tbl s))
            compileTimeAtomTable atomStrings
        iAtom s = internAtomPure fullInternTable s

    let mergedCodeRaw = allCode
        mergedLabels = allLabels
        foreignPreds = ["category_ancestor/4"]
        mergedCode = resolveCallInstrs mergedLabels foreignPreds mergedCodeRaw

    let seedCats = map head $ group $ sort $ map snd articleCategories
        root = head roots
        n = 5.0 :: Double
        negN = -n

    let !parentsIndexInterned = IM.fromListWith (++)
            [(iAtom child, [iAtom parent]) | (child, parent) <- categoryParents]

    let !ctx = (mkContext mergedCode mergedLabels)
            { wcForeignConfig = Map.singleton "max_depth" 10
            , wcLoweredPredicates = Lowered.loweredPredicates
            , wcInternTable   = fullInternTable
            , wcFfiFacts      = Map.singleton "category_parent" parentsIndexInterned
            }

    hPutStrLn stderr "Haskell WAM Register Store E2E Benchmark"
    hPutStrLn stderr "========================================="
    hPutStrLn stderr $ "Seeds: " ++ show (length seedCats) ++ "  Edges: " ++ show (length categoryParents)
    hPutStrLn stderr $ "Reps: " ++ show numReps
    hPutStrLn stderr ""

    -- Run both paths and compare correctness first
    let queryOneSeed runFn cat =
          let hopsVarId = 1000000
              s0 = emptyState
                { wsPC = fromMaybe 1 $ Map.lookup "category_ancestor/4" mergedLabels
                , wsRegs = IM.fromList [(1, Atom (iAtom cat)), (2, Atom (iAtom root)),
                                        (3, Unbound hopsVarId), (4, VList [Atom (iAtom cat)])]
                , wsCP = 0
                }
              solutions = runFn ctx s0 hopsVarId
          in sum [((hops + 1) ** negN) | hops <- solutions]

    -- Correctness check
    let imResults = map (\cat -> (cat, queryOneSeed collectSolutionsIM cat)) seedCats
        stResults = map (\cat -> (cat, queryOneSeed collectSolutionsST cat)) seedCats
        mismatches = [(cat, im, st) | ((cat, im), (_, st)) <- zip imResults stResults, abs (im - st) > 1e-10]

    hPutStrLn stderr $ "Correctness: " ++ show (length mismatches) ++ " mismatches out of " ++ show (length seedCats) ++ " seeds"
    if null mismatches
      then hPutStrLn stderr "  All seeds agree!"
      else mapM_ (\(cat, im, st) -> hPutStrLn stderr $ "  MISMATCH: " ++ cat ++ " im=" ++ show im ++ " st=" ++ show st) (take 5 mismatches)
    hPutStrLn stderr ""

    -- Benchmark: IntMap
    imTimings <- sequence [ do
      t0 <- getCurrentTime
      let !results = map (\cat -> queryOneSeed collectSolutionsIM cat) seedCats
          !_ = results `deepseq` ()
      t1 <- getCurrentTime
      let ms = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
      return ms
      | _ <- [1..numReps]]

    -- Benchmark: STArray
    stTimings <- sequence [ do
      t0 <- getCurrentTime
      let !results = map (\cat -> queryOneSeed collectSolutionsST cat) seedCats
          !_ = results `deepseq` ()
      t1 <- getCurrentTime
      let ms = realToFrac (diffUTCTime t1 t0) * 1000 :: Double
      return ms
      | _ <- [1..numReps]]

    let median xs = sort xs !! (length xs `div` 2)
        imMs = median imTimings
        stMs = median stTimings
        speedup = if stMs > 0 then imMs / stMs else 0

    hPutStrLn stderr "Results"
    hPutStrLn stderr "-------"
    hPutStrLn stderr $ "IntMap (current):   " ++ show (round imMs :: Int) ++ " ms (median)"
    hPutStrLn stderr $ "STArray (proposed): " ++ show (round stMs :: Int) ++ " ms (median)"
    hPutStrLn stderr $ "Speedup:            " ++ show speedup ++ "x"
    hPutStrLn stderr ""
    hPutStrLn stderr $ "Rounds: IntMap=" ++ show (map round imTimings :: [Int])
    hPutStrLn stderr $ "        STArray=" ++ show (map round stTimings :: [Int])

    if null mismatches
      then hPutStrLn stderr $ "\nRESULT OK (speedup=" ++ show speedup ++ "x)"
      else hPutStrLn stderr $ "\nRESULT MISMATCH"

extractDouble :: InternTable -> Value -> Maybe Double
extractDouble _ (Integer h) = Just (fromIntegral h)
extractDouble _ (Float h) = Just h
extractDouble tbl (Atom aid) = case reads (lookupAtom tbl aid) of [(h, "")] -> Just h; _ -> Nothing
extractDouble _ _ = Nothing
