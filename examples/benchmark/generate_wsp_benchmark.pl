:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_wsp_benchmark.pl
%%
%% Benchmarks weighted_shortest_path3/3 (Dijkstra) on the category_parent
%% graph with SYNTHETIC weights. Each edge gets a deterministic weight
%% computed from the child's interned ID — gives weights in [1.0, 2.0)
%% so shortest paths differ from unweighted BFS distances.
%%
%% Unlike tdist (integer hops), wsp returns Double distances, exercising
%% the Float output wrap in the multi-output FFI infrastructure.
%%
%% Usage:
%%   swipl -q -s generate_wsp_benchmark.pl -- <output-dir>

% The canonical weighted shortest path shape the detector matches:
wsp(X, Y, W) :- weighted_edge(X, Y, W).
wsp(X, Y, TotalW) :-
    weighted_edge(X, Mid, W1),
    wsp(Mid, Y, RestW),
    TotalW is RestW + W1.

main :-
    current_prolog_flag(argv, [OutputDir|_]),
    Predicates = [user:wsp/3],
    Options = [module_name('wam-wsp-bench'), emit_mode(interpreter)],
    write_wam_haskell_project(Predicates, Options, OutputDir),
    directory_file_path(OutputDir, 'src/Main.hs', MainPath),
    write_benchmark_main(MainPath),
    halt(0).
main :- halt(1).

write_benchmark_main(Path) :-
    open(Path, write, Stream),
    write(Stream,
'{-# LANGUAGE BangPatterns #-}
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import qualified Data.Set as Set
import Data.List (nub, sort, foldl\')
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Control.Parallel.Strategies (parMap, rdeepseq)
import Control.DeepSeq (deepseq)
import WamRuntime (nativeKernel_weighted_shortest_path)

loadTsvPairs :: FilePath -> IO [(String, String)]
loadTsvPairs path = do
    content <- readFile path
    let ls = drop 1 (lines content)
    return [(a, b) | l <- ls, let ws = splitOn \'\\t\' l, length ws >= 2, let [a, b] = take 2 ws]

splitOn :: Char -> String -> [String]
splitOn _ [] = [""]
splitOn d (c:cs)
  | c == d    = "" : splitOn d cs
  | otherwise = let (w:ws) = splitOn d cs in (c:w) : ws

main :: IO ()
main = do
    args <- getArgs
    let factsDir = if null args then "." else head args

    t0 <- getCurrentTime
    categoryParents <- loadTsvPairs (factsDir ++ "/category_parent.tsv")
    t1 <- getCurrentTime
    let loadMs = round (diffUTCTime t1 t0 * 1000) :: Int

    -- Build intern table
    let atomStrings =
            [child | (child, _) <- categoryParents] ++
            [parent | (_, parent) <- categoryParents]
        (!internMap, _) = foldl\'
            (\\(!m, !i) s -> if Map.member s m
                              then (m, i)
                              else (Map.insert s i m, i + 1))
            (Map.empty, 0 :: Int) atomStrings
        internAtom s = fromMaybe (-1) (Map.lookup s internMap)

    -- Build weighted edges. Synthetic weight = 1.0 + (childId mod 10) * 0.1
    -- gives weights in [1.0, 2.0), deterministic, exercises Dijkstra\'s
    -- priority queue with non-uniform weights.
    let !weightedEdges = IM.fromListWith (++)
            [(childId, [(parentId, weight)])
             | (child, parent) <- categoryParents
             , let childId = internAtom child
             , let parentId = internAtom parent
             , let weight = 1.0 + fromIntegral (childId `mod` 10) * 0.1 :: Double
             ]

    let !sources = nub $ sort $ map (internAtom . fst) categoryParents

    t2 <- getCurrentTime
    let !allResults = parMap rdeepseq (\\src ->
            let pairs = nativeKernel_weighted_shortest_path weightedEdges src
            in (src, pairs)
            ) sources
        !_ = allResults `deepseq` ()

    t3 <- getCurrentTime
    let queryMs = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs = round (diffUTCTime t3 t0 * 1000) :: Int
        totalPairs = sum [length ps | (_, ps) <- allResults]
        -- Sum of all distances as a sanity check (deterministic for given data)
        totalWeight = sum [w | (_, ps) <- allResults, (_, w) <- ps] :: Double

    hPutStrLn stderr $ "mode=wsp_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "source_count=" ++ show (length sources)
    hPutStrLn stderr $ "total_pairs=" ++ show totalPairs
    hPutStrLn stderr $ "total_weight=" ++ show totalWeight
'),
    close(Stream).
