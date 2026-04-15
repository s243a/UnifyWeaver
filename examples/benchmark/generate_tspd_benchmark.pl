:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_tspd_benchmark.pl
%%
%% Benchmarks transitive_step_parent_distance5/5 (BFS + first-hop +
%% parent + distance) on category_parent. Returns 4-tuples per source —
%% exercises the 4-output FFIStreamRetry path under load. This is the
%% widest output kernel currently supported.
%%
%% Usage: swipl -q -s generate_tspd_benchmark.pl -- <output-dir>

% Canonical tspd shape:
tspd(X, Y, Y, X, 1) :- category_parent(X, Y).
tspd(X, Y, Mid, P, D) :-
    category_parent(X, Mid),
    tspd(Mid, Y, _Step, P, D1),
    D is D1 + 1.

main :-
    current_prolog_flag(argv, [OutputDir|_]),
    Predicates = [user:tspd/5],
    Options = [module_name('wam-tspd-bench'), emit_mode(interpreter)],
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
import qualified Data.IntSet as IS
import Data.List (nub, sort, foldl\')
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Control.Parallel.Strategies (parMap, rdeepseq)
import Control.DeepSeq (deepseq)
import WamRuntime (nativeKernel_transitive_step_parent_distance)

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

    let atomStrings =
            [child | (child, _) <- categoryParents] ++
            [parent | (_, parent) <- categoryParents]
        (!internMap, _) = foldl\'
            (\\(!m, !i) s -> if Map.member s m
                              then (m, i)
                              else (Map.insert s i m, i + 1))
            (Map.empty, 0 :: Int) atomStrings
        internAtom s = fromMaybe (-1) (Map.lookup s internMap)

    let !edgesIndex = IM.fromListWith (++)
            [(internAtom child, [internAtom parent]) | (child, parent) <- categoryParents]

    let !sources = nub $ sort $ map (internAtom . fst) categoryParents

    t2 <- getCurrentTime
    let !allResults = parMap rdeepseq (\\src ->
            let quads = nativeKernel_transitive_step_parent_distance edgesIndex src
            in (src, quads)
            ) sources
        !_ = allResults `deepseq` ()

    t3 <- getCurrentTime
    let queryMs = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs = round (diffUTCTime t3 t0 * 1000) :: Int
        totalQuads = sum [length qs | (_, qs) <- allResults]
        sumDistances = sum [d | (_, qs) <- allResults, (_, _, _, d) <- qs] :: Int
        -- Count of unique (source, step) pairs as a "first-hop" sanity check
        uniqueSteps = length $ nub $ sort $
            [(src, st) | (src, qs) <- allResults, (_, st, _, _) <- qs]

    hPutStrLn stderr $ "mode=tspd_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "source_count=" ++ show (length sources)
    hPutStrLn stderr $ "total_quads=" ++ show totalQuads
    hPutStrLn stderr $ "sum_distances=" ++ show sumDistances
    hPutStrLn stderr $ "unique_first_hops=" ++ show uniqueSteps
'),
    close(Stream).
