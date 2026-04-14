:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_tdist_benchmark.pl
%%
%% Generates a Haskell project that benchmarks transitive_distance3/3
%% (the multi-output BFS kernel) on the category_parent graph.
%%
%% The benchmark:
%%   - For each unique category in the graph, computes the set of
%%     (target, distance) pairs reachable from it.
%%   - Times the aggregate loop across all source categories.
%%   - Supports seed-level parallelism via +RTS -N.
%%
%% Usage:
%%   swipl -q -s generate_tdist_benchmark.pl -- <output-dir>

% The canonical transitive_distance shape the detector matches:
tdist(X, Y, 1) :- category_parent(X, Y).
tdist(X, Y, D) :- category_parent(X, Z), tdist(Z, Y, D1), D is D1 + 1.

main :-
    current_prolog_flag(argv, [OutputDir|_]),
    Predicates = [user:tdist/3],
    Options = [module_name('wam-tdist-bench'), emit_mode(interpreter)],
    write_wam_haskell_project(Predicates, Options, OutputDir),
    % Overwrite Main.hs with our benchmark driver
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
import WamRuntime (nativeKernel_transitive_distance)

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

    -- Build atom intern table
    let atomStrings =
            [child | (child, _) <- categoryParents] ++
            [parent | (_, parent) <- categoryParents]
        (!internMap, _) = foldl\'
            (\\(!m, !i) s -> if Map.member s m
                              then (m, i)
                              else (Map.insert s i m, i + 1))
            (Map.empty, 0 :: Int) atomStrings
        internAtom s = fromMaybe (-1) (Map.lookup s internMap)

    -- Build edges index: child -> [parents] in interned form
    let !edgesIndex = IM.fromListWith (++)
            [(internAtom child, [internAtom parent]) | (child, parent) <- categoryParents]

    -- All unique source nodes (children of some edge = potential sources)
    let !sources = nub $ sort $ map (internAtom . fst) categoryParents

    t2 <- getCurrentTime

    -- For each source, compute all (target, distance) pairs.
    -- This exercises the BFS kernel end-to-end.
    let !allResults = parMap rdeepseq (\\src ->
            let pairs = nativeKernel_transitive_distance edgesIndex src
            in (src, pairs)
            ) sources
        !_ = allResults `deepseq` ()

    t3 <- getCurrentTime
    let queryMs = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs = round (diffUTCTime t3 t0 * 1000) :: Int
        totalPairs = sum [length ps | (_, ps) <- allResults]

    hPutStrLn stderr $ "mode=tdist_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "source_count=" ++ show (length sources)
    hPutStrLn stderr $ "total_pairs=" ++ show totalPairs
'),
    close(Stream).
