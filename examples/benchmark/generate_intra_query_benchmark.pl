:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_intra_query_benchmark.pl
%%
%% Phase 4.0 generator — wraps intra_query_seed.pl into a Haskell WAM
%% project with FFI *disabled*, producing a benchmark driver that runs
%% power_sum_bound/4 through the WAM interpreter for a small number of
%% seeds.
%%
%% The disabled-FFI + few-seeds combination is deliberate: we want the
%% interpreter to be in the hot path (so the choice points are visible
%% to future Par* instructions) and we want seed-level parMap to have
%% too few sparks to saturate the cores. Both conditions hold here.
%%
%% Usage:
%%   swipl -q -s generate_intra_query_benchmark.pl -- <output-dir>
%%
%% The resulting binary accepts:
%%   <facts-dir> [num-seeds]
%% and prints query timings to stderr. Run with `+RTS -N1` and
%% `+RTS -N4` to observe the (absence of) speedup that motivates
%% Phase 4.1+. The run always uses `parMap rdeepseq` — with -N1 that
%% collapses to sequential; with -N4 it sparks one core per seed.
%% We deliberately avoid running both modes in one process because
%% GHC would CSE the two calls (same inputs, same pure function).
%%
%% Related: docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md §4.0

workload_path(Path) :-
    source_file(workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'intra_query_seed.pl', Path).

main :-
    current_prolog_flag(argv, [OutputDir|_]),
    workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),

    % Freeze the argument mode the WAM compiler will see for
    % category_ancestor/4.
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4,
        user:power_sum_bound/4
    ],
    Options = [
        module_name('wam-intra-query-bench'),
        emit_mode(interpreter),
        no_kernels(true),
        % Use Data.Map.Strict (not HashMap) so our custom Main.hs can
        % build `mergedLabels` with plain `Map.fromList` / `Map.union`.
        use_hashmap(false)
    ],
    write_wam_haskell_project(Predicates, Options, OutputDir),

    % Overwrite Main.hs with our dedicated benchmark driver.
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
import Data.List (sortBy, foldl\')
import Data.Ord (comparing, Down(..))
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
import Control.Parallel.Strategies (parMap, rdeepseq)
import Control.DeepSeq (deepseq)
import WamTypes
import WamRuntime
import Predicates
import qualified Lowered

-- | Load a TSV file, skip header, return pairs.
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

-- | Build SwitchOnConstant dispatch table from grouped facts.
buildFactIndex :: [(String, String)] -> Map.Map String [(String, String)]
buildFactIndex pairs =
    foldl (\\m (a, b) -> Map.insertWith (++) a [(a, b)] m) Map.empty pairs

-- | Compile an indexed 2-column fact predicate into WAM instructions.
buildFact2Code :: String -> [(String, String)] -> Int -> ([Instruction], [(String, Int)])
buildFact2Code predName pairs startPC =
    let groups = Map.toAscList (buildFactIndex pairs)
        (dispatchList, groupCode, groupLabels, _) =
            foldl buildGroup ([], [], [], startPC + 1) groups
        switchInstr = SwitchOnConstant (Map.fromList dispatchList)
    in (switchInstr : groupCode, (predName ++ "/2", startPC) : groupLabels)
  where
    buildGroup (disp, code, labels, pc) (key, facts) =
      let groupLabel = predName ++ "_g_" ++ key
          (fcode, flabels, nextPC) = buildFactGroup predName key facts pc
      in (disp ++ [(Atom key, groupLabel)],
          code ++ fcode,
          labels ++ [(groupLabel, pc)] ++ flabels,
          nextPC)

    buildFactGroup _ _ [] pc = ([], [], pc)
    buildFactGroup pn key facts pc =
      let n = length facts
          buildFact i (a, b) curPC =
            let choiceInstr = if n == 1 then [] else
                  if i == 0 then [TryMeElse (pn ++ "_g_" ++ key ++ "_" ++ show (i+1))]
                  else if i == n - 1 then [TrustMe]
                  else [RetryMeElse (pn ++ "_g_" ++ key ++ "_" ++ show (i+1))]
                factInstrs = [GetConstant (Atom a) 1, GetConstant (Atom b) 2, Proceed]
                label = (pn ++ "_g_" ++ key ++ "_" ++ show i, curPC)
            in (choiceInstr ++ factInstrs,
                [label],
                curPC + length choiceInstr + length factInstrs)
          (allCode, allLabels, _) = foldl (\\(c, l, p) (i, f) ->
              let (fc, fl, np) = buildFact i f p in (c ++ fc, l ++ fl, np))
            ([], [], pc) (zip [0..] facts)
      in (allCode, allLabels, pc + length allCode)

-- | Run category_ancestor/4 for a seed, collecting all Hops values via
-- backtracking. Then fold W = (hops+1)^negN in Haskell. This matches the
-- canonical collectSolutions pattern from the default Main template.
--
-- We deliberately push the aggregate to Haskell rather than using a
-- WAM-compiled power_sum_bound/4 so the benchmark exercises only the
-- choice-point machinery (category_parent retries + cycle pruning),
-- which is exactly where intra-query parallelism (Phase 4.1+) would
-- fork.
runSeed :: WamContext -> Double -> String -> (String, Int, Double)
runSeed !ctx !negN cat =
    let !hopsVarId = 1000000
        !rootVarId = 1000001
        pcStart = fromMaybe 1 $ Map.lookup "category_ancestor/4" (wcLabels ctx)
        s0 = emptyState
            { wsPC = pcStart
            , wsRegs = IM.fromList
                [ (1, Atom cat)
                , (2, Unbound rootVarId)
                , (3, Unbound hopsVarId)
                , (4, VList [Atom cat])
                ]
            , wsCP = 0
            }
        !solutions = collectSolutions ctx s0 hopsVarId
        !n         = length solutions
        !weightSum = sum [ (fromIntegral (h + 1)) ** negN | h <- solutions ]
    in (cat, n, weightSum)

collectSolutions :: WamContext -> WamState -> Int -> [Int]
collectSolutions !ctx s0 hopsVarId =
    case run ctx s0 of
      Nothing -> []
      Just s1 ->
        let hopsVal = case IM.lookup hopsVarId (wsBindings s1) of
              Just v -> extractInt (derefVar (wsBindings s1) v)
              Nothing -> Nothing
            rest = case backtrack s1 of
              Just s2 -> collectSolutions ctx s2 hopsVarId
              Nothing -> []
        in case hopsVal of
          Just h  -> h : rest
          Nothing -> rest

extractInt :: Value -> Maybe Int
extractInt (Integer n) = Just (fromIntegral n)
extractInt (Float f)   = Just (round f)
extractInt (Atom s)    = case reads s of [(h, "")] -> Just h; _ -> Nothing
extractInt _           = Nothing

main :: IO ()
main = do
    args <- getArgs
    let (factsDir, numSeedsArg) = case args of
          (d:n:_) -> (d, Just (read n :: Int))
          [d]     -> (d, Nothing)
          []      -> (".", Nothing)
        numSeeds = fromMaybe 5 numSeedsArg

    t0 <- getCurrentTime
    categoryParents <- loadTsvPairs (factsDir ++ "/category_parent.tsv")
    t1 <- getCurrentTime
    let loadMs = round (diffUTCTime t1 t0 * 1000) :: Int

    -- Build merged WAM code: compiled predicates + runtime facts.
    -- The Predicates module exposes allCode/allLabels from the compiled
    -- workload predicates; we splice in category_parent/2 as a runtime
    -- indexed fact predicate (same as the default Main.hs template).
    let baseLen = length allCode
        (cpCode, cpLabels) =
            buildFact2Code "category_parent" categoryParents (baseLen + 1)
        mergedCodeRaw = allCode ++ cpCode
        mergedLabels = Map.union allLabels (Map.fromList cpLabels)
        foreignPreds = [] :: [String]
        mergedCode = resolveCallInstrs mergedLabels foreignPreds mergedCodeRaw

    -- Pick top-K seeds by out-degree. This biases toward nodes with
    -- the richest per-seed branching — precisely where intra-query
    -- parallelism would pay off.
    let outDegrees :: Map.Map String Int
        outDegrees = Map.fromListWith (+)
            [(a, 1) | (a, _) <- categoryParents]
        topSeeds = take numSeeds $ map fst $
            sortBy (comparing (Down . snd)) (Map.toList outDegrees)

    let !ctx = (mkContext mergedCode mergedLabels)
            { wcForeignConfig = Map.singleton "max_depth" 6
            , wcLoweredPredicates = Lowered.loweredPredicates
            }
        !negN = -5.0 :: Double

    t2 <- getCurrentTime

    -- Seed-level parMap. With +RTS -N1 this degrades to sequential;
    -- with +RTS -N4 it sparks one core per seed. The benchmark is
    -- structured so seed-level sparks saturate at `numSeeds` — if
    -- cores > numSeeds, extra cores sit idle. That is the intended
    -- failure mode to motivate Phase 4.1+.
    let !results = parMap rdeepseq (runSeed ctx negN) topSeeds
        !_ = results `deepseq` ()
    t3 <- getCurrentTime
    let queryMs    = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs    = round (diffUTCTime t3 t0 * 1000) :: Int
        totalSols  = sum [ n | (_, n, _) <- results ]
        totalWsum  = sum [ w | (_, _, w) <- results ]

    hPutStrLn stderr "mode=intra_query_seed_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "seed_count=" ++ show (length topSeeds)
    hPutStrLn stderr $ "edges_loaded=" ++ show (length categoryParents)
    hPutStrLn stderr $ "total_solutions=" ++ show totalSols
    hPutStrLn stderr $ "total_weight_sum=" ++ show totalWsum
    mapM_ (\\(s, n, w) ->
        hPutStrLn stderr ("seed_result=" ++ s ++ "\\tsols=" ++ show n ++ "\\tw=" ++ show w)
      ) results
'),
    close(Stream).
