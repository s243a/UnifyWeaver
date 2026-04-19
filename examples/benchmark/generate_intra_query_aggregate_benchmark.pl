:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_intra_query_aggregate_benchmark.pl
%%
%% Phase 4.2 follow-up: exercises the intra-query fork path END TO END.
%% Unlike generate_intra_query_benchmark.pl (which sums in Haskell via
%% collectSolutions), this benchmark calls power_sum_bound/4 through the
%% WAM interpreter so the aggregate_all(sum, …) compiles to
%% BeginAggregate/EndAggregate. The inner category_ancestor/4 has Par*
%% instructions emitted. When ParTryMeElse fires inside the sum
%% aggregate, forkParBranches should trigger parMap rdeepseq across
%% the alternative clauses.
%%
%% Usage:
%%   swipl -q -s generate_intra_query_aggregate_benchmark.pl -- <output-dir>
%%
%% The resulting binary accepts:
%%   <facts-dir> [num-seeds]
%% and prints query timings + per-seed weight sums to stderr.
%%
%% Related:
%%   docs/design/WAM_HASKELL_INTRA_QUERY_IMPLEMENTATION_PLAN.md §4.2

workload_path(Path) :-
    source_file(workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'intra_query_seed.pl', Path).

main :-
    current_prolog_flag(argv, [OutputDir|_]),
    workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),

    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4,
        user:power_sum_bound/4
    ],
    Options = [
        module_name('wam-intra-agg-bench'),
        emit_mode(interpreter),
        no_kernels(true),
        use_hashmap(false),
        intra_query_parallel(true)
    ],
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

buildFactIndex :: [(String, String)] -> Map.Map String [(String, String)]
buildFactIndex pairs =
    foldl (\\m (a, b) -> Map.insertWith (++) a [(a, b)] m) Map.empty pairs

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

-- | Run power_sum_bound/4 for a single seed. Registers:
-- A1 = seed atom, A2 = fresh unbound (Root), A3 = NegN,
-- A4 = fresh unbound (WeightSum output).
-- The WAM aggregate_all(sum, …) inside power_sum_bound compiles to
-- BeginAggregate/EndAggregate. With category_ancestor/4 having Par*
-- instructions, forkParBranches should fire inside the sum aggregate.
runSeedAggregate :: WamContext -> Double -> String -> (String, Double)
runSeedAggregate !ctx !negN cat =
    let !wsVarId   = 1000000
        !rootVarId = 1000001
        pcStart = fromMaybe 1 $ Map.lookup "power_sum_bound/4" (wcLabels ctx)
        s0 = emptyState
            { wsPC = pcStart
            , wsRegs = IM.fromList
                [ (1, Atom cat)
                , (2, Unbound rootVarId)
                , (3, Float negN)
                , (4, Unbound wsVarId)
                ]
            , wsCP = 0
            }
        !result = case run ctx s0 of
          Just s1 -> case IM.lookup wsVarId (wsBindings s1) of
            Just v -> case extractDouble (derefVar (wsBindings s1) v) of
              Just ws -> ws
              Nothing -> 0.0
            Nothing -> 0.0
          Nothing -> 0.0
    in (cat, result)

extractDouble :: Value -> Maybe Double
extractDouble (Integer n) = Just (fromIntegral n)
extractDouble (Float f)   = Just f
extractDouble (Atom s)    = case reads s of [(h, "")] -> Just h; _ -> Nothing
extractDouble _           = Nothing

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

    let baseLen = length allCode
        (cpCode, cpLabels) =
            buildFact2Code "category_parent" categoryParents (baseLen + 1)
        mergedCodeRaw = allCode ++ cpCode
        mergedLabels = Map.union allLabels (Map.fromList cpLabels)
        foreignPreds = [] :: [String]
        mergedCode = resolveCallInstrs mergedLabels foreignPreds mergedCodeRaw

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

    let !results = parMap rdeepseq (runSeedAggregate ctx negN) topSeeds
        !_ = results `deepseq` ()
    t3 <- getCurrentTime
    let queryMs    = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs    = round (diffUTCTime t3 t0 * 1000) :: Int
        totalWsum  = sum [ w | (_, w) <- results ]

    hPutStrLn stderr "mode=intra_query_aggregate_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "seed_count=" ++ show (length topSeeds)
    hPutStrLn stderr $ "edges_loaded=" ++ show (length categoryParents)
    hPutStrLn stderr $ "total_weight_sum=" ++ show totalWsum
    mapM_ (\\(s, w) ->
        hPutStrLn stderr ("seed_result=" ++ s ++ "\\tw=" ++ show w)
      ) results
'),
    close(Stream).
