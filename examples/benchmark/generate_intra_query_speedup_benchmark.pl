:- initialization(main, main).
:- use_module('../../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../../src/unifyweaver/targets/wam_target').

%% generate_intra_query_speedup_benchmark.pl
%%
%% Speedup-oriented intra-query parallelism benchmark. Unlike the
%% earlier benchmarks that had only 2 clauses (base + recursive) for
%% the fork, this one generates a synthetic multi_ancestor/3 predicate
%% with N clauses — one per seed category — so the ParTryMeElse fork
%% has N balanced branches to distribute across cores.
%%
%% Usage:
%%   swipl -q -s generate_intra_query_speedup_benchmark.pl -- \
%%       <facts-dir> <output-dir> [num-clauses]
%%
%% The resulting binary accepts:
%%   [facts-dir]
%% and prints timing + result to stderr.

workload_path(Path) :-
    source_file(workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'intra_query_seed.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsDir, OutputDir, NStr]
    ->  atom_number(NStr, NumClauses)
    ;   Argv = [FactsDir, OutputDir]
    ->  NumClauses = 10
    ;   format(user_error,
            'Usage: ... -- <facts-dir> <output-dir> [num-clauses]~n', []),
        halt(1)
    ),
    workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),

    % Load facts to find top-N seeds by out-degree
    directory_file_path(FactsDir, 'category_parent.tsv', CpPath),
    load_tsv_pairs(CpPath, Pairs),
    find_top_seeds(Pairs, NumClauses, Seeds),
    format(user_error, '[speedup] top-~w seeds: ~w~n', [NumClauses, Seeds]),

    % Assert multi_ancestor/3 with one clause per seed
    assert_multi_ancestor_clauses(Seeds),
    % Assert total_weight/1 with aggregate_all(sum, ...)
    assert_total_weight,

    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    Predicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4,
        user:multi_ancestor/3,
        user:total_weight/1
    ],
    Options = [
        module_name('wam-speedup-bench'),
        emit_mode(interpreter),
        no_kernels(true),
        use_hashmap(false),
        intra_query_parallel(true)
    ],
    write_wam_haskell_project(Predicates, Options, OutputDir),

    directory_file_path(OutputDir, 'src/Main.hs', MainPath),
    write_benchmark_main(MainPath, FactsDir, Seeds),
    halt(0).
main :- halt(1).

%% load_tsv_pairs(+Path, -Pairs) — read category_parent.tsv
load_tsv_pairs(Path, Pairs) :-
    read_file_to_string(Path, Content, []),
    split_string(Content, "\n", "\r", Lines),
    Lines = [_Header|DataLines],
    findall(A-B,
        (   member(Line, DataLines),
            Line \= "",
            split_string(Line, "\t", "", [AS, BS|_]),
            atom_string(A, AS),
            atom_string(B, BS)
        ),
        Pairs).

%% find_top_seeds(+Pairs, +N, -Seeds)
find_top_seeds(Pairs, N, Seeds) :-
    findall(Cat, member(Cat-_, Pairs), AllCats),
    msort(AllCats, Sorted),
    clumped(Sorted, Counted),
    sort(2, @>=, Counted, ByDegree),
    length(TopN, N),
    append(TopN, _, ByDegree),
    maplist([Cat-_, Cat]>>true, TopN, Seeds).

%% assert_multi_ancestor_clauses(+Seeds)
%  For each seed, assert a clause:
%    multi_ancestor(Anc, Hops, seed_atom) :-
%        category_ancestor(seed_atom, Anc, Hops, [seed_atom]).
assert_multi_ancestor_clauses([]).
assert_multi_ancestor_clauses([Seed|Rest]) :-
    Clause = (
        user:multi_ancestor(Anc, Hops, Seed) :-
            category_ancestor(Seed, Anc, Hops, [Seed])
    ),
    assertz(Clause),
    assert_multi_ancestor_clauses(Rest).

%% assert_total_weight/0
%  total_weight(Sum) :-
%      aggregate_all(sum(W),
%          (multi_ancestor(_, Hops, _),
%           H is Hops + 1,
%           W is H ** (-5)),
%          Sum).
assert_total_weight :-
    assertz((
        user:total_weight(Sum) :-
            aggregate_all(sum(W),
                (multi_ancestor(_, Hops, _),
                 H is Hops + 1,
                 W is H ** (-5)),
                Sum)
    )).

write_benchmark_main(Path, FactsDir, Seeds) :-
    length(Seeds, NSeed),
    open(Path, write, Stream),
    format(Stream,
'{-# LANGUAGE BangPatterns #-}
module Main where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.List (foldl\')
import Data.Maybe (fromMaybe)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Data.Time.Clock (getCurrentTime, diffUTCTime)
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
          (allC, allL, _) = foldl (\\(c, l, p) (i, f) ->
              let (fc, fl, np) = buildFact i f p in (c ++ fc, l ++ fl, np))
            ([], [], pc) (zip [0..] facts)
      in (allC, allL, pc + length allC)

main :: IO ()
main = do
    args <- getArgs
    let factsDir = case args of { (d:_) -> d; _ -> "~w" }

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

    let !ctx = (mkContext mergedCode mergedLabels)
            { wcForeignConfig = Map.singleton "max_depth" 6
            , wcLoweredPredicates = Lowered.loweredPredicates
            }

    -- Call total_weight/1 via WAM. A1 = Unbound(output).
    let wsVarId = 1000000
        pcStart = fromMaybe 1 $ Map.lookup "total_weight/1" mergedLabels
        s0 = emptyState
            { wsPC = pcStart
            , wsRegs = IM.fromList [(1, Unbound wsVarId)]
            , wsCP = 0
            }

    t2 <- getCurrentTime
    let !result = case run ctx s0 of
          Just s1 -> case IM.lookup wsVarId (wsBindings s1) of
            Just v -> case extractDouble (derefVar (wsBindings s1) v) of
              Just ws -> ws
              Nothing -> 0.0
            Nothing -> 0.0
          Nothing -> 0.0
    t3 <- getCurrentTime
    let queryMs = round (diffUTCTime t3 t2 * 1000) :: Int
        totalMs = round (diffUTCTime t3 t0 * 1000) :: Int

    hPutStrLn stderr "mode=intra_query_speedup_benchmark"
    hPutStrLn stderr $ "load_ms=" ++ show loadMs
    hPutStrLn stderr $ "query_ms=" ++ show queryMs
    hPutStrLn stderr $ "total_ms=" ++ show totalMs
    hPutStrLn stderr $ "clauses=~w"
    hPutStrLn stderr $ "total_weight_sum=" ++ show result

extractDouble :: Value -> Maybe Double
extractDouble (Integer n) = Just (fromIntegral n)
extractDouble (Float f)   = Just f
extractDouble (Atom s)    = case reads s of [(h, "")] -> Just h; _ -> Nothing
extractDouble _           = Nothing
', [FactsDir, NSeed]),
    close(Stream).
