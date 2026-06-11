% test_wam_haskell_lowered_t6.pl
%
% End-to-end test for the Haskell T6 lowering — first-argument indexing via a
% `case` on the interned atom id (GHC compiles a dense Int case to a jump
% table), lowering type T6 in docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% Haskell is int-interned: atoms render as `Atom <id>` (val_hs), so the T5 guard
% chain compares `v == Atom n`. T6 replaces that linear chain with a single
% `case t6i of n -> clauseK`. Benchmarked 1.4x at 8, 2.5x at 64, 4.5x at 256
% (GHC -O2; the guard chain is partly optimised but the Int case still wins).
%
% Gated like Rust/C++/F#/Go: T6 fires only when every clause discriminates on a
% distinct ATOM and there are >= t6_min_clauses of them (default 8). Below the
% threshold the few-clause predicate stays the T5 guard chain.
%
% Skipped automatically when `ghc` is not on PATH.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_target').
:- use_module('../src/unifyweaver/targets/wam_haskell_lowered_emitter').

:- dynamic user:shade/1, user:grade/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

user:grade(g01, R) :- R is 1 + 0.
user:grade(g02, R) :- R is 1 + 1.
user:grade(g03, R) :- R is 1 + 2.
user:grade(g04, R) :- R is 1 + 3.
user:grade(g05, R) :- R is 1 + 4.
user:grade(g06, R) :- R is 1 + 5.
user:grade(g07, R) :- R is 1 + 6.
user:grade(g08, R) :- R is 1 + 7.
user:grade(g09, R) :- R is 1 + 8.
user:grade(g10, R) :- R is 1 + 9.

user:few(a). user:few(b). user:few(c).

ghc_available :-
    catch(( process_create(path(ghc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_haskell_lowered_t6, [condition(ghc_available)]).

% Codegen gate: the many-clause atom predicates emit the case-on-id (T6); the
% few-clause one stays the guard chain (T5). Threshold is configurable.
test(gate_picks_t6_for_many_t5_for_few) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    lower_predicate_to_haskell(shade/1, Ws, [], lowered(_, _, ShadeCode)),
    assertion(sub_string(ShadeCode, _, _, _, "T6 first-argument indexing")),
    assertion(sub_string(ShadeCode, _, _, _, "case t6i of")),
    wam_target:compile_predicate_to_wam(grade/2, [], Wg),
    lower_predicate_to_haskell(grade/2, Wg, [], lowered(_, _, GradeCode)),
    assertion(sub_string(GradeCode, _, _, _, "T6 first-argument indexing")),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_haskell(few/1, Wf, [], lowered(_, _, FewCode)),
    assertion(\+ sub_string(FewCode, _, _, _, "T6 first-argument indexing")),
    lower_predicate_to_haskell(few/1, Wf, [t6_min_clauses(3)], lowered(_, _, FewT6)),
    assertion(sub_string(FewT6, _, _, _, "T6 first-argument indexing")).

% Build + run: the T6 case dispatch returns the Prolog-correct result for clause
% hits (incl. non-first clauses), no-match, and a grade remainder mismatch.
test(t6_exec) :-
    Dir = 'output/test_wam_haskell_t6_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_haskell_project(
        [user:shade/1, user:grade/2, user:few/1],
        [module_name('t6proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/src/Lowered.hs'], LoweredPath),
    read_file_to_string(LoweredPath, LSrc, []),
    assertion(sub_string(LSrc, _, _, _, "T6 first-argument indexing")),
    atomic_list_concat([Dir, '/src/TestMain.hs'], TestPath),
    haskell_t6_source(Src),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/src'], SrcDir),
    atomic_list_concat([Dir, '/t6_test'], Bin),
    format(atom(Cmd),
        'ghc --make -i~w ~w/TestMain.hs -o ~w -main-is Main 2>&1 && ~w',
        [SrcDir, SrcDir, Bin, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 11 PASS")
    ->  true
    ;   format(user_error, "~n[haskell t6 test output]~n~w~n", [OutStr]),
        throw(haskell_t6_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_haskell_lowered_t6).

haskell_t6_source(
"module Main where
import qualified Data.HashMap.Strict as Map
import qualified Data.IntMap.Strict as IM
import WamTypes
import WamRuntime ()
import Lowered
import Predicates (compileTimeAtomTable)

runP :: (WamContext -> WamState -> Maybe WamState) -> [(Int, Value)] -> Bool
runP f regs =
  let ctx = (mkContext [] Map.empty)
              { wcLoweredPredicates = loweredPredicates
              , wcInternTable = compileTimeAtomTable }
      s0  = emptyState { wsRegs = IM.fromList regs }
  in case f ctx s0 of { Just _ -> True; Nothing -> False }

i :: Int -> Value
i = Integer . fromIntegral
a :: String -> Value
a s = Atom (internAtomPure compileTimeAtomTable s)

main :: IO ()
main = do
  let cases =
        [ (\"shade(s01)\", runP lowered_shade_1 [(1,a \"s01\")], True)
        , (\"shade(s05)\", runP lowered_shade_1 [(1,a \"s05\")], True)
        , (\"shade(s10)\", runP lowered_shade_1 [(1,a \"s10\")], True)
        , (\"shade(zz)\", runP lowered_shade_1 [(1,a \"zz\")], False)
        , (\"grade(g01,1)\", runP lowered_grade_2 [(1,a \"g01\"),(2,i 1)], True)
        , (\"grade(g05,5)\", runP lowered_grade_2 [(1,a \"g05\"),(2,i 5)], True)
        , (\"grade(g10,10)\", runP lowered_grade_2 [(1,a \"g10\"),(2,i 10)], True)
        , (\"grade(g05,9)\", runP lowered_grade_2 [(1,a \"g05\"),(2,i 9)], False)
        , (\"grade(zz,1)\", runP lowered_grade_2 [(1,a \"zz\"),(2,i 1)], False)
        , (\"few(b)\", runP lowered_few_1 [(1,a \"b\")], True)
        , (\"few(z)\", runP lowered_few_1 [(1,a \"z\")], False)
        ]
      fails = [ n | (n,got,want) <- cases, got /= want ]
  mapM_ (\\(n,got,want) -> if got/=want then putStrLn (\"FAIL \" ++ n ++ \": got \" ++ show got ++ \" want \" ++ show want) else return ()) cases
  if null fails then putStrLn (\"ALL \" ++ show (length cases) ++ \" PASS\") else putStrLn (show (length fails) ++ \" FAILURES\")
").
