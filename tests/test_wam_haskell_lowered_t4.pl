% test_wam_haskell_lowered_t4.pl
%
% End-to-end execution test for the Haskell T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from
% Scala/Rust/Go/C++.
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers to ALL clauses inline: each clause becomes a
% `WamState -> Maybe WamState` do-block and the function tries them in order on
% the SAME input state (`c1 s_init `orElse` c2 s_init `orElse` ...`), taking
% the first Just. Haskell's immutability gives a free per-clause restore, so —
% unlike the imperative targets — no snapshot/restore and no choice point are
% needed; the interpreter is never entered for the predicate.
%
% Pins (BOUND first arg; the payoff is the non-first clauses running natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3);
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 body).
%
% Skipped automatically when `ghc` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_haskell_target').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

ghc_available :-
    catch(( process_create(path(ghc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_haskell_lowered_t4, [condition(ghc_available)]).

test(t4_exec_parity) :-
    Dir = 'output/test_wam_haskell_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_haskell_project(
        [user:grade/2, user:rel/2],
        [module_name('t4proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/src/Lowered.hs'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "T4 all-clauses inline")),
    atomic_list_concat([Dir, '/src/TestMain.hs'], TestPath),
    haskell_t4_source(Src),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]), write(S, Src), close(S)),
    atomic_list_concat([Dir, '/src'], SrcDir),
    atomic_list_concat([Dir, '/t4_test'], Bin),
    format(atom(Cmd),
        'ghc --make -i~w ~w/TestMain.hs -o ~w -main-is Main 2>&1 && ~w',
        [SrcDir, SrcDir, Bin, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 10 PASS")
    ->  true
    ;   format(user_error, "~n[haskell t4 test output]~n~w~n", [OutStr]),
        throw(haskell_t4_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_haskell_lowered_t4).

% Calls each lowered function with the query args preloaded (first arg bound)
% and checks Just (success) / Nothing (failure). Exercises the non-first
% clauses (grade clauses 2 & 3, rel clause 2) — the T4 payoff — plus no-match
% cases.
haskell_t4_source(
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

a :: String -> Value
a s = Atom (internAtomPure compileTimeAtomTable s)

main :: IO ()
main = do
  let cases =
        [ (\"grade(alice,a)\", runP lowered_grade_2 [(1,a \"alice\"),(2,a \"a\")], True)
        , (\"grade(bob,b)\",   runP lowered_grade_2 [(1,a \"bob\"),(2,a \"b\")], True)
        , (\"grade(alice,c)\", runP lowered_grade_2 [(1,a \"alice\"),(2,a \"c\")], True)
        , (\"grade(alice,b)\", runP lowered_grade_2 [(1,a \"alice\"),(2,a \"b\")], False)
        , (\"grade(carol,a)\", runP lowered_grade_2 [(1,a \"carol\"),(2,a \"a\")], False)
        , (\"grade(bob,c)\",   runP lowered_grade_2 [(1,a \"bob\"),(2,a \"c\")], False)
        , (\"rel(p,one)\", runP lowered_rel_2 [(1,a \"p\"),(2,a \"one\")], True)
        , (\"rel(q,two)\", runP lowered_rel_2 [(1,a \"q\"),(2,a \"two\")], True)
        , (\"rel(p,two)\", runP lowered_rel_2 [(1,a \"p\"),(2,a \"two\")], False)
        , (\"rel(q,one)\", runP lowered_rel_2 [(1,a \"q\"),(2,a \"one\")], False)
        ]
      fails = [ n | (n,got,want) <- cases, got /= want ]
  mapM_ (\\(n,got,want) -> if got/=want then putStrLn (\"FAIL \" ++ n ++ \": got \" ++ show got ++ \" want \" ++ show want) else return ()) cases
  if null fails then putStrLn (\"ALL \" ++ show (length cases) ++ \" PASS\") else putStrLn (show (length fails) ++ \" FAILURES\")
").
