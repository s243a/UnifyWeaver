% test_wam_haskell_lowered_t5.pl
%
% End-to-end execution test for the Haskell T5 lowering — "multi-clause as a
% first-argument dispatch" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md), ported from the
% Scala/Rust/Go emitters via the shared wam_clause_chain front-end.
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers ALL of its clauses to native Haskell, selected by a
% deref-and-match cascade, instead of lowering only clause 1 and reaching
% clauses 2+ through the interpreter on backtrack. When the first argument is
% bound this is deterministic dispatch with no interpreter hop; when it is
% unbound it defers to the interpreter via the same choice-point / backtrack /
% run fallback the ordinary multi-clause path uses (which enumerates every
% clause, binding the variable in turn).
%
% Pins (the cases preload a BOUND first arg, exercising every clause incl. the
% non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when `ghc` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_haskell_target').

:- dynamic user:color/1.
:- dynamic user:sz/2.
:- dynamic user:op/2.

user:color(red).
user:color(green).
user:color(blue).

user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).

user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
user:op(neg, R) :- R is 0 - 1.

ghc_available :-
    catch(( process_create(path(ghc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_haskell_lowered_t5, [condition(ghc_available)]).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_haskell_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM Haskell project with the lowered emitter enabled.
    write_wam_haskell_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5proj'), emit_mode(functions)], Dir),
    % Sanity: the generated lowered code must be the T5 dispatch.
    atomic_list_concat([Dir, '/src/Lowered.hs'], LoweredPath),
    ( exists_file(LoweredPath) -> read_file_to_string(LoweredPath, LSrc, []) ; LSrc = "" ),
    assertion(sub_string(LSrc, _, _, _, "t5fallback")),
    % 2. Test harness calling the lowered functions directly (bound first arg).
    atomic_list_concat([Dir, '/src/TestMain.hs'], TestPath),
    haskell_t5_source(Src),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]), write(S, Src), close(S)),
    % 3. Compile + run.
    atomic_list_concat([Dir, '/src'], SrcDir),
    atomic_list_concat([Dir, '/t5_test'], Bin),
    format(atom(Cmd),
        'ghc --make -i~w ~w/TestMain.hs -o ~w -main-is Main 2>&1 && ~w',
        [SrcDir, SrcDir, Bin, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 14 PASS")
    ->  true
    ;   format(user_error, "~n[haskell t5 test output]~n~w~n", [OutStr]),
        throw(haskell_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_haskell_lowered_t5).

% Calls each lowered function (WamContext -> WamState -> Maybe WamState) with
% a BOUND first argument and checks Just (success) / Nothing (failure). The
% cases exercise every clause including the non-first ones (green/blue,
% medium/large, mul/neg) — the T5 payoff — plus the no-match cases.
haskell_t5_source(
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
              -- op/2's clause bodies run an is/2 over a +/2 structure, and
              -- evalArith resolves the operator name through the context
              -- intern table — so it must hold the compile-time atom table.
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
        [ (\"color(red)\", runP lowered_color_1 [(1,a \"red\")], True)
        , (\"color(green)\", runP lowered_color_1 [(1,a \"green\")], True)
        , (\"color(blue)\", runP lowered_color_1 [(1,a \"blue\")], True)
        , (\"color(yellow)\", runP lowered_color_1 [(1,a \"yellow\")], False)
        , (\"sz(small,1)\", runP lowered_sz_2 [(1,a \"small\"),(2,i 1)], True)
        , (\"sz(medium,2)\", runP lowered_sz_2 [(1,a \"medium\"),(2,i 2)], True)
        , (\"sz(large,3)\", runP lowered_sz_2 [(1,a \"large\"),(2,i 3)], True)
        , (\"sz(small,2)\", runP lowered_sz_2 [(1,a \"small\"),(2,i 2)], False)
        , (\"sz(big,1)\", runP lowered_sz_2 [(1,a \"big\"),(2,i 1)], False)
        , (\"op(add,2)\", runP lowered_op_2 [(1,a \"add\"),(2,i 2)], True)
        , (\"op(mul,6)\", runP lowered_op_2 [(1,a \"mul\"),(2,i 6)], True)
        , (\"op(neg,-1)\", runP lowered_op_2 [(1,a \"neg\"),(2,i (-1))], True)
        , (\"op(add,3)\", runP lowered_op_2 [(1,a \"add\"),(2,i 3)], False)
        , (\"op(div,1)\", runP lowered_op_2 [(1,a \"div\"),(2,i 1)], False)
        ]
      fails = [ n | (n,got,want) <- cases, got /= want ]
  mapM_ (\\(n,got,want) -> if got/=want then putStrLn (\"FAIL \" ++ n ++ \": got \" ++ show got ++ \" want \" ++ show want) else return ()) cases
  if null fails then putStrLn (\"ALL \" ++ show (length cases) ++ \" PASS\") else putStrLn (show (length fails) ++ \" FAILURES\")
").
