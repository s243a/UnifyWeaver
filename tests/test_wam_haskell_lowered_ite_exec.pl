% test_wam_haskell_lowered_ite_exec.pl
%
% End-to-end execution test for Haskell if-then-else / negation / once
% lowering (emit_mode(functions)).
%
% Generates a WAM Haskell project with the lowered emitter enabled,
% compiles it with GHC, and runs a harness that calls each lowered function
% and asserts the boolean (Just/Nothing) outcome. Counterpart to the Go,
% Rust and C++ tests. Pins:
%
%   * sequential ITEs   — hseqite(10,pos,small) must fail;
%   * nested ITEs       — hnestite/2 (the inner block sits in the then-arm);
%   * negation (\+)     — hneg/1 (commit is the !/0 builtin, not cut_ite);
%   * simple ITEs       — hite/2.
%
% Before the shared-structurer conversion the flat split heuristic was not
% nesting-aware and only recognised cut_ite, so negation and nested ITEs
% failed to lower (in functions/mixed mode that failed generation). The
% runtime also lacked true/0 and fail/0, which negation needs. Skipped when
% GHC is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_haskell_target').

:- dynamic user:hite/2.
:- dynamic user:hneg/1.
:- dynamic user:hseqite/3.
:- dynamic user:hnestite/2.

user:hite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:hneg(X)          :- \+ X > 0.
user:hseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:hnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

ghc_available :-
    catch(( process_create(path(ghc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_haskell_lowered_ite_exec, [condition(ghc_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_haskell_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the WAM Haskell project with the lowered emitter enabled.
    write_wam_haskell_project(
        [user:hite/2, user:hneg/1, user:hseqite/3, user:hnestite/2],
        [module_name('iteproj'), emit_mode(functions)], Dir),
    % 2. Test harness calling the lowered functions directly. Atom IDs are
    %    resolved by name through the generated compile-time intern table
    %    (robust to intern-order changes).
    atomic_list_concat([Dir, '/src/TestMain.hs'], TestPath),
    haskell_test_source(Src),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]), write(S, Src), close(S)),
    % 3. Compile + run.
    atomic_list_concat([Dir, '/src'], SrcDir),
    atomic_list_concat([Dir, '/ite_test'], Bin),
    format(atom(Cmd),
        'ghc --make -i~w ~w/TestMain.hs -o ~w -main-is Main 2>&1 && ~w',
        [SrcDir, SrcDir, Bin, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[haskell ite test output]~n~w~n", [OutStr]),
        throw(haskell_test_failed(Status))
    ).

:- end_tests(wam_haskell_lowered_ite_exec).

% Calls each lowered function (WamContext -> WamState -> Maybe WamState)
% with the query arguments preloaded into the A-registers and checks
% Just (success) / Nothing (failure). hseqite(10,pos,small) and
% hnestite(20,small) are the sequential/nested discriminators; hneg
% exercises the !/0-commit negation path (needs true/0 + fail/0).
haskell_test_source(
"module Main where
import qualified Data.HashMap.Strict as Map
import qualified Data.IntMap.Strict as IM
import WamTypes
import WamRuntime ()
import Lowered
import Predicates (compileTimeAtomTable)

runP :: (WamContext -> WamState -> Maybe WamState) -> [(Int, Value)] -> Bool
runP f regs =
  let ctx = (mkContext [] Map.empty) { wcLoweredPredicates = loweredPredicates }
      s0  = emptyState { wsRegs = IM.fromList regs }
  in case f ctx s0 of { Just _ -> True; Nothing -> False }

i :: Int -> Value
i = Integer . fromIntegral
a :: String -> Value
a s = Atom (internAtomPure compileTimeAtomTable s)

main :: IO ()
main = do
  let cases =
        [ (\"hite(5,pos)\", runP lowered_hite_2 [(1,i 5),(2,a \"pos\")], True)
        , (\"hite(5,nonpos)\", runP lowered_hite_2 [(1,i 5),(2,a \"nonpos\")], False)
        , (\"hite(-1,nonpos)\", runP lowered_hite_2 [(1,i (-1)),(2,a \"nonpos\")], True)
        , (\"hite(-1,pos)\", runP lowered_hite_2 [(1,i (-1)),(2,a \"pos\")], False)
        , (\"hneg(5)\", runP lowered_hneg_1 [(1,i 5)], False)
        , (\"hneg(-1)\", runP lowered_hneg_1 [(1,i (-1))], True)
        , (\"hneg(0)\", runP lowered_hneg_1 [(1,i 0)], True)
        , (\"hseqite(10,pos,big)\", runP lowered_hseqite_3 [(1,i 10),(2,a \"pos\"),(3,a \"big\")], True)
        , (\"hseqite(10,pos,small)\", runP lowered_hseqite_3 [(1,i 10),(2,a \"pos\"),(3,a \"small\")], False)
        , (\"hseqite(3,pos,small)\", runP lowered_hseqite_3 [(1,i 3),(2,a \"pos\"),(3,a \"small\")], True)
        , (\"hseqite(-1,nonpos,small)\", runP lowered_hseqite_3 [(1,i (-1)),(2,a \"nonpos\"),(3,a \"small\")], True)
        , (\"hnestite(20,big)\", runP lowered_hnestite_2 [(1,i 20),(2,a \"big\")], True)
        , (\"hnestite(5,small)\", runP lowered_hnestite_2 [(1,i 5),(2,a \"small\")], True)
        , (\"hnestite(-1,neg)\", runP lowered_hnestite_2 [(1,i (-1)),(2,a \"neg\")], True)
        , (\"hnestite(20,small)\", runP lowered_hnestite_2 [(1,i 20),(2,a \"small\")], False)
        ]
      fails = [ n | (n,got,want) <- cases, got /= want ]
  mapM_ (\\(n,got,want) -> if got/=want then putStrLn (\"FAIL \" ++ n ++ \": got \" ++ show got ++ \" want \" ++ show want) else return ()) cases
  if null fails then putStrLn (\"ALL \" ++ show (length cases) ++ \" PASS\") else putStrLn (show (length fails) ++ \" FAILURES\")
").
