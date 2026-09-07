:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_haskell_frameless_ite_level.pl
%
% Probe for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write* form
% (dismissed as unreachable by the original fleet audit; proven reachable on
% wam_rust in ledger row D50, and reproduced here on wam_haskell).
%
% The shape
% ---------
% `compile_if_then_else/7` in the shared emitter (`wam_target.pl`) reserves a
% permanent Y register for the if-then-else barrier AFTER it has decided
% whether the clause needs an environment. So a clause needing no environment
% still gets `get_level Yn` ... `cut Yn` with NO `allocate`. `sat/2` clause 2
% below is exactly that clause (`\+ G` inlines to `(G -> fail ; true)` under
% `ite_use_y_level(true)`, which the Haskell target always enables), and
% `pick_a/4` / `pick_b/4` are callers that DO hold an environment with live Y
% registers across the call.
%
% Both Haskell interpreters used to mishandle it:
%
%   * `stepST` (the DEFAULT `runMutableRegs` path) wrote the level through
%     `putRegST`, whose Y branch targets the TOPMOST env frame -- the CALLER's
%     when the callee has no `allocate`. `get_level Y1` therefore replaced the
%     caller's permanent variable Y1 with a choice-point depth: a silent wrong
%     answer, not a crash.
%   * the pure `step` path escaped the clobber only by accident (it wrote the
%     flat `wsRegs` map, which Y access never reads), but it truncated the
%     choice-point stack with `take n` on a NEWEST-FIRST list, which keeps the
%     n youngest choice points instead of the n oldest.
%
% Both now keep the barrier on the if-then-else's own choice point
% (`cpLevels` / `mcpLevels`, `lookupIteLevel` / `lookupIteLevelST`), so the
% level never touches a register and is per-activation for free.
%
% What runs where
% ---------------
%   * the emission + generated-source tests run everywhere: they pin that the
%     shared emitter still produces the hazard shape, and that neither
%     interpreter routes `get_level` through a register any more;
%   * the execution test needs GHC and is skipped when it is absent.
%
%   swipl -q -g run_tests -t halt tests/test_wam_haskell_frameless_ite_level.pl

:- module(test_wam_haskell_frameless_ite_level,
          [test_wam_haskell_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_haskell_target',
              [write_wam_haskell_project/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4, user:wpick/3.

% --- the probe program (SWI is the oracle for the expectations below) ------

user:lt(A, B) :- A < B.

% Multi-clause callee. Clause 2 needs NO environment: its only permanent
% would be the barrier the emitter reserves for the inlined negation, so it
% emits `get_level Y1` with no `allocate`.
user:sat(_V, any).
user:sat(V, gte(G)) :- \+ user:lt(V, G).

% Callers that DO hold an environment across the call. Y numbering follows
% first use in the body, so pick_a parks the (unbound) output in Y1 and
% pick_b parks the (bound) input tag in Y1.
user:pick_a(Ver, C, Tag, Out) :- user:sat(Ver, C), Out = Tag.
user:pick_b(Ver, C, Tag, Out) :- user:sat(Ver, C), Tag = Out.

% Recursive caller: several activations of sat/2's if-then-else are live at
% once, each needing its own barrier.
user:wpick(0, Tag, Tag).
user:wpick(N, Tag, Out) :-
    N > 0,
    user:sat(N, gte(1)),
    M is N - 1,
    user:wpick(M, Tag, Out).

probe_preds([user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4,
             user:wpick/3]).

probe_dir('output/test_wam_haskell_frameless_ite_level').

ghc_available :-
    catch(( process_create(path(ghc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

generate_probe_project(Dir) :-
    probe_dir(Dir),
    probe_preds(Preds),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_haskell_project(Preds, [module_name(hfprobe)], Dir).

test_wam_haskell_frameless_ite_level :-
    run_tests(wam_haskell_frameless_ite_level).

:- begin_tests(wam_haskell_frameless_ite_level).

% The probe is only meaningful while the shared emitter still produces the
% hazard shape, so assert it on the emitted text.
test(sat_clause_has_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

% Source-level pin: the barrier lives on the choice point, not in a register.
test(runtime_keeps_levels_on_the_choice_point) :-
    once(generate_probe_project(Dir)),
    atomic_list_concat([Dir, '/src/WamTypes.hs'], TypesPath),
    atomic_list_concat([Dir, '/src/WamRuntime.hs'], RtPath),
    read_file_to_string(TypesPath, Types, []),
    read_file_to_string(RtPath, Rt, []),
    % both choice-point records carry a barrier-level map
    assertion(sub_string(Types, _, _, _, "cpLevels")),
    assertion(sub_string(Types, _, _, _, "mcpLevels")),
    % neither GetLevel arm writes the operand into a register any more
    assertion(\+ sub_string(Rt, _, _, _, "putRegST regs reg (Integer")),
    assertion(\+ sub_string(Rt, _, _, _,
                            "wsRegs = IM.insert reg (Integer (wsCPsLen s))")),
    % and both Cut arms look the level up on the choice-point chain
    assertion(sub_string(Rt, _, _, _, "lookupIteLevel reg (wsCPs s)")),
    assertion(sub_string(Rt, _, _, _, "lookupIteLevelST reg (pwCPs pw)")).

% The generated bytecode really does put GetLevel in an Allocate-less clause
% and a Y register in the caller's frame -- i.e. the two ends of the hazard.
test(generated_bytecode_shows_the_hazard_shape) :-
    probe_dir(Dir),
    ( exists_directory(Dir) -> true ; generate_probe_project(Dir) ),
    atomic_list_concat([Dir, '/src/Predicates.hs'], PredPath),
    read_file_to_string(PredPath, Src, []),
    assertion(sub_string(Src, _, _, _, "GetLevel 201")),
    assertion(sub_string(Src, _, _, _, "Cut 201")),
    assertion(sub_string(Src, _, _, _, "GetVariable 201")).

test(matches_swi, [condition(ghc_available)]) :-
    once(run_frameless_probe).

:- end_tests(wam_haskell_frameless_ite_level).

% --- execution harness (GHC only) ------------------------------------------

run_frameless_probe :-
    generate_probe_project(Dir),
    atomic_list_concat([Dir, '/src/TestMain.hs'], TestPath),
    haskell_test_source(Src),
    setup_call_cleanup(open(TestPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    atomic_list_concat([Dir, '/src'], SrcDir),
    atomic_list_concat([Dir, '/frameless_test'], Bin),
    format(atom(Cmd),
        'ghc --make -i~w ~w/TestMain.hs -o ~w -main-is Main 2>&1 && ~w',
        [SrcDir, SrcDir, Bin, Bin]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0), sub_string(OutStr, _, _, _, "ALL PASS")
    ->  true
    ;   format(user_error,
               "~n[haskell frameless-ITE-level output]~n~w~n", [OutStr]),
        throw(wam_haskell_frameless_ite_level_failed(Status))
    ).

% The expectations below are the SWI answers for the same goals:
%   sat(3, any)                    -> true
%   sat(3, gte(1))                 -> true      (\+ 3 < 1)
%   sat(0, gte(1))                 -> false
%   pick_a(3, gte(1), 77, Out)     -> Out = 77
%   pick_b(3, gte(1), 77, Out)     -> Out = 77
%   pick_a(0, gte(1), 77, _)       -> false
%   wpick(4, 77, Out)              -> Out = 77
% Every case is run through BOTH interpreters: `runMutableRegs` (the default
% ST path, which held the defect) and `run` (the pure path).
haskell_test_source(
"module Main where
import qualified Data.HashMap.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Maybe (fromMaybe)
import WamTypes
import WamRuntime
import Predicates

ctx0 :: WamContext
ctx0 = (mkContext allCode allLabels) { wcInternTable = compileTimeAtomTable }

gteV :: Int -> Value
gteV n = Str (internAtomPure compileTimeAtomTable \"gte/1\") [Integer n]

anyV :: Value
anyV = Atom (internAtomPure compileTimeAtomTable \"any\")

outVid :: Int
outVid = 999999

-- Run `entry` with the given A-registers. Register slots holding Nothing get
-- a fresh unbound variable whose binding is read back afterwards.
runQ :: (WamContext -> WamState -> Maybe WamState)
     -> String -> [Maybe Value] -> (Bool, Maybe Value)
runQ runner entry args =
  let pc = fromMaybe 0 (Map.lookup entry allLabels)
      slots = zip [1 ..] args
      regs = [ (i, fromMaybe (Unbound outVid) mv) | (i, mv) <- slots ]
      s0 = emptyState { wsPC = pc, wsRegs = IM.fromList regs, wsCP = 0 }
  in case runner ctx0 s0 of
       Nothing -> (False, Nothing)
       Just s1 -> (True, derefVar (wsBindings s1) <$>
                           IM.lookup outVid (wsBindings s1))

cases :: (WamContext -> WamState -> Maybe WamState)
      -> [(String, (Bool, Maybe Value), (Bool, Maybe Value))]
cases r =
  [ (\"sat(3,any)\",       runQ r \"sat/2\" [Just (Integer 3), Just anyV],
                          (True, Nothing))
  , (\"sat(3,gte(1))\",    runQ r \"sat/2\" [Just (Integer 3), Just (gteV 1)],
                          (True, Nothing))
  , (\"sat(0,gte(1))\",    runQ r \"sat/2\" [Just (Integer 0), Just (gteV 1)],
                          (False, Nothing))
  , (\"pick_a ok\",        runQ r \"pick_a/4\"
                            [Just (Integer 3), Just (gteV 1),
                             Just (Integer 77), Nothing],
                          (True, Just (Integer 77)))
  , (\"pick_b ok\",        runQ r \"pick_b/4\"
                            [Just (Integer 3), Just (gteV 1),
                             Just (Integer 77), Nothing],
                          (True, Just (Integer 77)))
  , (\"pick_a fail\",      runQ r \"pick_a/4\"
                            [Just (Integer 0), Just (gteV 1),
                             Just (Integer 77), Nothing],
                          (False, Nothing))
  , (\"pick_b fail\",      runQ r \"pick_b/4\"
                            [Just (Integer 0), Just (gteV 1),
                             Just (Integer 77), Nothing],
                          (False, Nothing))
  , (\"wpick(4,77,Out)\",  runQ r \"wpick/3\"
                            [Just (Integer 4), Just (Integer 77), Nothing],
                          (True, Just (Integer 77)))
  ]

main :: IO ()
main = do
  let runs = [(\"runMutableRegs\", runMutableRegs), (\"run\", run)]
      bad = [ (lbl, n, got, want)
            | (lbl, r) <- runs, (n, got, want) <- cases r, got /= want ]
  mapM_ (\\(lbl, n, got, want) ->
           putStrLn (\"FAIL [\" ++ lbl ++ \"] \" ++ n ++ \": got \" ++ show got
                     ++ \" want \" ++ show want)) bad
  if null bad then putStrLn \"ALL PASS\"
              else putStrLn (show (length bad) ++ \" FAILURES\")
").
