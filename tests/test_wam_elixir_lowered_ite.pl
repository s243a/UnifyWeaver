:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_elixir_lowered_ite.pl
%
% End-to-end execution test for WAM-Elixir if-then-else / negation / once
% lowering. Counterpart to the Go / Rust / C++ / Haskell / F# / Clojure /
% LLVM lowered-ITE exec tests.
%
% The WAM-Elixir lowered emitter compiles ( Cond -> Then ; Else ), \+ Goal
% and once/1 through genuine choice points (try_me_else pushes a CP for the
% else branch; the soft cut commits by dropping it). That control flow was
% UNIMPLEMENTED before the change this test guards: cut_ite raised
% "TODO: cut_ite" at runtime, the then-branch's `jump` to the continuation
% raised "TODO: jump", and the else branch never reached its continuation
% (it returned a raw mid-execution state). So every predicate containing an
% if-then-else crashed or returned a wrong answer. This test pins:
%
%   * simple ITE         — eite/2;
%   * negation (\+)       — eneg/1 (commit is the !/0 builtin: then = fail/0,
%                          else = true/0, run after a trail rollback);
%   * sequential ITEs    — eseqite/3 (two sibling blocks);
%   * nested ITEs         — enestite/2 (inner block in the then-arm — its
%                          inner try_me_else is a body-position choice point).
%
% It also exercises the shared head-arg setup (allocate / get_variable
% before the try_me_else) being snapshotted by the choice point so the else
% branch still sees the permanent registers.
%
% Reuses the classic-programs driver (tests/elixir_e2e/run_classic.exs):
% `elixir run_classic.exs <ModuleCamel> <pred/arity> <args...>` prints
% "true"/"false". Gated on `elixir` being on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_elixir_target').

:- dynamic user:eite/2.
:- dynamic user:eneg/1.
:- dynamic user:eseqite/3.
:- dynamic user:enestite/2.

user:eite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:eneg(X)          :- \+ X > 0.
user:eseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:enestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

elixir_available :-
    catch(( process_create(path(elixir), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_elixir_lowered_ite, [condition(elixir_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_elixir_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Compile each predicate to WAM and write the lowered Elixir project.
    Preds = [eite/2, eneg/1, eseqite/3, enestite/2],
    findall(N/A-W,
            ( member(N/A, Preds),
              wam_target:compile_predicate_to_wam(N/A, [], W) ),
            Pairs),
    write_wam_elixir_project(Pairs,
        [module_name('wam_elixir_ite'), emit_mode(lowered),
         source_module(user)], Dir),
    % Sanity: the predicates must actually be lowered (their per-pred
    % modules emitted), else the test would pass vacuously.
    forall(member(F, ['eite.ex', 'eneg.ex', 'eseqite.ex', 'enestite.ex']),
           ( atomic_list_concat([Dir, '/lib/', F], P),
             assertion(exists_file(P)) )),
    % 2. Copy the shared driver into the project root.
    classic_driver_path(DriverSrc),
    atomic_list_concat([Dir, '/run_classic.exs'], DriverDst),
    copy_file(DriverSrc, DriverDst),
    % 3. Run each case and assert the outcome.
    Cases = [ "true"-(eite/2)-['5', pos],
              "false"-(eite/2)-['5', nonpos],
              "true"-(eite/2)-['-1', nonpos],
              "false"-(eite/2)-['-1', pos],
              "false"-(eneg/1)-['5'],
              "true"-(eneg/1)-['-1'],
              "true"-(eneg/1)-['0'],
              "true"-(eseqite/3)-['10', pos, big],
              "false"-(eseqite/3)-['10', pos, small],
              "true"-(eseqite/3)-['3', pos, small],
              "true"-(eseqite/3)-['-1', nonpos, small],
              "true"-(enestite/2)-['20', big],
              "true"-(enestite/2)-['5', small],
              "true"-(enestite/2)-['-1', neg],
              "false"-(enestite/2)-['20', small] ],
    forall(member(Want-(Pred/Arity)-Args, Cases),
           ( format(atom(PredKey), '~w/~w', [Pred, Arity]),
             run_elixir_ite(Dir, PredKey, Args, Got),
             ( Got == Want
             -> true
             ;  format(user_error,
                  "~n[elixir ite] ~w(~w): got ~w want ~w~n",
                  [PredKey, Args, Got, Want]),
                throw(elixir_ite_mismatch(PredKey, Args, Want, Got))
             ) )),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_elixir_lowered_ite).

%% Resolved at consult time: reuse the classic-programs e2e driver.
:- ( prolog_load_context(directory, Dir),
     directory_file_path(Dir, 'elixir_e2e/run_classic.exs', P),
     assertz(ite_driver_path_fact(P))
   ; true ).
classic_driver_path(P) :- ite_driver_path_fact(P).

%% run_elixir_ite(+ProjectDir, +PredKey, +Args, -Output)
%  Invoke the driver; Output is the last non-empty stdout line ("true" /
%  "false"). The compiler emits unused-variable warnings to stderr on the
%  per-invocation recompile, which we discard.
run_elixir_ite(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, AbsDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>(atom_string(A, S)), Args, ArgStrs),
    append(['run_classic.exs', "WamElixirIte", PredStr], ArgStrs, ProcArgs),
    process_create(path(elixir), ProcArgs,
                   [cwd(AbsDir), stdout(pipe(Out)), stderr(null),
                    process(Pid)]),
    read_string(Out, _, OutStr0), close(Out),
    process_wait(Pid, _),
    split_string(OutStr0, "\n", " \t\r", Lines0),
    exclude(==(""), Lines0, Lines),
    ( last(Lines, Last) -> Output = Last ; Output = "" ).
