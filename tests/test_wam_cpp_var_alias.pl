:- encoding(utf8).
% Var-var aliasing regression tests for the WAM C++ target (M143).
%
% PR #2976's cross-target sweep found unify_cells copied one unbound
% marker over the other instead of aliasing the cells, so `X = Y`
% created no link: `X = Y, X = 1, Y = 2` SUCCEEDED, and
% `X = Y, X = 42, Y =:= 42` failed (in BOTH interpreter and lowered
% modes — the bug was in the shared runtime). unify_cells now records
% alias pairs; bind_cell propagates a later real value across them,
% and alias_pop trail entries dissolve pairs on backtrack.
%
% Gated on g++; generates a default-mode project with emit_main(true)
% and queries via the generated CLI, mirroring
% tests/test_wam_cpp_lowered_ite_exec.pl.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_cpp_target').

:- dynamic user:cppal_chain/1.
:- dynamic user:cppal_conflict/1.
:- dynamic user:cppal_chain3/1.
:- dynamic user:cppal_backtrack/1.

% Binding through a var-var alias must propagate to both.
user:cppal_chain(R)    :- X = Y, X = 42, ( Y =:= 42 -> R is 1 ; R is 0 ).
% Conflicting bindings through an alias must FAIL (the smoking gun:
% this succeeded before the fix).
user:cppal_conflict(R) :- X = Y, X = 1, Y = 2, R is 1.
% Three-deep chain.
user:cppal_chain3(R)   :- X = Y, Y = Z, X = 42, ( Z =:= 42 -> R is 1 ; R is 0 ).
% Alias must dissolve on backtrack out of the aliasing goal (this
% SEGFAULTED with the first-cut fix until every inline trail-unwind
% loop was routed through alias-aware unwind_trail_to).
user:cppal_backtrack(R) :-
    ( X = Y, X = 1, Y = 2 -> R is 0
    ; Y = 7, X = 5, ( Y =:= 7, X =:= 5 -> R is 1 ; R is 0 )
    ).

gpp_available :-
    catch(( process_create(path('g++'), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_cpp_var_alias, [condition(gpp_available)]).

test(var_alias_exec_parity) :-
    Dir = 'output/test_wam_cpp_var_alias',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_cpp_project(
        [user:cppal_chain/1, user:cppal_conflict/1,
         user:cppal_chain3/1, user:cppal_backtrack/1],
        [module_name(cppalias), emit_main(true)], Dir),
    atomic_list_concat([Dir, '/cpp'], CppDir),
    atomic_list_concat([CppDir, '/alias_bin'], ExePath),
    format(atom(BuildCmd),
        'g++ -std=c++17 -O0 -o ~w ~w/*.cpp 2>&1', [ExePath, CppDir]),
    shell_ok(BuildCmd),
    % (Pred, Arg, ExpectTrue): R=1 / R=0 query pairs read back R's value.
    Cases = [ 'cppal_chain/1'-1-true,  'cppal_chain/1'-0-false,
              'cppal_conflict/1'-1-false,
              'cppal_chain3/1'-1-true, 'cppal_chain3/1'-0-false,
              'cppal_backtrack/1'-1-true ],
    forall(member(Pred-Arg-Expect, Cases),
           run_case(ExePath, Pred, Arg, Expect)).

:- end_tests(wam_cpp_var_alias).

shell_ok(Cmd) :-
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true
    ; format(user_error, "~n[cpp var-alias build output]~n~w~n", [OutStr]),
      throw(cpp_var_alias_build_failed(Status))
    ).

run_case(ExePath, Pred, Arg, Expect) :-
    format(atom(Cmd), '~w "~w" ~w', [ExePath, Pred, Arg]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, _Status),
    ( Expect == true,  sub_string(OutStr, _, _, _, "true")  -> true
    ; Expect == false, sub_string(OutStr, _, _, _, "false") -> true
    ; format(user_error, "~n[cpp var-alias] ~w(~w): expected ~w, got: ~w~n",
             [Pred, Arg, Expect, OutStr]),
      throw(cpp_var_alias_case_failed(Pred, Arg))
    ).
