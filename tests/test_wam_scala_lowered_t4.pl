:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_scala_lowered_t4.pl
%
% End-to-end execution test for the Scala T4 lowering — "multi-clause, all
% clauses" (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% A multi-clause predicate whose clauses are all supported deterministic
% bodies but do NOT discriminate on a distinct first-argument constant (so T5
% declines) now lowers (emit_mode(functions)) to ALL clauses inline: each
% clause is a sibling Boolean closure, tried in order with a trail/register
% restore between attempts. The first clause that succeeds wins
% (first-solution / deterministic-prefix semantics, matching loCall), and the
% interpreter is NEVER entered for the predicate — the entry has no
% runPredicate fallback (unlike the multi_clause_1 / T3 shape, which lowers
% only clause 1 and defers clauses 2+ to the bytecode interpreter).
%
% Pins (the GeneratedProgram CLI passes ground args; the payoff is that the
% non-first clauses run natively):
%   * grade/2 — fact chain with a REPEATED first arg (alice in clauses 1 & 3),
%               so it is not a distinct-first-arg chain; grade(alice,c) needs
%               clause 3, grade(bob,b) needs clause 2;
%   * rel/2   — RULE chain with a VARIABLE first arg (=/2 in each body);
%               rel(q,two) needs clause 2.
%
% Gated on `scalac` + `scala` being on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_scala_lowered_emitter').

:- dynamic user:grade/2.
:- dynamic user:rel/2.

user:grade(alice, a).
user:grade(bob,   b).
user:grade(alice, c).

user:rel(X, one) :- X = p.
user:rel(X, two) :- X = q.

scala_available :-
    catch(( process_create(path(scalac), ['-version'],
                           [stdout(null), stderr(null), process(P1)]),
            process_wait(P1, _) ), _, fail),
    catch(( process_create(path(scala), ['-version'],
                           [stdout(null), stderr(null), process(P2)]),
            process_wait(P2, _) ), _, fail).

:- begin_tests(wam_scala_lowered_t4, [condition(scala_available)]).

% Both predicates must lower as T4 (multi_clause_n), not multi_clause_1.
test(gate_picks_multi_clause_n) :-
    forall(member(PI, [grade/2, rel/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_scala_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )).

test(t4_exec_parity) :-
    Dir = 'output/test_wam_scala_t4_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_scala_project(
        [user:grade/2, user:rel/2],
        [package('generated.t4.core'), runtime_package('generated.t4.core'),
         module_name('t4'), emit_mode(functions)], Dir),
    directory_file_path(Dir, 'src/main/scala/generated/t4/core/GeneratedProgram.scala', ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    % Sanity: both predicates lowered as T4 (all clauses inline).
    forall(member(F, ["lowered_grade_2", "lowered_rel_2", "T4 all-clauses inline"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    % The entry must be the direct form (no runPredicate fallback): this is
    % what makes "the interpreter is never entered for the predicate" hold.
    assertion(sub_string(ProgSrc, _, _, _,
        "=> lowered_grade_2(WamRuntime.newState(prog.dispatch(\"grade/2\"), args), prog))")),
    assertion(sub_string(ProgSrc, _, _, _,
        "=> lowered_rel_2(WamRuntime.newState(prog.dispatch(\"rel/2\"), args), prog))")),
    t4_compile(Dir),
    % grade/2 — repeated first arg; clauses 2 and 3 are the payoff.
    t4_check(Dir, 'grade/2', [alice, a], "true"),
    t4_check(Dir, 'grade/2', [bob,   b], "true"),
    t4_check(Dir, 'grade/2', [alice, c], "true"),
    t4_check(Dir, 'grade/2', [alice, b], "false"),
    t4_check(Dir, 'grade/2', [carol, a], "false"),
    t4_check(Dir, 'grade/2', [bob,   c], "false"),
    % rel/2 — variable first arg, =/2 body; clause 2 is the payoff.
    t4_check(Dir, 'rel/2', [p, one], "true"),
    t4_check(Dir, 'rel/2', [q, two], "true"),
    t4_check(Dir, 'rel/2', [p, two], "false"),
    t4_check(Dir, 'rel/2', [q, one], "false"),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_scala_lowered_t4).

% --- compile + run harness (mirrors test_wam_scala_lowered_t5) -------------

t4_compile(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    directory_file_path(AbsDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF, [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F) ),
        Sources),
    Sources \= [],
    process_create(path(scalac), ['-d', ClassDir | Sources],
                   [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, ErrStr), close(O), close(E),
    process_wait(Pid, exit(Code)),
    ( Code =:= 0 -> true ; throw(error(scala_compile_failed(Code, ErrStr), _)) ).

t4_check(ProjectDir, PredKey, Args, Expected) :-
    absolute_file_name(ProjectDir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir, 'generated.t4.core.GeneratedProgram', PredStr],
           ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, Out0), read_string(E, _, ErrStr), close(O), close(E),
    process_wait(Pid, exit(Code)),
    normalize_space(string(Actual), Out0),
    ( Code =:= 0 -> true ; throw(error(scala_run_failed(Code, PredKey, Args, ErrStr), _)) ),
    ( Actual == Expected
    -> true
    ;  throw(error(t4_assertion(PredKey, Args, Expected, Actual), _)) ).
