:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_scala_lowered_t5.pl
%
% End-to-end execution test for the Scala T5 lowering — "multi-clause as an
% if-then-else chain" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% A predicate whose clauses discriminate on a DISTINCT first-argument
% constant now lowers (emit_mode(functions)) to a bound-checked first-arg
% dispatch over ALL clauses, instead of the multi_clause_1 shape (clause 1
% inline, clauses 2+ via the interpreter fallback). The payoff is that
% non-first clauses become fast-path too when the first argument is bound;
% an unbound first argument still defers to the interpreter (enumeration).
%
% Soundness rests on the discriminators being distinct (so at most one
% clause matches a bound first arg — deterministic), so the first-solution
% contract of the lowered entry holds for both boolean and enumeration
% queries. Predicates with non-distinct or non-constant first-arg heads
% decline T5 and keep their previous lowering.
%
% Pins (the GeneratedProgram CLI passes ground args; first arg is bound, so
% these exercise the T5 fast-path dispatch including the non-first clauses):
%   * color/1  — fact chain, atom discriminators;
%   * sz/2     — fact chain with a second head-match in each remainder;
%   * op/2     — RULE chain (each remainder runs an is/2 builtin).
%
% Gated on `scalac` + `scala` being on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_scala_lowered_emitter').

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

scala_available :-
    catch(( process_create(path(scalac), ['-version'],
                           [stdout(null), stderr(null), process(P1)]),
            process_wait(P1, _) ), _, fail),
    catch(( process_create(path(scala), ['-version'],
                           [stdout(null), stderr(null), process(P2)]),
            process_wait(P2, _) ), _, fail).

:- begin_tests(wam_scala_lowered_t5, [condition(scala_available)]).

% The three predicates must lower as T5 (clause_chain), not multi_clause_1.
test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_scala_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_scala_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    write_wam_scala_project(
        [user:color/1, user:sz/2, user:op/2],
        [package('generated.t5.core'), runtime_package('generated.t5.core'),
         module_name('t5'), emit_mode(functions)], Dir),
    % Sanity: the generated program must contain the T5 dispatch for each.
    directory_file_path(Dir, 'src/main/scala/generated/t5/core/GeneratedProgram.scala', ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_color_1", "lowered_sz_2", "lowered_op_2",
                      "T5 first-argument dispatch"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    t5_compile(Dir),
    % color/1 — atom fact chain (all three clauses + a miss).
    t5_check(Dir, 'color/1', [red],    "true"),
    t5_check(Dir, 'color/1', [green],  "true"),
    t5_check(Dir, 'color/1', [blue],   "true"),
    t5_check(Dir, 'color/1', [yellow], "false"),
    % sz/2 — second head match in the remainder.
    t5_check(Dir, 'sz/2', [small, 1],  "true"),
    t5_check(Dir, 'sz/2', [medium, 2], "true"),
    t5_check(Dir, 'sz/2', [large, 3],  "true"),
    t5_check(Dir, 'sz/2', [small, 2],  "false"),
    t5_check(Dir, 'sz/2', [big, 1],    "false"),
    % op/2 — rule chain, each remainder runs is/2.
    t5_check(Dir, 'op/2', [add, 2],  "true"),
    t5_check(Dir, 'op/2', [mul, 6],  "true"),
    t5_check(Dir, 'op/2', [neg, -1], "true"),
    t5_check(Dir, 'op/2', [add, 3],  "false"),
    t5_check(Dir, 'op/2', [div, 1],  "false"),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_scala_lowered_t5).

% --- compile + run harness (mirrors test_wam_scala_classic_programs) -------

t5_compile(ProjectDir) :-
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

t5_check(ProjectDir, PredKey, Args, Expected) :-
    absolute_file_name(ProjectDir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir, 'generated.t5.core.GeneratedProgram', PredStr],
           ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
                   [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, Out0), read_string(E, _, ErrStr), close(O), close(E),
    process_wait(Pid, exit(Code)),
    normalize_space(string(Actual), Out0),
    ( Code =:= 0 -> true ; throw(error(scala_run_failed(Code, PredKey, Args, ErrStr), _)) ),
    ( Actual == Expected
    -> true
    ;  throw(error(t5_assertion(PredKey, Args, Expected, Actual), _)) ).
