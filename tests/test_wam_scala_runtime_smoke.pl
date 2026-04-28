:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_runtime_smoke.pl
%
% Runtime smoke tests for the WAM Scala hybrid target.
% Uses the scalac compiler to compile the generated project and
% the scala runner to execute it, verifying correctness of the
% stepping interpreter.
%
% Tests are gated on scalac availability to match the Clojure target
% pattern. Set SCALA_SMOKE_TESTS=1 or install scalac to enable.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').

% ============================================================
% Test predicates
% ============================================================

:- dynamic user:wam_fact/1.
:- dynamic user:wam_execute_caller/1.
:- dynamic user:wam_call_caller/1.
:- dynamic user:wam_choice_fact/1.
:- dynamic user:wam_choice_caller/1.
:- dynamic user:wam_choice_or_z/1.
:- dynamic user:wam_bind_then_fact/1.
:- dynamic user:wam_bind_after_call/1.
:- dynamic user:wam_bind_before_execute/1.
:- dynamic user:wam_if_then_else/1.
:- dynamic user:wam_struct_fact/1.
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_make_struct/1.
:- dynamic user:wam_env1/1.
:- dynamic user:wam_env2/1.
:- dynamic user:wam_trail_choice/1.

user:wam_fact(a).
user:wam_execute_caller(X)    :- user:wam_fact(X).
user:wam_call_caller(X)       :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X)     :- user:wam_choice_fact(X).
user:wam_choice_or_z(X)       :- user:wam_choice_fact(X).
user:wam_choice_or_z(z).
user:wam_bind_then_fact(X)    :- Y = X, user:wam_fact(Y).
user:wam_bind_after_call(X)   :- user:wam_fact(X), X = a.
user:wam_bind_before_execute(X) :- X = a, user:wam_fact(X).
user:wam_if_then_else(X)      :- (X = a -> true ; X = b).
user:wam_struct_fact(f(a)).
user:wam_use_struct(X)        :- user:wam_struct_fact(X).
user:wam_make_struct(X)       :- X = f(a).
user:wam_env1(X)              :- Y = X, Z = a, Y = Z.
user:wam_env2(X)              :- user:wam_fact(X), Y = X, user:wam_fact(Y).
user:wam_trail_choice(X)      :- (Y = a ; Y = b), X = Y.

% ============================================================
% Condition: only run if scalac is available
% ============================================================

%% scala_available/0 — true if scalac is on PATH
scala_available :-
    (   getenv('SCALA_SMOKE_TESTS', "1") -> true
    ;   catch(
            process_create(path(scalac), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            _,
            fail
        ),
        process_wait(Pid, exit(0))
    ).

% ============================================================
% Tests (gated on scala_available)
% ============================================================

:- begin_tests(wam_scala_runtime_smoke,
               [ condition(scala_available) ]).

test(fact_match) :-
    with_scala_project(
        [user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_fact/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_fact/1', 'b', "false")
        )).

test(execute_caller) :-
    with_scala_project(
        [user:wam_execute_caller/1, user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_execute_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_execute_caller/1', 'b', "false")
        )).

test(call_caller) :-
    with_scala_project(
        [user:wam_call_caller/1, user:wam_fact/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_call_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_call_caller/1', 'b', "false")
        )).

test(multi_clause_choice) :-
    with_scala_project(
        [user:wam_choice_fact/1, user:wam_choice_caller/1, user:wam_choice_or_z/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_choice_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'c', "true"),
            verify_scala(TmpDir, 'wam_choice_caller/1', 'd', "false"),
            verify_scala(TmpDir, 'wam_choice_or_z/1', 'z', "true")
        )).

test(binding_variants) :-
    with_scala_project(
        [user:wam_fact/1,
         user:wam_bind_then_fact/1,
         user:wam_bind_after_call/1,
         user:wam_bind_before_execute/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_bind_then_fact/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_then_fact/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_bind_after_call/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_after_call/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_bind_before_execute/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_bind_before_execute/1', 'b', "false")
        )).

test(if_then_else) :-
    with_scala_project(
        [user:wam_if_then_else/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_if_then_else/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_if_then_else/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_if_then_else/1', 'c', "false")
        )).

test(structure_matching) :-
    with_scala_project(
        [user:wam_struct_fact/1, user:wam_use_struct/1, user:wam_make_struct/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_use_struct/1', 'f(a)', "true"),
            verify_scala(TmpDir, 'wam_use_struct/1', 'f(b)', "false"),
            verify_scala(TmpDir, 'wam_make_struct/1', 'f(a)', "true")
        )).

test(environment_frames) :-
    with_scala_project(
        [user:wam_fact/1, user:wam_env1/1, user:wam_env2/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_env1/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_env1/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_env2/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_env2/1', 'b', "false")
        )).

test(trail_and_backtrack) :-
    with_scala_project(
        [user:wam_trail_choice/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_trail_choice/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_trail_choice/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_trail_choice/1', 'c', "false")
        )).

:- end_tests(wam_scala_runtime_smoke).

% ============================================================
% Test fixture helpers
% ============================================================

%% with_scala_project(+Preds, +Opts, -TmpDir, :Goal)
%  Generates a Scala project, compiles it, runs Goal (which may call
%  verify_scala/4), then cleans up. TmpDir is bound for Goal's use.
with_scala_project(Preds, _Opts, TmpDir, Goal) :-
    unique_scala_tmp_dir('tmp_scala_smoke', TmpDir),
    write_wam_scala_project(
        Preds,
        [ package('generated.wam_scala_smoke.core'),
          runtime_package('generated.wam_scala_smoke.core'),
          module_name('wam-scala-smoke')
        ],
        TmpDir),
    compile_scala_project(TmpDir),
    setup_call_cleanup(
        true,
        call(Goal),
        delete_directory_and_contents(TmpDir)).

%% compile_scala_project(+ProjectDir) is det.
%  Compiles all generated .scala files with scalac into ProjectDir/classes/.
compile_scala_project(ProjectDir) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    find_scala_sources(AbsProjectDir, Sources),
    Sources \= [],
    process_create(path(scalac),
                   ['-d', ClassDir | Sources],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, _OutStr),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_compile_failed(ExitCode, ErrStr), _))
    ).

%% find_scala_sources(+AbsProjectDir, -Sources)
%  Returns list of absolute paths of all .scala files under ProjectDir/src/.
find_scala_sources(AbsProjectDir, Sources) :-
    directory_file_path(AbsProjectDir, 'src', SrcDir),
    findall(F,
        ( directory_member(SrcDir, RelF,
              [extensions([scala]), recursive(true)]),
          directory_file_path(SrcDir, RelF, F)
        ),
        Sources).

%% verify_scala(+ProjectDir, +PredKey, +Arg, +Expected)
%  Runs the predicate via `scala` and checks stdout matches Expected.
verify_scala(ProjectDir, PredKey, Arg, Expected) :-
    run_scala_predicate(ProjectDir, PredKey, Arg, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Arg, Expected, Actual), _))
    ).

%% run_scala_predicate(+ProjectDir, +PredKey, +Arg, -Output)
run_scala_predicate(ProjectDir, PredKey, Arg, Output) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    atom_string(Arg, ArgStr),
    atom_string(PredKey, PredStr),
    process_create(path(scala),
                   ['-classpath', ClassDir,
                    'generated.wam_scala_smoke.core.GeneratedProgram',
                    PredStr, ArgStr],
                   [cwd(AbsProjectDir),
                    stdout(pipe(Out)), stderr(pipe(Err)),
                    process(Pid)]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    process_wait(Pid, exit(ExitCode)),
    normalize_space(string(Output), OutStr0),
    (   ExitCode =:= 0
    ->  true
    ;   throw(error(scala_run_failed(ExitCode, PredKey, Arg, ErrStr), _))
    ).

unique_scala_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).
