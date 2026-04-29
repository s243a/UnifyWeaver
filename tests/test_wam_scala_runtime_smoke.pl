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
:- dynamic user:wam_cut_first/1.
:- dynamic user:wam_cut_caller/1.
:- dynamic user:wam_list_match/1.
:- dynamic user:wam_use_list/1.
:- dynamic user:wam_make_list/1.
:- dynamic user:wam_foreign_yes/1.
:- dynamic user:wam_foreign_yes_caller/1.
:- dynamic user:wam_foreign_pair/2.
:- dynamic user:wam_foreign_pair_query/1.
:- dynamic user:wam_foreign_multi/2.
:- dynamic user:wam_foreign_multi_query/1.
:- dynamic user:wam_inc/2.
:- dynamic user:wam_double/2.
:- dynamic user:wam_combo/3.
:- dynamic user:wam_neg/2.
:- dynamic user:wam_lt/2.
:- dynamic user:wam_gt/2.
:- dynamic user:wam_geq/2.
:- dynamic user:wam_leq/2.
:- dynamic user:wam_eq_arith/2.
:- dynamic user:wam_neq_arith/2.
:- dynamic user:wam_cut_obs/1.
:- dynamic user:wam_always_fail/1.
:- dynamic user:wam_dummy_a/1.
:- dynamic user:wam_neg_dummy/1.
:- dynamic user:wam_float_mul/2.
:- dynamic user:wam_float_div/2.
:- dynamic user:wam_float_lit/1.
:- dynamic user:wam_float_cmp_lt/2.
:- dynamic user:wam_int_div_true/2.
:- dynamic user:wam_pair_inline/2.
:- dynamic user:wam_pair_sidecar/2.

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
% Hard cut: first clause commits on a, second clause handles b.
user:wam_cut_first(a) :- !.
user:wam_cut_first(b).
% Cut inside an inner predicate, then call from an outer caller, to
% exercise the cutBar interaction across allocate/deallocate frames.
user:wam_cut_caller(X) :- user:wam_cut_first(X).
% List match exercises get_list / unify_constant / unify_variable
% and the get_structure [|]/2 nested case.
user:wam_list_match([a, b]).
user:wam_use_list(L)   :- user:wam_list_match(L).
% List build exercises put_list / set_constant / set_variable and
% put_structure [|]/2.
user:wam_make_list(X)  :- X = [a, b].

% Foreign predicate stubs. When `foreign_predicates([P/A,...])` is
% passed to the project writer, the WAM body is replaced with a single
% CallForeign instruction; these Prolog bodies exist only so the
% predicates are declared.
user:wam_foreign_yes(_).
user:wam_foreign_yes_caller(X)  :- user:wam_foreign_yes(X).
user:wam_foreign_pair(_, _).
user:wam_foreign_pair_query(Y)  :- user:wam_foreign_pair(a, Y).
user:wam_foreign_multi(_, _).
user:wam_foreign_multi_query(Y) :- user:wam_foreign_multi(a, Y).
% wam_cut_obs: Y is fresh in the body; foreign yields Y=a then Y=b on
% backtrack. The cut after the foreign call must drop the foreign CP so
% that a later unify failure cannot redo the foreign and try the second
% solution. With correct cut: wam_cut_obs(b) → false. Without: would be true.
user:wam_cut_obs(X) :- user:wam_foreign_multi(a, Y), !, Y = X.

% Arithmetic: WAM emits builtin_call is/2, =:=/2, </2, >/2, =</2, >=/2,
% =\=/2 and builds expression terms via put_structure +/-/*/'/' .
user:wam_inc(N, M)        :- M is N + 1.
user:wam_double(N, M)     :- M is N * 2.
user:wam_combo(A, B, R)   :- R is A + B * 2.
user:wam_neg(N, M)        :- M is -N.
user:wam_lt(X, Y)         :- X < Y.
user:wam_gt(X, Y)         :- X > Y.
user:wam_geq(X, Y)        :- X >= Y.
user:wam_leq(X, Y)        :- X =< Y.
user:wam_eq_arith(X, Y)   :- X =:= Y.
user:wam_neq_arith(X, Y)  :- X =\= Y.

% --- fail/0 and \+/1 ---
user:wam_always_fail(_X) :- fail.
user:wam_dummy_a(a).
% Negation-as-failure: succeeds only when the inner goal fails.
user:wam_neg_dummy(X) :- \+ user:wam_dummy_a(X).

% --- Float arithmetic ---
user:wam_float_mul(X, Y)    :- Y is X * 2.5.
user:wam_float_div(X, Y)    :- Y is X / 2.0.
user:wam_float_lit(Y)       :- Y is 3.14.
user:wam_float_cmp_lt(X, Y) :- X < Y.
% True division of two integers: 5/2 -> 2.5, not 2 (per SWI semantics).
user:wam_int_div_true(X, Y) :- Y is X / 2.

% --- S7 fact backend seam ---
% Same relation under two backends. The inline form has WAM-compiled
% facts; the sidecar form passes the same tuples via scala_fact_sources
% so the codegen synthesises a ForeignHandler that enumerates them.
% Both forms must give identical answers for every query.
user:wam_pair_inline(a, b).
user:wam_pair_inline(b, c).
user:wam_pair_inline(c, d).
user:wam_pair_sidecar(_, _).

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

% --- S3+S4 lockdown coverage ---

test(hard_cut) :-
    with_scala_project(
        [user:wam_cut_first/1, user:wam_cut_caller/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_cut_first/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_first/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_cut_first/1', 'c', "false"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_cut_caller/1', 'c', "false")
        )).

test(list_match) :-
    with_scala_project(
        [user:wam_list_match/1, user:wam_use_list/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_use_list/1',   '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_use_list/1',   '[a,c]', "false"),
            verify_scala(TmpDir, 'wam_list_match/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_list_match/1', '[a,b,c]', "false")
        )).

test(list_build) :-
    with_scala_project(
        [user:wam_make_list/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_make_list/1', '[a,b]', "true"),
            verify_scala(TmpDir, 'wam_make_list/1', '[a,c]', "false")
        )).

% --- Phase S5: foreign predicate integration ---

test(foreign_boolean) :-
    foreign_yes_handler(YesHandler),
    with_scala_project(
        [user:wam_foreign_yes/1, user:wam_foreign_yes_caller/1],
        [ foreign_predicates([wam_foreign_yes/1]),
          intern_atoms([a, b]),
          scala_foreign_handlers([handler(wam_foreign_yes/1, YesHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_yes_caller/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_foreign_yes_caller/1', 'b', "false")
        )).

test(foreign_binding) :-
    foreign_pair_handler(PairHandler),
    with_scala_project(
        [user:wam_foreign_pair/2, user:wam_foreign_pair_query/1],
        [ foreign_predicates([wam_foreign_pair/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_pair/2, PairHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_pair_query/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_foreign_pair_query/1', 'c', "false")
        )).

test(foreign_multi) :-
    foreign_multi_handler(MultiHandler),
    with_scala_project(
        [user:wam_foreign_multi/2, user:wam_foreign_multi_query/1],
        [ foreign_predicates([wam_foreign_multi/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_multi/2, MultiHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_foreign_multi_query/1', 'c', "false")
        )).

% S6: cut after a multi-solution foreign must drop the foreign choice
% point so a later unify failure cannot redo the foreign.
% Discriminating case: wam_cut_obs(b) with cut → false; without cut → true.
test(cut_after_foreign_multi) :-
    foreign_multi_handler(MultiHandler),
    with_scala_project(
        [user:wam_foreign_multi/2, user:wam_cut_obs/1],
        [ foreign_predicates([wam_foreign_multi/2]),
          intern_atoms([a, b, c]),
          scala_foreign_handlers([handler(wam_foreign_multi/2, MultiHandler)]) ],
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_cut_obs/1', 'a', "true"),
            verify_scala(TmpDir, 'wam_cut_obs/1', 'b', "false"),
            verify_scala(TmpDir, 'wam_cut_obs/1', 'c', "false")
        )).

% --- Arithmetic builtins ---

test(arith_is) :-
    with_scala_project(
        [user:wam_inc/2, user:wam_double/2, user:wam_combo/3, user:wam_neg/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_inc/2',    ['5', '6'],     "true"),
            verify_scala_args(TmpDir, 'wam_inc/2',    ['5', '7'],     "false"),
            verify_scala_args(TmpDir, 'wam_double/2', ['7', '14'],    "true"),
            verify_scala_args(TmpDir, 'wam_double/2', ['7', '15'],    "false"),
            verify_scala_args(TmpDir, 'wam_combo/3',  ['3','4','11'], "true"),
            verify_scala_args(TmpDir, 'wam_combo/3',  ['3','4','12'], "false"),
            verify_scala_args(TmpDir, 'wam_neg/2',    ['5', '-5'],    "true"),
            verify_scala_args(TmpDir, 'wam_neg/2',    ['5', '5'],     "false")
        )).

test(arith_compare) :-
    with_scala_project(
        [user:wam_lt/2, user:wam_gt/2, user:wam_geq/2, user:wam_leq/2,
         user:wam_eq_arith/2, user:wam_neq_arith/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_lt/2',        ['1', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_lt/2',        ['2', '1'], "false"),
            verify_scala_args(TmpDir, 'wam_lt/2',        ['2', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_gt/2',        ['2', '1'], "true"),
            verify_scala_args(TmpDir, 'wam_gt/2',        ['1', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_geq/2',       ['2', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_geq/2',       ['1', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_leq/2',       ['2', '2'], "true"),
            verify_scala_args(TmpDir, 'wam_leq/2',       ['3', '2'], "false"),
            verify_scala_args(TmpDir, 'wam_eq_arith/2',  ['3', '3'], "true"),
            verify_scala_args(TmpDir, 'wam_eq_arith/2',  ['3', '4'], "false"),
            verify_scala_args(TmpDir, 'wam_neq_arith/2', ['3', '4'], "true"),
            verify_scala_args(TmpDir, 'wam_neq_arith/2', ['3', '3'], "false")
        )).

% --- fail/0 builtin ---
test(builtin_fail) :-
    with_scala_project(
        [user:wam_always_fail/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_always_fail/1', 'a', "false"),
            verify_scala(TmpDir, 'wam_always_fail/1', 'b', "false")
        )).

% --- \+/1 negation-as-failure ---
% wam_neg_dummy(a) → false (wam_dummy_a(a) succeeds, so \+ fails)
% wam_neg_dummy(b) → true  (wam_dummy_a(b) fails,    so \+ succeeds)
test(builtin_negation) :-
    with_scala_project(
        [user:wam_dummy_a/1, user:wam_neg_dummy/1],
        _Opts,
        TmpDir,
        (
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'a', "false"),
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'b', "true"),
            verify_scala(TmpDir, 'wam_neg_dummy/1', 'c', "true")
        )).

% --- Float arithmetic ---
test(arith_float) :-
    with_scala_project(
        [user:wam_float_mul/2, user:wam_float_div/2,
         user:wam_float_lit/1, user:wam_float_cmp_lt/2,
         user:wam_int_div_true/2],
        _Opts,
        TmpDir,
        (
            verify_scala_args(TmpDir, 'wam_float_mul/2',    ['4', '10.0'],   "true"),
            verify_scala_args(TmpDir, 'wam_float_mul/2',    ['4', '11.0'],   "false"),
            verify_scala_args(TmpDir, 'wam_float_div/2',    ['9', '4.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_float_div/2',    ['9', '4.0'],    "false"),
            verify_scala_args(TmpDir, 'wam_float_lit/1',    ['3.14'],        "true"),
            verify_scala_args(TmpDir, 'wam_float_lit/1',    ['3.15'],        "false"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['1.5', '2.5'],  "true"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['2.5', '1.5'],  "false"),
            % Mixed Int/Float comparison should promote.
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['1', '2.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_float_cmp_lt/2', ['2.5', '3'],    "true"),
            % True division of integers yields a Float.
            verify_scala_args(TmpDir, 'wam_int_div_true/2', ['5', '2.5'],    "true"),
            verify_scala_args(TmpDir, 'wam_int_div_true/2', ['5', '2'],      "false")
        )).

% --- S7 fact backend seam: parity inline-vs-sidecar ---
% Both versions answer the same set of queries identically.
test(fact_source_inline_vs_sidecar) :-
    InlineQueries =
        [['a','b']-true, ['a','c']-false, ['b','c']-true,
         ['c','d']-true, ['x','y']-false],
    % Inline backend: WAM-compiled facts.
    with_scala_project(
        [user:wam_pair_inline/2],
        _OptsInline,
        TmpDir1,
        forall(member(Args-Expected, InlineQueries),
               ( (Expected == true -> S = "true" ; S = "false"),
                 verify_scala_args(TmpDir1, 'wam_pair_inline/2', Args, S)
               ))),
    % Sidecar backend: same tuples, declared via scala_fact_sources.
    with_scala_project(
        [user:wam_pair_sidecar/2],
        [ scala_fact_sources(
              [source(wam_pair_sidecar/2, [[a,b], [b,c], [c,d]])]) ],
        TmpDir2,
        forall(member(Args-Expected, InlineQueries),
               ( (Expected == true -> S = "true" ; S = "false"),
                 verify_scala_args(TmpDir2, 'wam_pair_sidecar/2', Args, S)
               ))).

:- end_tests(wam_scala_runtime_smoke).

% ============================================================
% Test fixture helpers
% ============================================================

%% with_scala_project(+Preds, +Opts, -TmpDir, :Goal)
%  Generates a Scala project, compiles it, runs Goal (which may call
%  verify_scala/4), then cleans up. TmpDir is bound for Goal's use.
with_scala_project(Preds, ExtraOpts0, TmpDir, Goal) :-
    (   var(ExtraOpts0) -> ExtraOpts = [] ; ExtraOpts = ExtraOpts0 ),
    unique_scala_tmp_dir('tmp_scala_smoke', TmpDir),
    BaseOpts = [ package('generated.wam_scala_smoke.core'),
                 runtime_package('generated.wam_scala_smoke.core'),
                 module_name('wam-scala-smoke') ],
    append(ExtraOpts, BaseOpts, AllOpts),
    write_wam_scala_project(Preds, AllOpts, TmpDir),
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
%  Single-arg form for predicates of arity 1.
verify_scala(ProjectDir, PredKey, Arg, Expected) :-
    verify_scala_args(ProjectDir, PredKey, [Arg], Expected).

%% verify_scala_args(+ProjectDir, +PredKey, +Args, +Expected)
%  Multi-arg form: Args is a list of atoms/strings to pass on the CLI.
verify_scala_args(ProjectDir, PredKey, Args, Expected) :-
    run_scala_predicate_args(ProjectDir, PredKey, Args, Actual),
    (   Actual == Expected
    ->  true
    ;   throw(error(assertion_error(PredKey, Args, Expected, Actual), _))
    ).

run_scala_predicate_args(ProjectDir, PredKey, Args, Output) :-
    absolute_file_name(ProjectDir, AbsProjectDir),
    directory_file_path(AbsProjectDir, 'classes', ClassDir),
    atom_string(PredKey, PredStr),
    maplist([A, S]>>atom_string(A, S), Args, ArgStrs),
    append(['-classpath', ClassDir,
            'generated.wam_scala_smoke.core.GeneratedProgram',
            PredStr], ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
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
    ;   throw(error(scala_run_failed(ExitCode, PredKey, Args, ErrStr), _))
    ).

unique_scala_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

% ============================================================
% Foreign handler bodies (Scala source)
% ============================================================
% Each handler is a Scala expression of type `ForeignHandler` that
% gets injected into the generated `foreignHandlers` Map literal.
% Handlers run inside the GeneratedProgram object so they can
% reference `internTable` directly.

foreign_yes_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        args(0) match {\n          case Atom(id) if id == aId => ForeignSucceed\n          case _                     => ForeignFail\n        }\n      }\n    }".

foreign_pair_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        val bId = internTable.stringToId.getOrElse(\"b\", -1)\n        args(0) match {\n          case Atom(id) if id == aId => ForeignBindings(Map(2 -> Atom(bId)))\n          case _                     => ForeignFail\n        }\n      }\n    }".

foreign_multi_handler(Code) :-
    Code = "new ForeignHandler {\n      def apply(args: Array[WamTerm]): ForeignResult = {\n        val aId = internTable.stringToId.getOrElse(\"a\", -1)\n        val bId = internTable.stringToId.getOrElse(\"b\", -1)\n        args(0) match {\n          case Atom(id) if id == aId =>\n            ForeignMulti(Seq(\n              Map(2 -> Atom(aId)),\n              Map(2 -> Atom(bId))\n            ))\n          case _ => ForeignFail\n        }\n      }\n    }".
