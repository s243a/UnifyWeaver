:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_lowered_emitter.pl
%
% Tests for the WAM Scala per-predicate lowered emitter
% (wam_scala_lowered_emitter.pl), the feature that brings the Scala
% hybrid WAM target to parity with the Haskell/Rust/C++/F#/Go/Clojure
% targets — all of which ship a `wam_*_lowered_emitter.pl`.
%
% Two suites:
%   wam_scala_lowered_structural  — always runs.  Exercises emit-mode
%       resolution, predicate partitioning, lowerability analysis, and
%       the generated source (lowered functions + loweredEntries map).
%   wam_scala_lowered_runtime     — gated on scalac/scala (or
%       SCALA_SMOKE_TESTS=1).  Generates the SAME predicates in both
%       interpreter and functions mode, compiles both, and asserts that
%       every query yields identical AND correct results — i.e. the
%       lowered fast path is behaviour-preserving.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/targets/wam_scala_lowered_emitter').

% ============================================================
% Sample predicates
% ============================================================

:- dynamic user:lp_fact/2.
:- dynamic user:lp_edge/2.
:- dynamic user:lp_append/3.
:- dynamic user:lp_member/2.
:- dynamic user:lp_len/2.

% Factorial — arithmetic recursion (is/2, >/2, recursive call).
user:lp_fact(0, 1).
user:lp_fact(N, F) :- N > 0, N1 is N - 1, lp_fact(N1, F1), F is N * F1.

% Facts — first-argument indexed, multi-clause, deterministic.
user:lp_edge(a, b).
user:lp_edge(b, c).
user:lp_edge(c, d).

% append/3 — list recursion (get_list / unify_* read+write modes).
user:lp_append([], L, L).
user:lp_append([H|T], L, [H|R]) :- lp_append(T, L, R).

% member/2 — NONdeterministic builtin in the body; must NOT be lowered
% as a deterministic fast path (it relies on the interpreter's
% choice points). Used to verify the partition keeps it interpreted.
user:lp_member(X, [X|_]).
user:lp_member(X, [_|T]) :- lp_member(X, T).

% length/2 via accumulator — deterministic in (+,?) mode.
user:lp_len([], 0).
user:lp_len([_|T], N) :- lp_len(T, N0), N is N0 + 1.

% ============================================================
% Structural suite (always runs)
% ============================================================

:- begin_tests(wam_scala_lowered_structural).

% --- emit-mode resolution ---------------------------------

test(emit_mode_default_interpreter) :-
    scala_resolve_emit_mode([], Mode),
    assertion(Mode == interpreter).

test(emit_mode_functions) :-
    scala_resolve_emit_mode([emit_mode(functions)], Mode),
    assertion(Mode == functions).

test(emit_mode_mixed) :-
    scala_resolve_emit_mode([emit_mode(mixed([lp_fact/2]))], Mode),
    assertion(Mode == mixed([lp_fact/2])).

test(emit_mode_invalid_throws, throws(_)) :-
    scala_resolve_emit_mode([emit_mode(nonsense)], _).

% --- lowerability analysis --------------------------------

test(fact_is_lowerable) :-
    wam_predicate_wamcode(user:lp_fact/2, Code),
    assertion(wam_scala_lowerable(user:lp_fact/2, Code, _)).

test(edge_is_lowerable) :-
    wam_predicate_wamcode(user:lp_edge/2, Code),
    assertion(wam_scala_lowerable(user:lp_edge/2, Code, _)).

test(append_is_lowerable) :-
    wam_predicate_wamcode(user:lp_append/3, Code),
    assertion(wam_scala_lowerable(user:lp_append/3, Code, _)).

% member/2's clause 1 is just a head match (no nondeterministic builtin),
% so it IS lowerable; the entry wrapper's interpreter fallback preserves
% the multi-solution semantics.  This documents that lowerability is a
% clause-1 property, not a whole-predicate determinism claim.
test(member_clause1_lowerable) :-
    wam_predicate_wamcode(user:lp_member/2, Code),
    assertion(wam_scala_lowerable(user:lp_member/2, Code, _)).

% --- partition --------------------------------------------

test(partition_interpreter_keeps_all) :-
    scala_partition_predicates(interpreter,
        [user:lp_fact/2, user:lp_edge/2], Interp, Lowered),
    assertion(Interp == [user:lp_fact/2, user:lp_edge/2]),
    assertion(Lowered == []).

test(partition_functions_lowers_deterministic) :-
    scala_partition_predicates(functions,
        [user:lp_fact/2, user:lp_edge/2, user:lp_append/3],
        _Interp, Lowered),
    assertion(memberchk(user:lp_fact/2, Lowered)),
    assertion(memberchk(user:lp_edge/2, Lowered)),
    assertion(memberchk(user:lp_append/3, Lowered)).

test(partition_mixed_only_listed) :-
    scala_partition_predicates(mixed([lp_fact/2]),
        [user:lp_fact/2, user:lp_edge/2], Interp, Lowered),
    assertion(memberchk(user:lp_fact/2, Lowered)),
    assertion(memberchk(user:lp_edge/2, Interp)),
    assertion(\+ memberchk(user:lp_edge/2, Lowered)).

% --- function-name generation -----------------------------

test(func_name_sanitized) :-
    scala_lowered_func_name(lp_fact/2, N1),
    assertion(N1 == 'lowered_lp_fact_2'),
    scala_lowered_func_name('weird$name'/3, N2),
    assertion(N2 == 'lowered_weird_name_3').

% --- generated source: functions mode ---------------------

test(functions_mode_emits_lowered_code) :-
    once((
        tmp_dir('tmp_scala_low_fn', Dir),
        write_wam_scala_project(
            [user:lp_fact/2, user:lp_edge/2],
            [ package('gen.low.core'), runtime_package('gen.low.core'),
              module_name('low'), emit_mode(functions) ],
            Dir),
        program_source(Dir, 'gen.low.core', Src),
        assertion(sub_string(Src, _, _, _, "def lowered_lp_fact_2")),
        assertion(sub_string(Src, _, _, _, "def lowered_lp_edge_2")),
        % loweredEntries holds real entry wrappers (the "-> ((prog:" form
        % is unique to lowered entries; the dispatch/labels maps use plain
        % "-> <Int>").
        assertion(sub_string(Src, _, _, _, "\"lp_fact/2\" -> ((prog: WamProgram")),
        assertion(sub_string(Src, _, _, _, "loweredEntries")),
        % The entry wrapper falls back to the interpreter on a miss.
        assertion(sub_string(Src, _, _, _, "WamRuntime.runPredicate(prog, startPc, args)")),
        delete_directory_and_contents(Dir)
    )).

% --- generated source: interpreter mode (default) ---------
% Must be behaviourally identical to the pre-lowering target: no lowered
% functions, and an EMPTY loweredEntries map so runEntry just delegates.

test(interpreter_mode_has_empty_lowered) :-
    once((
        tmp_dir('tmp_scala_low_it', Dir),
        write_wam_scala_project(
            [user:lp_fact/2, user:lp_edge/2],
            [ package('gen.it.core'), runtime_package('gen.it.core'),
              module_name('it') ],
            Dir),
        program_source(Dir, 'gen.it.core', Src),
        assertion(\+ sub_string(Src, _, _, _, "def lowered_lp_fact_2")),
        % loweredEntries present (always) but with no entry wrappers.
        assertion(sub_string(Src, _, _, _, "loweredEntries")),
        assertion(\+ sub_string(Src, _, _, _, "-> ((prog: WamProgram")),
        delete_directory_and_contents(Dir)
    )).

:- end_tests(wam_scala_lowered_structural).

% ============================================================
% Runtime parity suite (gated on scalac/scala)
% ============================================================

:- begin_tests(wam_scala_lowered_runtime,
               [condition(scala_available)]).

% lp_edge/2 — facts.  Identical results across modes, all correct.
test(edge_parity,
     [setup(with_both_modes([user:lp_edge/2], 'gen.edge', Run)),
      cleanup(cleanup_run(Run))]) :-
    assert_same_and(Run, 'lp_edge/2', [a,b], "true"),
    assert_same_and(Run, 'lp_edge/2', [b,c], "true"),
    assert_same_and(Run, 'lp_edge/2', [c,d], "true"),
    assert_same_and(Run, 'lp_edge/2', [a,c], "false"),
    assert_same_and(Run, 'lp_edge/2', [x,y], "false").

% lp_fact/2 — arithmetic recursion (exercises is/2, >/2, recursive call,
% and the switch_on_constant_fallthrough first-arg index prefix).
test(fact_parity,
     [setup(with_both_modes([user:lp_fact/2], 'gen.fact', Run)),
      cleanup(cleanup_run(Run))]) :-
    assert_same_and(Run, 'lp_fact/2', ['0','1'],   "true"),
    assert_same_and(Run, 'lp_fact/2', ['5','120'], "true"),
    assert_same_and(Run, 'lp_fact/2', ['6','720'], "true"),
    assert_same_and(Run, 'lp_fact/2', ['5','121'], "false").

% lp_append/3 — list recursion (get_list, unify_* read+write).
test(append_parity,
     [setup(with_both_modes([user:lp_append/3], 'gen.app', Run)),
      cleanup(cleanup_run(Run))]) :-
    assert_same_and(Run, 'lp_append/3', ['[a,b]','[c]','[a,b,c]'], "true"),
    assert_same_and(Run, 'lp_append/3', ['[]','[c]','[c]'],        "true"),
    assert_same_and(Run, 'lp_append/3', ['[a]','[b]','[a,x]'],     "false").

% lp_member/2 — nondeterministic; the lowered fast path's clause-1 miss
% must fall back to the interpreter so later list elements still match.
test(member_parity,
     [setup(with_both_modes([user:lp_member/2], 'gen.mem', Run)),
      cleanup(cleanup_run(Run))]) :-
    assert_same_and(Run, 'lp_member/2', [a, '[a,b,c]'], "true"),
    assert_same_and(Run, 'lp_member/2', [c, '[a,b,c]'], "true"),
    assert_same_and(Run, 'lp_member/2', [z, '[a,b,c]'], "false").

:- end_tests(wam_scala_lowered_runtime).

% ============================================================
% Helpers
% ============================================================

scala_available :-
    (   getenv('SCALA_SMOKE_TESTS', "1") -> true
    ;   catch(
            process_create(path(scalac), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            _, fail),
        process_wait(Pid, exit(0))
    ).

wam_predicate_wamcode(user:Pred/Arity, Code) :- !,
    wam_target:compile_predicate_to_wam(Pred/Arity, [], Code).
wam_predicate_wamcode(PI, Code) :-
    wam_target:compile_predicate_to_wam(PI, [], Code).

tmp_dir(Prefix, Dir) :-
    get_time(T), Stamp is floor(T * 1000),
    format(atom(Dir), '~w_~w', [Prefix, Stamp]).

%% program_source(+Dir, +Package, -Src)
program_source(Dir, Package, Src) :-
    atom_string(Package, PkgStr),
    split_string(PkgStr, ".", "", Parts),
    atomic_list_concat(Parts, '/', PkgPath),
    format(atom(Rel), 'src/main/scala/~w/GeneratedProgram.scala', [PkgPath]),
    directory_file_path(Dir, Rel, Path),
    read_file_to_string(Path, Src, []).

%% with_both_modes(+Preds, +PkgBase, -Run)
%  Generates and compiles the predicates twice (interpreter + functions
%  mode), returning a closure descriptor Run = run(InterpDir, FnDir, Pkg)
%  for assert_same_and/4 to invoke. Cleanup is deferred to the caller via
%  the dynamic registry so test bodies stay readable; we clean eagerly
%  inside assert_same_and's owner instead — here we just build dirs.
with_both_modes(Preds, PkgBase, run(ItDir, FnDir, ItPkg, FnPkg)) :-
    format(atom(ItPkg), '~w.it.core', [PkgBase]),
    format(atom(FnPkg), '~w.fn.core', [PkgBase]),
    tmp_dir('tmp_scala_pit', ItDir),
    tmp_dir('tmp_scala_pfn', FnDir),
    write_wam_scala_project(Preds,
        [package(ItPkg), runtime_package(ItPkg), module_name('pit')], ItDir),
    write_wam_scala_project(Preds,
        [package(FnPkg), runtime_package(FnPkg), module_name('pfn'),
         emit_mode(functions)], FnDir),
    compile_project(ItDir),
    compile_project(FnDir).

%% assert_same_and(+Run, +PredKey, +Args, +Expected)
%  Runs PredKey(Args) in both the interpreter and functions builds,
%  asserting they agree with each other AND with Expected.
assert_same_and(run(ItDir, FnDir, ItPkg, FnPkg), PredKey, Args, Expected) :-
    run_query(ItDir, ItPkg, PredKey, Args, ItOut),
    run_query(FnDir, FnPkg, PredKey, Args, FnOut),
    (   ItOut == FnOut, ItOut == Expected
    ->  true
    ;   throw(error(parity_mismatch(PredKey, Args,
                expected(Expected), interp(ItOut), functions(FnOut)), _))
    ).

cleanup_run(run(ItDir, FnDir, _, _)) :-
    ignore(delete_directory_and_contents(ItDir)),
    ignore(delete_directory_and_contents(FnDir)).

compile_project(Dir) :-
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    findall(F,
        ( directory_member(AbsDir, RelF,
              [extensions([scala]), recursive(true)]),
          directory_file_path(AbsDir, RelF, F)
        ), Sources),
    Sources \= [],
    process_create(path(scalac), ['-d', ClassDir | Sources],
        [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, Err), close(O), close(E),
    process_wait(Pid, exit(Code)),
    (   Code =:= 0 -> true
    ;   throw(error(scala_compile_failed(Dir, Code, Err), _))
    ).

run_query(Dir, Package, PredKey, Args, Output) :-
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    format(atom(Main), '~w.GeneratedProgram', [Package]),
    atom_string(PredKey, PredStr),
    maplist([A,S]>>atom_string(A,S), Args, ArgStrs),
    append(['-classpath', ClassDir, Main, PredStr], ArgStrs, ProcArgs),
    process_create(path(scala), ProcArgs,
        [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, Out0), read_string(E, _, _), close(O), close(E),
    process_wait(Pid, exit(_)),
    normalize_space(string(Output), Out0).
