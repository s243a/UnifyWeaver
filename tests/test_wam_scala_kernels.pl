:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% test_wam_scala_kernels.pl
%
% Tests for hot-path graph-kernel native lowering in the WAM Scala target
% (kernel_dispatch(true)), bringing Scala to parity with the
% Rust/Haskell/Elixir/Go kernel route. A predicate matching a recognised
% graph-kernel shape is replaced by a synthesized Scala ForeignHandler
% that performs the traversal natively, bypassing the WAM step loop.
%
% Two suites:
%   wam_scala_kernels_structural — always runs. Detection fires; kernel
%       mode emits a CallForeign stub + native handler; the edge relation
%       stays WAM-compiled; interpreter mode is unaffected.
%   wam_scala_kernels_runtime    — gated on scalac/scala. Compiles the
%       same predicates with and without kernel dispatch and asserts the
%       transitive-closure results are identical AND correct.
%
% Currently covers transitive_closure2 (the first kernel kind ported to
% Scala); the remaining six follow the same pattern.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_scala_target').
:- use_module('../src/unifyweaver/core/recursive_kernel_detection',
              [detect_recursive_kernel/4]).

% ------------------------------------------------------------
% Sample graph: a -> b -> c -> d, a -> e -> f
% ------------------------------------------------------------

:- dynamic user:kedge/2.
:- dynamic user:ktc/2.

user:kedge(a, b).
user:kedge(b, c).
user:kedge(c, d).
user:kedge(a, e).
user:kedge(e, f).

user:ktc(X, Y) :- kedge(X, Y).
user:ktc(X, Y) :- kedge(X, Z), ktc(Z, Y).

% transitive_distance3: shortest-path distance over the same edges.
:- dynamic user:ktd/3.
user:ktd(X, Y, 1) :- kedge(X, Y).
user:ktd(X, Y, D) :- kedge(X, Z), ktd(Z, Y, D1), D is D1 + 1.

% transitive_parent_distance4: target + immediate predecessor + distance.
:- dynamic user:kpd/4.
user:kpd(X, Y, X, 1) :- kedge(X, Y).
user:kpd(X, Y, P, D) :- kedge(X, Z), kpd(Z, Y, P, D1), D is D1 + 1.

% transitive_step_parent_distance5: target + first hop + parent + distance.
:- dynamic user:ksp/5.
user:ksp(X, Y, Y, X, 1) :- kedge(X, Y).
user:ksp(X, Y, Z, P, D) :- kedge(X, Z), ksp(Z, Y, _, P, D1), D is D1 + 1.

% ============================================================
% Structural suite (always runs)
% ============================================================

:- begin_tests(wam_scala_kernels_structural).

test(detects_transitive_closure2) :-
    collect_clauses(ktc, 2, Clauses),
    detect_recursive_kernel(ktc, 2, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_closure2, ktc/2, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kedge/2), Cfg)).

test(detects_transitive_distance3) :-
    collect_clauses(ktd, 3, Clauses),
    detect_recursive_kernel(ktd, 3, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_distance3, ktd/3, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kedge/2), Cfg)).

test(detects_transitive_parent_distance4) :-
    collect_clauses(kpd, 4, Clauses),
    detect_recursive_kernel(kpd, 4, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_parent_distance4, kpd/4, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kedge/2), Cfg)).

test(detects_transitive_step_parent_distance5) :-
    collect_clauses(ksp, 5, Clauses),
    detect_recursive_kernel(ksp, 5, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(transitive_step_parent_distance5, ksp/5, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kedge/2), Cfg)).

% Kernel mode: ktc becomes a CallForeign stub backed by a native BFS
% handler; the edge relation stays WAM-compiled.
test(kernel_mode_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_kn', Dir),
        write_wam_scala_project(
            [user:ktc/2, user:kedge/2],
            [ package('kk.core'), runtime_package('kk.core'),
              module_name('kk'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'kk.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"ktc\", 2)")),
        assertion(sub_string(Src, _, _, _, "collectBinarySolutions(sharedProgram, \"kedge/2\")")),
        % edge predicate is still compiled (has a dispatch entry, not foreign).
        assertion(sub_string(Src, _, _, _, "\"kedge/2\" ->")),
        delete_directory_and_contents(Dir)
    )).

% transitive_distance3 kernel mode: ktd becomes a CallForeign stub backed
% by a native BFS-with-distance handler that binds register 3 (distance).
test(distance_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_knd', Dir),
        write_wam_scala_project(
            [user:ktd/3, user:kedge/2],
            [ package('kd.core'), runtime_package('kd.core'),
              module_name('kd'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'kd.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"ktd\", 3)")),
        assertion(sub_string(Src, _, _, _, "3 -> IntTerm")),
        assertion(sub_string(Src, _, _, _, "\"kedge/2\" ->")),
        delete_directory_and_contents(Dir)
    )).

% transitive_parent_distance4 kernel mode: kpd becomes a CallForeign stub
% backed by a native BFS handler binding target/parent/distance.
test(parent_distance_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_knp', Dir),
        write_wam_scala_project(
            [user:kpd/4, user:kedge/2],
            [ package('kp.core'), runtime_package('kp.core'),
              module_name('kp'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'kp.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"kpd\", 4)")),
        assertion(sub_string(Src, _, _, _, "4 -> IntTerm")),
        assertion(sub_string(Src, _, _, _, "\"kedge/2\" ->")),
        delete_directory_and_contents(Dir)
    )).

% transitive_step_parent_distance5 kernel mode: ksp becomes a CallForeign
% stub backed by a native BFS handler binding target/step/parent/distance.
test(step_parent_distance_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_kns', Dir),
        write_wam_scala_project(
            [user:ksp/5, user:kedge/2],
            [ package('ks.core'), runtime_package('ks.core'),
              module_name('ks'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'ks.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"ksp\", 5)")),
        assertion(sub_string(Src, _, _, _, "5 -> IntTerm")),
        assertion(sub_string(Src, _, _, _, "\"kedge/2\" ->")),
        delete_directory_and_contents(Dir)
    )).

% Without kernel_dispatch, behaviour is unchanged: ktc is WAM-compiled,
% no foreign handler synthesized.
test(default_mode_no_kernel) :-
    once((
        ktmp('tmp_scala_nokn', Dir),
        write_wam_scala_project(
            [user:ktc/2, user:kedge/2],
            [ package('nk.core'), runtime_package('nk.core'),
              module_name('nk') ],
            Dir),
        kprogram_source(Dir, 'nk.core', Src),
        assertion(\+ sub_string(Src, _, _, _, "CallForeign(\"ktc\", 2)")),
        assertion(\+ sub_string(Src, _, _, _, "collectBinarySolutions")),
        delete_directory_and_contents(Dir)
    )).

:- end_tests(wam_scala_kernels_structural).

% ============================================================
% Runtime parity suite (gated on scalac/scala)
% ============================================================

:- begin_tests(wam_scala_kernels_runtime,
               [condition(kscala_available)]).

test(transitive_closure_parity,
     [setup(kbuild_both([user:ktc/2, user:kedge/2], 'gen.ktc', Run)),
      cleanup(kcleanup(Run))]) :-
    % a -> b -> c -> d, a -> e -> f
    ksame(Run, 'ktc/2', [a,b], "true"),
    ksame(Run, 'ktc/2', [a,c], "true"),
    ksame(Run, 'ktc/2', [a,d], "true"),
    ksame(Run, 'ktc/2', [a,f], "true"),
    ksame(Run, 'ktc/2', [b,d], "true"),
    ksame(Run, 'ktc/2', [b,a], "false"),
    ksame(Run, 'ktc/2', [e,d], "false"),
    ksame(Run, 'ktc/2', [c,f], "false").

% transitive_distance3: tree graph so each reachable node has a single
% path length, hence kernel (shortest distance) and interpreter agree.
test(transitive_distance_parity,
     [setup(kbuild_both([user:ktd/3, user:kedge/2], 'gen.ktd', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'ktd/3', [a,b,'1'], "true"),
    ksame(Run, 'ktd/3', [a,c,'2'], "true"),
    ksame(Run, 'ktd/3', [a,d,'3'], "true"),
    ksame(Run, 'ktd/3', [a,e,'1'], "true"),
    ksame(Run, 'ktd/3', [a,f,'2'], "true"),
    ksame(Run, 'ktd/3', [b,d,'2'], "true"),
    ksame(Run, 'ktd/3', [a,d,'2'], "false"),
    ksame(Run, 'ktd/3', [a,c,'3'], "false").

% transitive_parent_distance4: (target, immediate-parent, distance).
test(transitive_parent_distance_parity,
     [setup(kbuild_both([user:kpd/4, user:kedge/2], 'gen.kpd', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'kpd/4', [a,b,a,'1'], "true"),
    ksame(Run, 'kpd/4', [a,c,b,'2'], "true"),
    ksame(Run, 'kpd/4', [a,d,c,'3'], "true"),
    ksame(Run, 'kpd/4', [a,f,e,'2'], "true"),
    ksame(Run, 'kpd/4', [b,d,c,'2'], "true"),
    ksame(Run, 'kpd/4', [a,d,b,'3'], "false"),
    ksame(Run, 'kpd/4', [a,c,a,'2'], "false").

% transitive_step_parent_distance5: (target, first-hop, parent, distance).
test(transitive_step_parent_distance_parity,
     [setup(kbuild_both([user:ksp/5, user:kedge/2], 'gen.ksp', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'ksp/5', [a,b,b,a,'1'], "true"),
    ksame(Run, 'ksp/5', [a,c,b,b,'2'], "true"),
    ksame(Run, 'ksp/5', [a,d,b,c,'3'], "true"),
    ksame(Run, 'ksp/5', [a,f,e,e,'2'], "true"),
    ksame(Run, 'ksp/5', [b,d,c,c,'2'], "true"),
    ksame(Run, 'ksp/5', [a,d,e,c,'3'], "false"),
    ksame(Run, 'ksp/5', [a,d,b,b,'3'], "false").

:- end_tests(wam_scala_kernels_runtime).

% ============================================================
% Helpers
% ============================================================

collect_clauses(Pred, Arity, Clauses) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses).

kscala_available :-
    (   getenv('SCALA_SMOKE_TESTS', "1") -> true
    ;   catch(process_create(path(scalac), ['--version'],
                  [stdout(null), stderr(null), process(Pid)]), _, fail),
        process_wait(Pid, exit(0))
    ).

ktmp(Prefix, Dir) :-
    get_time(T), Stamp is floor(T * 1000),
    format(atom(Dir), '~w_~w', [Prefix, Stamp]).

kprogram_source(Dir, Package, Src) :-
    atom_string(Package, PkgStr),
    split_string(PkgStr, ".", "", Parts),
    atomic_list_concat(Parts, '/', PkgPath),
    format(atom(Rel), 'src/main/scala/~w/GeneratedProgram.scala', [PkgPath]),
    directory_file_path(Dir, Rel, Path),
    read_file_to_string(Path, Src, []).

%% kbuild_both(+Preds, +PkgBase, -run(ItDir, KnDir, ItPkg, KnPkg))
%  Builds the predicates twice: interpreter (no kernel) and kernel mode.
kbuild_both(Preds, PkgBase, run(ItDir, KnDir, ItPkg, KnPkg)) :-
    format(atom(ItPkg), '~w.it.core', [PkgBase]),
    format(atom(KnPkg), '~w.kn.core', [PkgBase]),
    ktmp('tmp_scala_kpit', ItDir),
    ktmp('tmp_scala_kpkn', KnDir),
    write_wam_scala_project(Preds,
        [package(ItPkg), runtime_package(ItPkg), module_name('kpit')], ItDir),
    write_wam_scala_project(Preds,
        [package(KnPkg), runtime_package(KnPkg), module_name('kpkn'),
         kernel_dispatch(true)], KnDir),
    kcompile(ItDir),
    kcompile(KnDir).

ksame(run(ItDir, KnDir, ItPkg, KnPkg), PredKey, Args, Expected) :-
    krun(ItDir, ItPkg, PredKey, Args, ItOut),
    krun(KnDir, KnPkg, PredKey, Args, KnOut),
    (   ItOut == KnOut, ItOut == Expected
    ->  true
    ;   throw(error(kernel_parity_mismatch(PredKey, Args,
                expected(Expected), interp(ItOut), kernel(KnOut)), _))
    ).

kcleanup(run(ItDir, KnDir, _, _)) :-
    ignore(delete_directory_and_contents(ItDir)),
    ignore(delete_directory_and_contents(KnDir)).

kcompile(Dir) :-
    absolute_file_name(Dir, AbsDir),
    directory_file_path(AbsDir, 'classes', ClassDir),
    make_directory_path(ClassDir),
    findall(F,
        ( directory_member(AbsDir, RelF, [extensions([scala]), recursive(true)]),
          directory_file_path(AbsDir, RelF, F) ), Sources),
    Sources \= [],
    process_create(path(scalac), ['-d', ClassDir | Sources],
        [cwd(AbsDir), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, _), read_string(E, _, Err), close(O), close(E),
    process_wait(Pid, exit(Code)),
    ( Code =:= 0 -> true ; throw(error(scala_compile_failed(Dir, Code, Err), _)) ).

krun(Dir, Package, PredKey, Args, Output) :-
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
