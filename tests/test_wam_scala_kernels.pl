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
% Deliberate non-atom key: the native TD3 Source guard must reject it even
% though an otherwise valid outgoing edge exists in the indexed relation.
user:kedge(1, integer_key_target).

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

% category_ancestor: depth-bounded ancestor search with a visited list over
% a separate parent relation. Parent chain c1 -> c2 -> c3 -> c4.
:- dynamic user:kcat_parent/2.
:- dynamic user:kca/4.
:- dynamic user:max_depth/1.
user:max_depth(10).
user:kcat_parent(c1, c2).
user:kcat_parent(c2, c3).
user:kcat_parent(c3, c4).
user:kca(Cat, Parent, 1, Visited) :-
    kcat_parent(Cat, Parent), \+ member(Parent, Visited).
user:kca(Cat, Anc, Hops, Visited) :-
    max_depth(M), length(Visited, D), D < M, !,
    kcat_parent(Cat, Mid), \+ member(Mid, Visited),
    kca(Mid, Anc, H1, [Mid|Visited]), Hops is H1 + 1.

% weighted_shortest_path3: Dijkstra over a ternary weighted edge relation.
% Float weights so interpreter (FloatTerm sums) and kernel (Double) agree.
% Weighted chain a -1.0-> b -2.0-> c -3.0-> d.
:- dynamic user:kwedge/3.
:- dynamic user:kwsp/3.
user:kwedge(a, b, 1.0).
user:kwedge(b, c, 2.0).
user:kwedge(c, d, 3.0).
user:kwsp(X, Y, W) :- kwedge(X, Y, W).
user:kwsp(X, Y, T) :- kwedge(X, Z, W1), kwsp(Z, Y, R), T is R + W1.

% astar_shortest_path4: goal-directed A* over the same float-weighted chain,
% with a heuristic oracle (khdist/3) and Minkowski dimensionality. The plain
% interpreter ignores the heuristic/dim (they're kernel config) and just sums
% weights, so on this single-path chain interpreter and kernel agree.
:- dynamic user:khdist/3.
:- dynamic user:kastar/4.
:- dynamic user:direct_dist_pred/1.
:- dynamic user:dimensionality/1.
user:direct_dist_pred(khdist/3).
user:dimensionality(5).
user:khdist(a, d, 5.0).
user:khdist(b, d, 4.0).
user:khdist(c, d, 3.0).
user:kastar(X, Y, _, W) :- kwedge(X, Y, W).
user:kastar(X, Y, Dim, T) :- kwedge(X, Z, W1), kastar(Z, Y, Dim, R), T is R + W1.

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

test(detects_category_ancestor) :-
    collect_clauses(kca, 4, Clauses),
    detect_recursive_kernel(kca, 4, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(category_ancestor, kca/4, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kcat_parent/2), Cfg)),
    assertion(memberchk(max_depth(10), Cfg)).

test(detects_weighted_shortest_path3) :-
    collect_clauses(kwsp, 3, Clauses),
    detect_recursive_kernel(kwsp, 3, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(weighted_shortest_path3, kwsp/3, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kwedge/3), Cfg)).

test(detects_astar_shortest_path4) :-
    collect_clauses(kastar, 4, Clauses),
    detect_recursive_kernel(kastar, 4, Clauses, Kernel),
    assertion(Kernel = recursive_kernel(astar_shortest_path4, kastar/4, _)),
    Kernel = recursive_kernel(_, _, Cfg),
    assertion(memberchk(edge_pred(kwedge/3), Cfg)),
    assertion(memberchk(direct_dist_pred(khdist/3), Cfg)),
    assertion(memberchk(dimensionality(5), Cfg)).

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
        sub_string(Src, GuardAt, _, _, "args(0) match"),
        sub_string(Src, QueueAt, _, _,
                   "Queue[(WamTerm, Int)]((source, 0))"),
        assertion(GuardAt < QueueAt),
        assertion(sub_string(Src, _, _, _, "case source @ Atom(_) =>")),
        assertion(sub_string(Src, _, _, _, "case _ => ForeignFail")),
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

% category_ancestor kernel mode: kca becomes a CallForeign stub backed by a
% depth-bounded DFS handler (max_depth baked in, visited list parsed via
% wamListToVector); the parent relation stays WAM-compiled.
test(category_ancestor_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_knc', Dir),
        write_wam_scala_project(
            [user:kca/4, user:kcat_parent/2],
            [ package('kc.core'), runtime_package('kc.core'),
              module_name('kc'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'kc.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"kca\", 4)")),
        assertion(sub_string(Src, _, _, _, "maxDepth: Int = 10")),
        assertion(sub_string(Src, _, _, _, "wamListToVector")),
        assertion(sub_string(Src, _, _, _, "\"kcat_parent/2\" ->")),
        delete_directory_and_contents(Dir)
    )).

% weighted_shortest_path3 kernel mode: kwsp becomes a CallForeign stub backed
% by a native Dijkstra handler over the ternary weighted edge relation.
test(weighted_shortest_path_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_knw', Dir),
        write_wam_scala_project(
            [user:kwsp/3, user:kwedge/3],
            [ package('kw.core'), runtime_package('kw.core'),
              module_name('kw'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'kw.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"kwsp\", 3)")),
        assertion(sub_string(Src, _, _, _, "collectTernarySolutions(sharedProgram, \"kwedge/3\")")),
        assertion(sub_string(Src, _, _, _, "PriorityQueue")),
        assertion(sub_string(Src, _, _, _, "\"kwedge/3\" ->")),
        delete_directory_and_contents(Dir)
    )).

% astar_shortest_path4 kernel mode: kastar becomes a CallForeign stub backed
% by a native A* handler reading both the weighted edges and the heuristic
% oracle (khdist/3); both relations stay WAM-compiled.
test(astar_kernel_emits_handler_and_stub) :-
    once((
        ktmp('tmp_scala_kna', Dir),
        write_wam_scala_project(
            [user:kastar/4, user:kwedge/3, user:khdist/3],
            [ package('ka.core'), runtime_package('ka.core'),
              module_name('ka'), kernel_dispatch(true) ],
            Dir),
        kprogram_source(Dir, 'ka.core', Src),
        assertion(sub_string(Src, _, _, _, "CallForeign(\"kastar\", 4)")),
        assertion(sub_string(Src, _, _, _, "collectTernarySolutions(sharedProgram, \"kwedge/3\")")),
        assertion(sub_string(Src, _, _, _, "collectTernarySolutions(sharedProgram, \"khdist/3\")")),
        assertion(sub_string(Src, _, _, _, "def heuristic")),
        assertion(sub_string(Src, _, _, _, "\"kwedge/3\" ->")),
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

test(transitive_distance_rejects_non_atom_source,
     [setup(kbuild_both([user:ktd/3, user:kedge/2], 'gen.ktd_source_gate', Run)),
      cleanup(kcleanup(Run))]) :-
    % Run the native build directly: generic recursive Prolog accepts the
    % integer-key edge, while the strict TD3 native contract requires Atom.
    Run = run(_ItDir, KnDir, _ItPkg, KnPkg),
    krun(KnDir, KnPkg, 'ktd/3', ['1',integer_key_target,'1'], Out),
    assertion(Out == "false").

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

% category_ancestor: depth-bounded ancestor search. The recursive clause
% reads max_depth/1 at runtime, so max_depth/1 is included in the predicate
% list (otherwise the interpreter's max_depth/1 call has no dispatch target).
% With it compiled, interpreter and kernel modes agree and both match SWI
% ground truth. Parent chain c1 -> c2 -> c3 -> c4, max_depth 10.
test(category_ancestor_parity,
     [setup(kbuild_both([user:kca/4, user:kcat_parent/2, user:max_depth/1],
                        'gen.kca', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'kca/4', [c1,c2,'1','[]'], "true"),
    ksame(Run, 'kca/4', [c1,c3,'2','[]'], "true"),
    ksame(Run, 'kca/4', [c1,c4,'3','[]'], "true"),
    ksame(Run, 'kca/4', [c2,c4,'2','[]'], "true"),
    ksame(Run, 'kca/4', [c1,c4,'2','[]'], "false"),
    ksame(Run, 'kca/4', [c1,c2,'2','[]'], "false").

% weighted_shortest_path3: Dijkstra over a float-weighted chain. Single path
% per target, so kernel (shortest weight) and interpreter agree exactly.
test(weighted_shortest_path_parity,
     [setup(kbuild_both([user:kwsp/3, user:kwedge/3], 'gen.kwsp', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'kwsp/3', [a,b,'1.0'], "true"),
    ksame(Run, 'kwsp/3', [a,c,'3.0'], "true"),
    ksame(Run, 'kwsp/3', [a,d,'6.0'], "true"),
    ksame(Run, 'kwsp/3', [b,d,'5.0'], "true"),
    ksame(Run, 'kwsp/3', [a,c,'2.0'], "false"),
    ksame(Run, 'kwsp/3', [a,d,'5.0'], "false").

% astar_shortest_path4: goal-directed A* over the float-weighted chain. Single
% path per target, so kernel (A* shortest) and interpreter (weighted sum) agree.
% khdist/3 is included so the kernel's heuristic-oracle enumeration resolves.
test(astar_parity,
     [setup(kbuild_both([user:kastar/4, user:kwedge/3, user:khdist/3],
                        'gen.kastar', Run)),
      cleanup(kcleanup(Run))]) :-
    ksame(Run, 'kastar/4', [a,b,'5','1.0'], "true"),
    ksame(Run, 'kastar/4', [a,c,'5','3.0'], "true"),
    ksame(Run, 'kastar/4', [a,d,'5','6.0'], "true"),
    ksame(Run, 'kastar/4', [b,d,'5','5.0'], "true"),
    ksame(Run, 'kastar/4', [a,d,'5','5.0'], "false"),
    ksame(Run, 'kastar/4', [a,c,'5','2.0'], "false").

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
