:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_snapshot_multi.pl -- M0 scenario runner for the cross-snapshot
% "newest-compatible" driver resolver_snapshot_multi.pl.
%
% Each scenario builds a TINY hand-made union pool (assert store_pkg / pool_dep
% / pool_conflict directly, via the ms_* asserters), fixes a locked set, calls
% newest_compatible_snap/4, and checks the Result against an ASSERTED
% expectation.
%
%   swipl -q -g main -t halt examples/pkg_resolver/test_snapshot_multi.pl
%
% Exits nonzero if any scenario fails.

:- module(test_snapshot_multi, [main/0]).

:- use_module(resolver_snapshot_multi).

:- dynamic fail_count/1.

% Env with the locked set carried in Base (as Name-Ver pairs). SnapId/PoolId are
% both `pool`; the driver folds SnapId := PoolId to view the union.
mk_env(Locked, env_snap(pool, pool, Locked, [], [], [], [], [])).

% -- reporting -------------------------------------------------------------

pass(Name) :-
    format("PASS  ~w~n", [Name]).

fail_(Name, Why) :-
    format("FAIL  ~w  -- ~w~n", [Name, Why]),
    retract(fail_count(N)),
    N1 is N + 1,
    assertz(fail_count(N1)).

expect_version(Name, Result, ExpV) :-
    (   Result = compatible(ExpV, _)
    ->  pass(Name)
    ;   fail_(Name, got(Result)-expected(compatible(ExpV)))
    ).

expect_version_in(Name, Result, ExpV, InPair) :-
    (   Result = compatible(ExpV, Sel),
        memberchk(InPair, Sel)
    ->  pass(Name)
    ;   fail_(Name, got(Result)-expected(compatible(ExpV)/InPair))
    ).

expect_no_candidate(Name, Result) :-
    (   Result == no_candidate
    ->  pass(Name)
    ;   fail_(Name, got(Result)-expected(no_candidate))
    ).

% -- scenarios -------------------------------------------------------------

% 1. Newest version resolves -> returns it.
s1 :-
    ms_clear,
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version('1 newest resolves', R, v(3,0,0)).

% 2. Newest conflicts with a locked package (conflict table) -> next-newest.
s2 :-
    ms_clear,
    ms_pkg(pool, 'L', v(1,0,0)),
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    ms_conflict(pool, 'P', v(3,0,0), 'L'),
    mk_env(['L'-v(1,0,0)], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version_in('2 newest conflicts locked -> skip', R, v(2,0,0), 'L'-v(1,0,0)).

% 3. ALL versions incompatible with the locked set -> no_candidate, fast.
s3 :-
    ms_clear,
    ms_pkg(pool, 'L', v(1,0,0)),
    forall(member(N, [1,2,3,4,5]),
           ( ms_pkg(pool, 'P', v(N,0,0)),
             ms_conflict(pool, 'P', v(N,0,0), 'L') )),
    mk_env(['L'-v(1,0,0)], Env),
    get_time(T0),
    newest_compatible_snap(Env, 'P', any, R),
    get_time(T1),
    D is T1 - T0,
    format("      [scenario 3 elapsed: ~4f s]~n", [D]),
    expect_no_candidate('3 all incompatible -> no_candidate', R),
    (   D < 1.0
    ->  pass('3 all incompatible terminates fast (<1s)')
    ;   fail_('3 all incompatible terminates fast (<1s)', slow(D))
    ).

% 4. Constraint lt(v(3,0,0)) -> newest UNDER the constraint.
s4 :-
    ms_clear,
    forall(member(N, [1,2,3,4,5]), ms_pkg(pool, 'P', v(N,0,0))),
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', lt(v(3,0,0)), R),
    expect_version('4 constraint lt(3.0.0) -> newest under', R, v(2,0,0)).

% 5. Newest needs a new dependency that IS available & compatible -> newest.
s5 :-
    ms_clear,
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    ms_pkg(pool, 'Q', v(1,0,0)),
    ms_dep(pool, 'P', v(3,0,0), dep('Q', gte(v(1,0,0)))),
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version_in('5 newest pulls available dep', R, v(3,0,0), 'Q'-v(1,0,0)).

% 6. Newest needs a dep that conflicts with the locked set -> skip to older.
s6 :-
    ms_clear,
    ms_pkg(pool, 'L', v(1,0,0)),
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    ms_dep(pool, 'P', v(3,0,0), dep('L', gte(v(2,0,0)))),  % locked L=1 can't satisfy
    ms_dep(pool, 'P', v(2,0,0), dep('L', gte(v(1,0,0)))),  % satisfied by locked L=1
    mk_env(['L'-v(1,0,0)], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version_in('6 newest dep conflicts locked -> skip', R, v(2,0,0), 'L'-v(1,0,0)).

% 7. Union spanning snapshots: P v1 came from snap0, v3 from snap5; both live in
%    the union under PoolId -> newest across the union is considered.
s7 :-
    ms_clear,
    ms_pkg(pool, 'P', v(1,0,0)),   % (snap0)
    ms_pkg(pool, 'P', v(3,0,0)),   % (snap5)
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version('7 union across snapshots', R, v(3,0,0)).

% 8. Empty locked set: newest version has an unsatisfiable dep -> newest that
%    actually resolves.
s8 :-
    ms_clear,
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    ms_dep(pool, 'P', v(3,0,0), dep('X', any)),   % X absent from the union
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version('8 empty lock -> newest resolvable', R, v(2,0,0)).

% 9. Single version available & compatible -> returns it.
s9 :-
    ms_clear,
    ms_pkg(pool, 'P', v(1,0,0)),
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version('9 single version compatible', R, v(1,0,0)).

% 10. deb(...) versions -> newest by deb ordering.
s10 :-
    ms_clear,
    ms_pkg(pool, 'P', deb(0, [s([],1)], [])),
    ms_pkg(pool, 'P', deb(0, [s([],2)], [])),
    ms_pkg(pool, 'P', deb(0, [s([],3)], [])),
    mk_env([], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version('10 deb ordering -> newest', R, deb(0, [s([],3)], [])).

% 11. Single version, incompatible with the locked set -> no_candidate.
s11 :-
    ms_clear,
    ms_pkg(pool, 'L', v(1,0,0)),
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_conflict(pool, 'P', v(1,0,0), 'L'),
    mk_env(['L'-v(1,0,0)], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_no_candidate('11 single version incompatible', R).

% 12. Multi-skip: v3 conflicts (table), v2 dep unsatisfiable vs lock, v1 ok.
s12 :-
    ms_clear,
    ms_pkg(pool, 'L', v(1,0,0)),
    ms_pkg(pool, 'P', v(1,0,0)),
    ms_pkg(pool, 'P', v(2,0,0)),
    ms_pkg(pool, 'P', v(3,0,0)),
    ms_conflict(pool, 'P', v(3,0,0), 'L'),
    ms_dep(pool, 'P', v(2,0,0), dep('L', gte(v(2,0,0)))),  % fails vs locked L=1
    ms_dep(pool, 'P', v(1,0,0), dep('L', gte(v(1,0,0)))),  % ok
    mk_env(['L'-v(1,0,0)], Env),
    newest_compatible_snap(Env, 'P', any, R),
    expect_version_in('12 multi-skip to oldest compatible', R, v(1,0,0), 'L'-v(1,0,0)).

% -- driver ----------------------------------------------------------------

main :-
    retractall(fail_count(_)),
    assertz(fail_count(0)),
    forall(member(G, [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]),
           run_scenario(G)),
    fail_count(F),
    nl,
    (   F =:= 0
    ->  format("ALL SCENARIOS PASSED~n", [])
    ;   format("~w SCENARIO CHECK(S) FAILED~n", [F]),
        halt(1)
    ).

run_scenario(G) :-
    (   catch(call(G), E, (print_message(error, E), fail))
    ->  true
    ;   fail_(G, threw_or_failed)
    ).
